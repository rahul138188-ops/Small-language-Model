"""Portable float32 training; saves portable inference and resumable checkpoints."""
import argparse, csv, hashlib, json, math, random, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from model import Config, StoryGPT

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/tinystories'); p.add_argument('--out',default='checkpoints/my_run')
    p.add_argument('--config',default='configs/tiny.json'); p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--eval-every',type=int,default=250); p.add_argument('--eval-batches',type=int,default=20)
    p.add_argument('--device',default='auto'); p.add_argument('--seed',type=int,default=42); p.add_argument('--resume'); p.add_argument('--label',default='tinystories-training')
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.accumulation,a.eval_every,a.eval_batches)<=0: p.error('Step and batch counts must be positive')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(2)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    data=Path(a.data); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume: raise FileExistsError('Use a new output folder or --resume')
    meta=json.loads((data/'metadata.json').read_text()); cfg=json.loads(Path(a.config).read_text()); cfg['vocab_size']=meta['vocab_size']; c=Config(**cfg)
    if hashlib.sha256((data/'tokenizer.json').read_bytes()).hexdigest()!=meta['tokenizer_sha256']: raise ValueError('Tokenizer checksum mismatch')
    arrays={s:np.load(data/f'{s}.npy',mmap_mode='r') for s in ('train','val')}
    if any(len(x)<=c.context_length for x in arrays.values()): raise ValueError('Each split must exceed context length')
    rng=torch.Generator().manual_seed(a.seed); vrng=torch.Generator().manual_seed(a.seed+1)
    def batch(split):
        arr=arrays[split]; starts=torch.randint(len(arr)-c.context_length,(a.batch_size,),generator=rng if split=='train' else vrng)
        z=torch.tensor(np.stack([arr[int(i):int(i)+c.context_length+1] for i in starts]),dtype=torch.long,device=device)
        return z[:,:-1],z[:,1:]
    model=StoryGPT(c).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.1)
    start=0; best=float('inf')
    if a.resume:
        old=torch.load(a.resume,map_location='cpu',weights_only=True)
        if old['config']!=asdict(c) or old['data']['tokenizer_sha256']!=meta['tokenizer_sha256']: raise ValueError('Resume config/tokenizer mismatch')
        model.load_state_dict(old['model']); opt.load_state_dict(old['optimizer']); start=old['step']; best=old['best_val']
        rng.set_state(old['batch_rng']); torch.set_rng_state(old['torch_rng'])
        if device.startswith('cuda') and old.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(old['cuda_rng'])
    def save(name,step,val):
        payload=dict(config=asdict(c),model={k:v.detach().cpu() for k,v in model.state_dict().items()},tokenizer=(data/'tokenizer.json').read_text(),data=meta,step=step,val_loss=val,best_val=best,label=a.label,parameters=sum(x.numel() for x in model.parameters()))
        if name=='last.pt': payload.update(optimizer=opt.state_dict(),batch_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
        temp=out/(name+'.tmp'); torch.save(payload,temp); temp.replace(out/name)
    print(f'Device={device}; parameters={sum(x.numel() for x in model.parameters()):,}',flush=True)
    began=time.time(); log=out/'metrics.csv'; exists=log.exists()
    with log.open('a',newline='') as f:
        writer=csv.writer(f)
        if not exists: writer.writerow(['step','train_loss','val_loss','perplexity','elapsed_seconds'])
        for step in range(start+1,a.steps+1):
            model.train(); opt.zero_grad(set_to_none=True); total=0.0
            warmup=min(100,max(1,a.steps//10)); progress=max(0,(step-warmup)/max(1,a.steps-warmup))
            lr=a.lr*min(1,step/warmup)*(0.1+0.9*0.5*(1+math.cos(math.pi*progress)))
            for group in opt.param_groups: group['lr']=lr
            for _ in range(a.accumulation):
                x,y=batch('train'); _,loss,_=model(x,y); total+=loss.item()/a.accumulation; (loss/a.accumulation).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if step%a.eval_every==0 or step==a.steps:
                model.eval(); vrng.manual_seed(a.seed+1)
                with torch.inference_mode():
                    val=sum(model(*batch('val'))[1].item() for _ in range(a.eval_batches))/a.eval_batches
                if val<best: best=val; save('best.pt',step,val)
                save('last.pt',step,val); writer.writerow([step,total,val,math.exp(min(val,80)),time.time()-began]); f.flush()
                print(f'step={step} train={total:.4f} val={val:.4f}',flush=True)
    (out/'run.json').write_text(json.dumps(dict(arguments=vars(a),device=device,elapsed_seconds=time.time()-began,parameters=sum(x.numel() for x in model.parameters()),status='completed'),indent=2))
if __name__=='__main__': main()
