"""Compare greedy cached/uncached decoding on the same checkpoint."""
import argparse,json,time
import torch
from generate import load

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/tiny/best.pt'); p.add_argument('--device',default='cpu'); p.add_argument('--tokens',type=int,default=64); p.add_argument('--repeats',type=int,default=3); a=p.parse_args()
    if a.tokens<1 or a.repeats<1: p.error('tokens and repeats must be positive')
    torch.set_num_threads(2); m,tok,_=load(a.checkpoint,a.device)
    x=torch.tensor([tok.encode('Once upon a time').ids],device=a.device)
    def sync():
        if a.device.startswith('cuda'): torch.cuda.synchronize()
    report={}; outputs=[]
    for cache in [False,True]:
        m.generate(x,max_new_tokens=4,top_k=1,use_cache=cache)
        sync(); start=time.perf_counter()
        for _ in range(a.repeats): y=m.generate(x,max_new_tokens=a.tokens,top_k=1,use_cache=cache)
        sync(); elapsed=time.perf_counter()-start
        report['cached' if cache else 'uncached']={'tokens_per_second':a.tokens*a.repeats/elapsed,'seconds':elapsed}
        outputs.append(y)
    report['identical_greedy_tokens']=torch.equal(*outputs)
    report['note']='Window rebuilds after context limit; speed is hardware and prompt dependent.'
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
