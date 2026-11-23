import argparse
import torch
from tokenizers import Tokenizer
from model import Config, StoryGPT

def load(checkpoint,device='cpu'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=StoryGPT(Config(**ck['config'])).to(device); model.load_state_dict(ck['model']); model.eval()
    return model,Tokenizer.from_str(ck['tokenizer']),ck

def complete(model,tok,prompt,**kwargs):
    ids=tok.encode(prompt).ids or [tok.token_to_id('<|endoftext|>')]
    x=torch.tensor([ids],dtype=torch.long,device=next(model.parameters()).device)
    result=model.generate(x,eos_id=tok.token_to_id('<|endoftext|>'),**kwargs)
    return tok.decode(result[0].tolist())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='checkpoints/tiny/best.pt'); p.add_argument('--prompt',default='Once upon a time')
    p.add_argument('--max-tokens',type=int,default=100); p.add_argument('--temperature',type=float,default=.8); p.add_argument('--top-k',type=int,default=40); p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--device',default='cpu'); p.add_argument('--no-cache',action='store_true'); a=p.parse_args()
    model,tok,meta=load(a.checkpoint,a.device); print('Checkpoint:',meta['label'],'step:',meta['step'])
    print(complete(model,tok,a.prompt,max_new_tokens=a.max_tokens,temperature=a.temperature,top_k=a.top_k,top_p=a.top_p,use_cache=not a.no_cache))
if __name__=='__main__': main()
