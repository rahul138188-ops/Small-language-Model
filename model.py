"""Small decoder-only transformer with optional learned positions or RoPE."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int = 512
    context_length: int = 256
    dim: int = 192
    heads: int = 3
    layers: int = 3
    dropout: float = 0.1
    position: str = 'rope'
    def __post_init__(self):
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError('Head dimension must be an even integer')
        if self.position not in ('rope', 'learned'):
            raise ValueError('position must be rope or learned')
        if min(self.vocab_size,self.context_length,self.dim,self.heads,self.layers)<=0:
            raise ValueError('Dimensions must be positive')

def rotate(x, offset):
    d = x.size(-1)
    inv = 10000.0 ** (-torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d)
    angles = torch.arange(offset,offset+x.size(-2),device=x.device).float()[:,None]*inv
    c,s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a,b = x[...,0::2],x[...,1::2]
    return torch.stack((a*c-b*s,a*s+b*c),dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.qkv=nn.Linear(c.dim,3*c.dim); self.proj=nn.Linear(c.dim,c.dim)
        self.drop=nn.Dropout(c.dropout)
    def forward(self,x,past=None,use_cache=False):
        b,t,d=x.shape; h=self.c.heads
        q,k,v=[z.reshape(b,t,h,d//h).transpose(1,2) for z in self.qkv(x).chunk(3,-1)]
        offset=0 if past is None else past[0].size(-2)
        if self.c.position=='rope': q,k=rotate(q,offset),rotate(k,offset)
        if past is not None:
            k=torch.cat((past[0],k),dim=-2); v=torch.cat((past[1],v),dim=-2)
        # Explicit offset-aware causal mask also handles cached multi-token chunks.
        mask=torch.arange(k.size(-2),device=x.device)[None,:] <= torch.arange(offset,offset+t,device=x.device)[:,None]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.c.dropout if self.training else 0.0)
        y=self.drop(self.proj(y.transpose(1,2).contiguous().view(b,t,d)))
        return y, (k,v) if use_cache else None

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.norm1=nn.LayerNorm(c.dim); self.norm2=nn.LayerNorm(c.dim)
        self.attn=Attention(c)
        self.ff=nn.Sequential(nn.Linear(c.dim,4*c.dim),nn.GELU(),nn.Linear(4*c.dim,c.dim),nn.Dropout(c.dropout))
    def forward(self,x,past=None,use_cache=False):
        y,cache=self.attn(self.norm1(x),past,use_cache); x=x+y
        return x+self.ff(self.norm2(x)),cache

class StoryGPT(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        self.tokens=nn.Embedding(c.vocab_size,c.dim)
        self.positions=nn.Embedding(c.context_length,c.dim) if c.position=='learned' else None
        self.drop=nn.Dropout(c.dropout); self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm=nn.LayerNorm(c.dim); self.head=nn.Linear(c.dim,c.vocab_size,bias=False)
        self.apply(self._init); self.head.weight=self.tokens.weight
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,ids,targets=None,past=None,use_cache=False):
        offset=0 if past is None else past[0][0].size(-2)
        if ids.size(1)+offset>self.config.context_length: raise ValueError('Context limit exceeded')
        x=self.tokens(ids)
        if self.positions is not None: x=x+self.positions(torch.arange(offset,offset+ids.size(1),device=ids.device))
        x=self.drop(x); caches=[]
        for i,block in enumerate(self.blocks):
            x,cache=block(x,None if past is None else past[i],use_cache); caches.append(cache)
        logits=self.head(self.norm(x))
        loss=None if targets is None else F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss,caches if use_cache else None
    @torch.inference_mode()
    def generate(self,ids,max_new_tokens=100,temperature=0.8,top_k=40,top_p=0.95,use_cache=True,eos_id=None):
        if temperature<=0 or not 0<top_p<=1 or top_k<0 or max_new_tokens<0:
            raise ValueError('Invalid sampling arguments')
        if ids.ndim!=2 or ids.size(0)!=1 or ids.size(1)==0: raise ValueError('Provide one nonempty prompt')
        was_training=self.training; self.eval(); cache=None
        try:
            for _ in range(max_new_tokens):
                # At the context boundary rebuild the window; do not silently change positions.
                if cache is not None and cache[0][0].size(-2)<self.config.context_length:
                    inp=ids[:,-1:]
                else:
                    inp=ids[:,-self.config.context_length:]; cache=None
                logits,_,cache=self(inp,past=cache,use_cache=use_cache)
                z=logits[:,-1,:]/temperature
                if top_k: z=z.masked_fill(z<torch.topk(z,min(top_k,z.size(-1))).values[:,-1:],float('-inf'))
                if top_p<1:
                    vals,idx=z.sort(descending=True); remove=vals.softmax(-1).cumsum(-1)>top_p
                    remove[:,1:]=remove[:,:-1].clone(); remove[:,0]=False
                    z=z.masked_fill(torch.zeros_like(remove).scatter(1,idx,remove),float('-inf'))
                nxt=torch.multinomial(z.softmax(-1),1); ids=torch.cat((ids,nxt),dim=1)
                if eos_id is not None and nxt.item()==eos_id: break
            return ids
        finally: self.train(was_training)
