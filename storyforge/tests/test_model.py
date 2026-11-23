import torch
from model import Config,StoryGPT

def make(position='rope'):
    torch.manual_seed(5)
    return StoryGPT(Config(vocab_size=32,dim=24,heads=3,layers=2,context_length=16,dropout=0,position=position)).eval()

def test_causal():
    m=make(); x=torch.randint(32,(1,8)); y=x.clone(); y[:,5:]=(y[:,5:]+1)%32
    torch.testing.assert_close(m(x)[0][:,:5],m(y)[0][:,:5])

def test_cache_equivalence():
    for pos in ['rope','learned']:
        m=make(pos); x=torch.randint(32,(1,8)); full=m(x)[0]
        _,_,cache=m(x[:,:3],use_cache=True)
        z,_,_=m(x[:,3:],past=cache,use_cache=True)
        torch.testing.assert_close(z,full[:,3:],rtol=1e-4,atol=1e-5)

def test_backward_and_weight_tying():
    m=make(); x=torch.randint(32,(2,8)); _,loss,_=m(x,x); loss.backward()
    assert m.head.weight is m.tokens.weight
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_parameter_saving():
    a,b=make('learned'),make('rope')
    assert sum(p.numel() for p in a.parameters())-sum(p.numel() for p in b.parameters())==16*24

def test_generation_cache_and_window():
    m=make(); x=torch.randint(32,(1,14))
    torch.manual_seed(10); a=m.generate(x,max_new_tokens=6,top_k=1,use_cache=True)
    torch.manual_seed(10); b=m.generate(x,max_new_tokens=6,top_k=1,use_cache=False)
    assert torch.equal(a,b)
