import json
import numpy as np
import pytest
import torch
from prepare_data import prepare
from model import Config, StoryGPT
from generate import load

def test_data_splits_and_checkpoint_roundtrip(tmp_path):
    prepare(['A cat runs.','A cat runs.','A dog walks.'],['A cat runs.','A bird flies.'],tmp_path,300,'test')
    meta=json.loads((tmp_path/'metadata.json').read_text())
    assert meta['stories']=={'train':2,'val':1}
    assert np.load(tmp_path/'train.npy').dtype==np.int32
    with pytest.raises(FileExistsError): prepare(['a'],['b'],tmp_path)
    cfg=Config(vocab_size=meta['vocab_size'],dim=24,heads=3,layers=1,context_length=16,dropout=0)
    model=StoryGPT(cfg).eval()
    from dataclasses import asdict
    ck=tmp_path/'test.pt'
    torch.save({'config':asdict(cfg),'model':model.state_dict(),'tokenizer':(tmp_path/'tokenizer.json').read_text()},ck)
    restored,tok,_=load(ck)
    x=torch.tensor([tok.encode('cat').ids])
    torch.testing.assert_close(model(x)[0],restored(x)[0])


def test_reused_tokenizer_checksum(tmp_path):
    first=tmp_path/'first'; second=tmp_path/'second'
    prepare(['A cat runs.'],['A dog walks.'],first,300,'test')
    prepare(['A red fox.'],['A green bird.'],second,300,'test',first/'tokenizer.json')
    a=json.loads((first/'metadata.json').read_text())
    b=json.loads((second/'metadata.json').read_text())
    assert a['tokenizer_sha256']==b['tokenizer_sha256']
