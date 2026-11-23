"""Train byte-level BPE on training text only and save separate token arrays."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def prepare(train,val,out,vocab_size=512,source='local text',tokenizer_path=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists(): raise FileExistsError('Choose a fresh --out directory to avoid mixing tokenizers and checkpoints')
    if vocab_size<257: raise ValueError('Byte BPE needs at least 257 vocabulary entries')
    train=list(dict.fromkeys(s.strip() for s in train if s.strip()))
    train_set=set(train); val=list(dict.fromkeys(s.strip() for s in val if s.strip() and s.strip() not in train_set))
    if not train or not val: raise ValueError('Need nonempty, disjoint train and validation stories')
    if tokenizer_path:
        tok=Tokenizer.from_file(str(tokenizer_path))
        if tok.token_to_id('<|endoftext|>') is None: raise ValueError('Tokenizer needs an end-of-text token')
        # Preserve exact serialized bytes, so resume's checksum remains meaningful.
        (out/'tokenizer.json').write_bytes(Path(tokenizer_path).read_bytes())
    else:
        tok=Tokenizer(models.BPE()); tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder=decoders.ByteLevel()
        trainer=trainers.BpeTrainer(vocab_size=vocab_size,special_tokens=['<|endoftext|>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(train,trainer=trainer); tok.save(str(out/'tokenizer.json'))
    eos=tok.token_to_id('<|endoftext|>'); counts={}
    for name,stories in [('train',train),('val',val)]:
        ids=[]
        for story in stories: ids.extend(tok.encode(story).ids+[eos])
        np.save(out/f'{name}.npy',np.asarray(ids,dtype=np.int32)); counts[name]=len(ids)
    meta=dict(source=source,vocab_size=tok.get_vocab_size(),tokens=counts,stories=dict(train=len(train),val=len(val)),tokenizer_sha256=hashlib.sha256((out/'tokenizer.json').read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='data/tinystories'); p.add_argument('--vocab-size',type=int,default=512)
    p.add_argument('--train-stories',type=int,default=100000); p.add_argument('--val-stories',type=int,default=5000)
    p.add_argument('--tokenizer',help='Reuse an existing tokenizer for checkpoint continuation'); p.add_argument('--train-file'); p.add_argument('--val-file'); a=p.parse_args()
    if a.train_file:
        if not a.val_file: p.error('--val-file required with --train-file')
        tr=Path(a.train_file).read_text(encoding='utf-8').split('<|endoftext|>'); va=Path(a.val_file).read_text(encoding='utf-8').split('<|endoftext|>'); source='user-supplied text'
    else:
        from datasets import load_dataset
        from itertools import islice
        # Disable background parquet prefetch: early-stopped streams must close cleanly.
        import pyarrow.dataset as pads
        ds=load_dataset('roneneldan/TinyStories',streaming=True,fragment_scan_options=pads.ParquetFragmentScanOptions(pre_buffer=False))
        tr=[x['text'] for x in islice(ds['train'],a.train_stories)]; va=[x['text'] for x in islice(ds['validation'],a.val_stories)]; source='roneneldan/TinyStories'
    prepare(tr,va,a.out,a.vocab_size,source,a.tokenizer)
if __name__=='__main__': main()
