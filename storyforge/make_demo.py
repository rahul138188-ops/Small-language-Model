"""Offline integration fixture: not a TinyStories-trained language model."""
from prepare_data import prepare
from pathlib import Path
import json
names=['Mia','Leo','Ava','Ben','Zoe','Sam','Ada','Tom']
objects=['ball','kite','book','boat','hat','box','bell','cup']
tr=[]; va=[]
for i,name in enumerate(names):
    for j,obj in enumerate(objects):
        story=f'{name} found a little {obj} in the garden. The sun was warm. {name} showed the {obj} to a friend. They played together and went home happy.'
        (va if (i+j)%7==0 else tr).append(story)
prepare(tr,va,'data/demo',512,'synthetic template fixture; pipeline test only')
# Same full-size architecture, shorter contexts only for the offline integration run.
c=json.loads(Path('configs/tiny.json').read_text()); c['context_length']=64
Path('configs/demo.json').write_text(json.dumps(c,indent=2))
