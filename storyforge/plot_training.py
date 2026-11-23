"""Plot actual logged losses; never synthesizes results."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',default='checkpoints/tiny/metrics.csv'); p.add_argument('--out',default='docs/training_loss.png'); a=p.parse_args()
    with open(a.csv) as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError('Training log contains no measurements')
    steps=[int(r['step']) for r in rows]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(steps,[float(r['train_loss']) for r in rows],label='Training (logged step)')
    ax.plot(steps,[float(r['val_loss']) for r in rows],label='Validation (fixed sampled windows)')
    ax.set(xlabel='Optimizer step',ylabel='Cross-entropy loss',title='StoryForge training measurements')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.out,dpi=160); plt.close(fig)
if __name__=='__main__': main()
