import argparse, csv, io, json
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from shiftguard_v2 import collect_samples, build_model, DOMAINS

class CorruptDataset(Dataset):
    def __init__(self, samples, tf, kind, severity): self.samples,self.tf,self.kind,self.severity=samples,tf,kind,severity
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path,y=self.samples[i]
        with Image.open(path) as im: im=im.convert('RGB')
        s=self.severity
        if self.kind=='gaussian_noise':
            a=np.asarray(im).astype(np.float32)+np.random.default_rng(i+s*100).normal(0,8*s,np.asarray(im).shape)
            im=Image.fromarray(np.uint8(np.clip(a,0,255)))
        elif self.kind=='blur': im=im.filter(ImageFilter.GaussianBlur(0.6*s))
        elif self.kind=='brightness': im=ImageEnhance.Brightness(im).enhance(1+0.12*s)
        elif self.kind=='contrast': im=ImageEnhance.Contrast(im).enhance(max(0.1,1-0.12*s))
        elif self.kind=='jpeg':
            b=io.BytesIO(); im.save(b,format='JPEG',quality=max(8,100-18*s)); b.seek(0); im=Image.open(b).convert('RGB')
        return self.tf(im), y

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); good=total=0
    for x,y in loader:
        z=model(x.to(device)); good += (z.argmax(1).cpu()==y).sum().item(); total += y.numel()
    return good/total

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='data/PACS'); ap.add_argument('--target',default='Sketch'); ap.add_argument('--output',default='runs/robustness.csv'); args=ap.parse_args()
    _,_,test_s,_=collect_samples(Path(args.data_root),args.target,42,0.15)
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]
    for method in ('aug','shiftguard'):
        ck=Path('runs/ablation')/f'{args.target}_resnet50_{method}_seed42.pt'
        if not ck.exists(): raise FileNotFoundError(ck)
        payload=torch.load(ck,map_location='cpu',weights_only=False); model=build_model('resnet50',7,False); model.load_state_dict(payload['model']); model.to(device)
        for kind in ('gaussian_noise','blur','brightness','contrast','jpeg'):
            for severity in range(1,6):
                loader=DataLoader(CorruptDataset(test_s,tf,kind,severity),batch_size=64,shuffle=False,num_workers=4,pin_memory=device.type=='cuda')
                acc=evaluate(model,loader,device); rows.append({'target':args.target,'method':method,'corruption':kind,'severity':severity,'accuracy':acc}); print(method,kind,severity,acc,flush=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
