#!/usr/bin/env python3
"""Three-seed corruption evaluation for corrected PACS checkpoints."""
from __future__ import annotations
import argparse, csv, io
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from shiftguard_corrected import build_model, collect_target_samples, collect_source_samples, IMAGENET_MEAN, IMAGENET_STD

KINDS=('gaussian_noise','blur','brightness','contrast','jpeg')
class CorruptDataset(Dataset):
 def __init__(self,samples,tf,kind,severity): self.samples,self.tf,self.kind,self.severity=samples,tf,kind,severity
 def __len__(self): return len(self.samples)
 def __getitem__(self,index):
  path,label=self.samples[index]
  with Image.open(path) as source: image=source.convert('RGB')
  s=self.severity
  if self.kind=='gaussian_noise':
   array=np.asarray(image).astype(np.float32); rng=np.random.default_rng(index+s*100)
   image=Image.fromarray(np.uint8(np.clip(array+rng.normal(0,8*s,array.shape),0,255)))
  elif self.kind=='blur': image=image.filter(ImageFilter.GaussianBlur(.6*s))
  elif self.kind=='brightness': image=ImageEnhance.Brightness(image).enhance(1+.12*s)
  elif self.kind=='contrast': image=ImageEnhance.Contrast(image).enhance(max(.1,1-.12*s))
  elif self.kind=='jpeg':
   buffer=io.BytesIO(); image.save(buffer,format='JPEG',quality=max(8,100-18*s)); buffer.seek(0); image=Image.open(buffer).convert('RGB')
  return self.tf(image),label
@torch.no_grad()
def accuracy(model,loader,device):
 model.eval(); correct=total=0
 for images,labels in loader:
  logits=model(images.to(device,non_blocking=True)); correct+=(logits.argmax(1).cpu()==labels).sum().item(); total+=labels.numel()
 return correct/total

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--output',required=True); args=ap.parse_args()
 root=Path('data/PACS'); _,_,classes=collect_source_samples(root,'Sketch',args.seed,.15); samples=collect_target_samples(root,'Sketch',classes)
 tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
 device=torch.device(args.device); rows=[]
 for name in ('strong_aug','feature_plus_kl'):
  checkpoint=Path('runs/corrected_ablation/tasks')/name/'Sketch'/str(args.seed)/f'Sketch_resnet50_{name}_seed{args.seed}.pt'
  payload=torch.load(checkpoint,map_location='cpu',weights_only=False); model=build_model('resnet50',7,False); model.load_state_dict(payload['model']); model.to(device)
  for kind in KINDS:
   for severity in range(1,6):
    loader=DataLoader(CorruptDataset(samples,tf,kind,severity),batch_size=128,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)
    value=accuracy(model,loader,device); rows.append({'seed':args.seed,'method':name,'corruption':kind,'severity':severity,'accuracy':value}); print(args.seed,name,kind,severity,value,flush=True)
 out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as handle: writer=csv.DictWriter(handle,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
if __name__=='__main__': main()
