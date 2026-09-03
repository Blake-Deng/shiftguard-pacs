#!/usr/bin/env python3
"""Run the frozen corrected ShiftGuard ablation matrix.

All configurations use the same source split, strong/weak views, optimizer and
checkpoint protocol. Target is evaluated once per completed run, after source
validation selects the checkpoint.
"""
from __future__ import annotations
import json, os, queue, statistics, subprocess, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON=sys.executable
DOMAINS=['Photo','Art_Painting','Cartoon','Sketch']
SEEDS=[42,123,3407]
GPUS=[0,1,2]
OUT=ROOT/'runs/corrected_ablation'
LOG=ROOT/'logs/corrected_ablation'
CONFIGS=[
 {'run_name':'strong_aug','method':'aug','lambda_feat':0.0,'lambda_kl':0.0},
 {'run_name':'one_way_kl','method':'kl','lambda_feat':0.0,'lambda_kl':0.05},
 {'run_name':'feature_plus_kl','method':'feat_kl','lambda_feat':0.10,'lambda_kl':0.05},
 {'run_name':'adaptive','method':'adaptive','lambda_feat':0.10,'lambda_kl':0.05},
]

def stem(t): return f"{t['target']}_resnet50_{t['run_name']}_seed{t['seed']}"
def cmd(t): return [PYTHON,str(ROOT/'shiftguard_corrected.py'),'--data-root',str(ROOT/'data/PACS'),'--target',t['target'],'--method',t['method'],'--run-name',t['run_name'],'--model','resnet50','--seed',str(t['seed']),'--epochs','30','--batch-size','64','--workers','8','--lambda-feat',str(t['lambda_feat']),'--lambda-kl',str(t['lambda_kl']),'--temperature','2.0','--gate-tau','0.5','--warmup-epochs','5','--device','cuda:0','--output',str(t['output']),'--save-checkpoint']
def tasks():
 out=[]
 for c in CONFIGS:
  for d in DOMAINS:
   for s in SEEDS: out.append({**c,'target':d,'seed':s,'output':OUT/'tasks'/c['run_name']/d/str(s)})
 return out

def run_all(ts):
 q=queue.Queue(); failures=[]; lock=threading.Lock(); LOG.mkdir(parents=True,exist_ok=True)
 for t in ts:
  t['output'].mkdir(parents=True,exist_ok=True)
  if not (t['output']/(stem(t)+'.json')).exists(): q.put(t)
 def worker(gpu):
  while True:
   try:t=q.get_nowait()
   except queue.Empty:return
   p=LOG/(stem(t)+'.log'); e=os.environ.copy(); e['CUDA_VISIBLE_DEVICES']=str(gpu); e['PYTHONUNBUFFERED']='1'
   print(f'[gpu {gpu}] start {stem(t)}',flush=True); start=time.time()
   with p.open('w') as h:
    h.write('COMMAND: '+' '.join(cmd(t))+'\n'); h.flush()
    rc=subprocess.run(cmd(t),cwd=ROOT,env=e,stdout=h,stderr=subprocess.STDOUT).returncode
   print(f'[gpu {gpu}] done rc={rc} min={(time.time()-start)/60:.1f} {stem(t)}',flush=True)
   if rc: 
    with lock: failures.append((stem(t),rc,str(p)))
   q.task_done()
 th=[threading.Thread(target=worker,args=(g,)) for g in GPUS]
 [x.start() for x in th]; [x.join() for x in th]
 if failures: raise RuntimeError(failures)

def summarize():
 summary=[]
 for c in CONFIGS:
  items=[]
  for d in DOMAINS:
   for s in SEEDS:
    p=OUT/'tasks'/c['run_name']/d/str(s)/(f'{d}_resnet50_{c["run_name"]}_seed{s}.json')
    x=json.loads(p.read_text()); assert x['target_evaluations']==1 and x['target_accuracy'] is not None
    items.append(x)
  per={}
  for d in DOMAINS:
   v=[100*x['target_accuracy'] for x in items if x['target']==d]
   per[d]={'mean':statistics.mean(v),'std':statistics.stdev(v)}
  macros=[statistics.mean([100*x['target_accuracy'] for x in items if x['seed']==s]) for s in SEEDS]
  summary.append({'run_name':c['run_name'],'method':c['method'],'lambda_feat':c['lambda_feat'],'lambda_kl':c['lambda_kl'],'per_domain':per,'macro_by_seed':macros,'macro_mean':statistics.mean(macros),'macro_std':statistics.stdev(macros)})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)

def main():
 start=time.time(); run_all(tasks()); summarize(); print('COMPLETE minutes',round((time.time()-start)/60,1),flush=True)
if __name__=='__main__':main()
