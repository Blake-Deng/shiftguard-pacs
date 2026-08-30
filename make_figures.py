import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT=Path('figures'); OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
NAVY='#18324B'; BLUE='#2878B5'; TEAL='#2A9D8F'; ORANGE='#E69F00'; RED='#D55E00'; GRAY='#667085'; LIGHT='#F3F6F8'; DARK='#1D2939'
DOMAINS=['Photo','Art_Painting','Cartoon','Sketch']; LABELS=['Photo','Art painting','Cartoon','Sketch']

def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.04)
 plt.close(fig)

def load_dir(dirname):
 unique={}
 base=Path(dirname) if dirname == 'runs' else Path('runs', dirname)
 for p in base.glob('*.json'):
  d=json.loads(p.read_text()); unique[(d['target'],d['method'],d['seed'])]=d['target_accuracy']*100
 g=defaultdict(list)
 for (target,method,_seed), value in unique.items(): g[(target,method)].append(value)
 return g

# Figure 1: method schematic
fig,ax=plt.subplots(figsize=(7.2,3.0)); ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis('off')
def box(x,y,w,h,text,fc=LIGHT,ec='#B7C4CE',fs=9,weight='normal'):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.08',fc=fc,ec=ec,lw=1.2); ax.add_patch(p)
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=DARK,weight=weight,wrap=True)
def arrow(x1,y1,x2,y2,color=GRAY,style='-|>',lw=1.3):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=10,color=color,lw=lw))
box(.25,2.35,1.25,.75,'source image\n$x,y$',fc='#EAF2F8',ec=BLUE,weight='bold')
box(.25,.85,1.25,.75,'weak view\n$x_w$',fc='#E8F5F2',ec=TEAL,weight='bold')
box(1.9,2.35,1.45,.75,'strong view\n$x_s$',fc='#FFF4DE',ec=ORANGE,weight='bold')
arrow(1.5,2.7,1.9,2.7); arrow(1.5,1.22,1.9,2.65)
box(3.85,1.65,1.55,1.15,'shared encoder\n$f_\theta$',fc='#EAF2F8',ec=BLUE,weight='bold',fs=10)
arrow(3.35,2.7,3.85,2.35); arrow(3.35,1.22,3.85,2.1)
box(6.0,2.35,1.2,.7,'$z_w,p_w$',fc='#F4ECF7',ec='#8E6BB3',weight='bold'); box(6.0,.95,1.2,.7,'$z_s,p_s$',fc='#F4ECF7',ec='#8E6BB3',weight='bold')
arrow(5.4,2.35,6.0,2.7); arrow(5.4,2.1,6.0,1.3)
box(7.75,2.3,1.85,.8,'classification\n$L_{cls}$',fc='#E8F5F2',ec=TEAL,weight='bold')
box(7.75,1.05,1.85,.8,'consistency\n$L_{feat}+L_{KL}$',fc='#FFF4DE',ec=ORANGE,weight='bold')
arrow(7.2,2.7,7.75,2.7); arrow(7.2,1.3,7.75,1.45)
box(7.55,.12,2.25,.55,'$d=1-\cos(z_w,z_s)$  →  $w=clip(d/mean(d))$',fc='#FDECEC',ec=RED,fs=8,weight='bold')
arrow(8.65,1.05,8.65,.67,RED,'-|>',1.2)
ax.text(5.0,3.55,'ShiftGuard: adaptive consistency regularization',ha='center',fontsize=12,color=NAVY,weight='bold')
ax.text(5.0,-.12,'Target-domain images are used only for final evaluation; model selection uses source validation only.',ha='center',fontsize=7.5,color=GRAY)
save(fig,'fig1_shiftguard_overview')

# Figure 2: main results and gains
# The root CSV contains one historical duplicate; load_dir deduplicates by seed.
g=load_dir('runs')
gsg=defaultdict(list)
for p in Path('runs/tuned_formal').glob('*.json'):
 d=json.loads(p.read_text()); gsg[(d['target'],'shiftguard')].append(d['target_accuracy']*100)
for k,v in gsg.items(): g[k]=v
methods=['erm','aug','shiftguard']; colors=[GRAY,ORANGE,BLUE]
fig,axs=plt.subplots(1,2,figsize=(7.2,2.75),gridspec_kw={'width_ratios':[1.65,1]})
fig.subplots_adjust(top=.82,wspace=.38,left=.08,right=.98,bottom=.24)
fig.suptitle('PACS leave-one-domain-out generalization',x=.5,y=.98,fontsize=12,color=NAVY,weight='bold')
ax=axs[0]; x=np.arange(4); width=.23
for j,m in enumerate(methods):
 means=[np.mean(g[(d,m)]) for d in DOMAINS]; std=[np.std(g[(d,m)],ddof=1) for d in DOMAINS]
 ax.bar(x+(j-1)*width,means,width,yerr=std,capsize=2,color=colors[j],label={'erm':'ERM','aug':'Strong Aug.','shiftguard':'ShiftGuard'}[m],edgecolor='white',linewidth=.5)
ax.set_xticks(x,LABELS); ax.set_ylim(55,101); ax.set_ylabel('Target accuracy (%)'); ax.set_title('(a) Main results',loc='left',weight='bold',color=NAVY); ax.grid(axis='y',alpha=.22)
fig.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.42,.035),columnspacing=1.0,handlelength=1.2)
ax=axs[1]; deltas=[np.mean(g[(d,'shiftguard')])-np.mean(g[(d,'erm')]) for d in DOMAINS]
barcols=[TEAL if v>=0 else RED for v in deltas]; ax.barh(np.arange(4),deltas,color=barcols,height=.55); ax.axvline(0,color=DARK,lw=.8); ax.set_yticks(np.arange(4),LABELS); ax.invert_yaxis(); ax.set_xlabel('ShiftGuard − ERM (pp)'); ax.set_title('(b) Gain over ERM',loc='left',weight='bold',color=NAVY); ax.grid(axis='x',alpha=.2)
for i,v in enumerate(deltas): ax.text(v+(0.25 if v>=0 else -0.25),i,f'{v:+.1f}',va='center',ha='left' if v>=0 else 'right',fontsize=8,weight='bold')
save(fig,'fig2_main_results')

# Figure 3: ablation heatmap and backbone transfer
ab=load_dir('ablation'); ams=['aug','feat','feat_kl','shiftguard']; anames=['Aug.','Feature','Feature + KL','Adaptive']
mat=np.array([[np.mean(ab[(d,m)]) for m in ams] for d in DOMAINS])
fig=plt.figure(figsize=(7.2,3.0)); gs=fig.add_gridspec(1,2,width_ratios=[1.55,1],wspace=.42,left=.07,right=.98,top=.82,bottom=.20)
fig.suptitle('Where does consistency help?',x=.5,y=.98,fontsize=12,color=NAVY,weight='bold')
ax=fig.add_subplot(gs[0]); cmap=plt.get_cmap('RdBu_r'); norm=TwoSlopeNorm(vmin=78,vcenter=84.0,vmax=98)
im=ax.imshow(mat,cmap=cmap,norm=norm,aspect='auto'); ax.set_xticks(range(4),anames,rotation=20,ha='right'); ax.set_yticks(range(4),LABELS); ax.set_title('(a) Component ablation',loc='left',weight='bold',color=NAVY); ax.set_xlabel('Regularization variant')
for i in range(4):
 for j in range(4): ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',color='white' if mat[i,j]>90 else DARK,weight='bold',fontsize=8)
for sp in ax.spines.values(): sp.set_visible(False)
fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label='Accuracy (%)')
ax=fig.add_subplot(gs[1]);
# Architecture transfer: paired endpoints with offset labels and improvement callouts.
res_erm=np.mean([json.loads(p.read_text())['target_accuracy']*100 for p in Path('runs').glob('*.json') if 'tuned_formal' not in str(p) and json.loads(p.read_text())['method']=='erm' and json.loads(p.read_text())['target'] in DOMAINS])
res_sg=np.mean([json.loads(p.read_text())['target_accuracy']*100 for p in Path('runs/tuned_formal').glob('*.json')])
vit_erm=np.mean([json.loads(p.read_text())['target_accuracy']*100 for p in Path('runs/vit_s').glob('*erm*.json')]); vit_sg=np.mean([json.loads(p.read_text())['target_accuracy']*100 for p in Path('runs/vit_s').glob('*shiftguard*.json')])
xx=[0,1]
ax.plot(xx,[res_erm,vit_erm],color=GRAY,lw=2.6,marker='o',ms=8,markeredgecolor='white',markeredgewidth=1.4,label='ERM',zorder=3)
ax.plot(xx,[res_sg,vit_sg],color=BLUE,lw=2.6,marker='o',ms=8,markeredgecolor='white',markeredgewidth=1.4,label='ShiftGuard',zorder=3)
ax.fill_between(xx,[res_erm,vit_erm],[res_sg,vit_sg],color=BLUE,alpha=.08,zorder=1)
ax.set_xticks(xx,['ResNet-50','ViT-S/16']); ax.set_xlim(-.25,1.55); ax.set_ylim(70,90); ax.set_ylabel('Mean accuracy (%)'); ax.set_title('(b) Backbone transfer',loc='left',weight='bold',color=NAVY); ax.grid(axis='y',alpha=.22); ax.legend(frameon=False,loc='lower left')
for x0,v in zip([0,1],[res_erm,vit_erm]): ax.annotate(f'{v:.1f}',(x0,v),xytext=(9,-13),textcoords='offset points',ha='left',color=GRAY,fontsize=8,weight='bold',bbox=dict(boxstyle='round,pad=.18',fc='white',ec='none',alpha=.9))
for x0,v in zip([0,1],[res_sg,vit_sg]): ax.annotate(f'{v:.1f}',(x0,v),xytext=(9,10),textcoords='offset points',ha='left',color=BLUE,fontsize=8,weight='bold',bbox=dict(boxstyle='round,pad=.18',fc='white',ec='none',alpha=.9))
ax.text(.5,.985,'vertical gap = ShiftGuard gain',transform=ax.transAxes,ha='center',va='top',fontsize=7,color=NAVY,weight='bold')
save(fig,'fig3_ablation_transfer')

# Figure 4: grouped robustness distribution plot
rd=pd.read_csv('runs/robustness.csv'); kinds=sorted(rd.corruption.unique());
fig,ax=plt.subplots(figsize=(7.2,3.65)); fig.subplots_adjust(left=.08,right=.98,top=.82,bottom=.25)
fig.suptitle('Robustness distribution across corruption types',x=.5,y=.97,fontsize=12,color=NAVY,weight='bold')
RICH_ORANGE='#F06400'; RICH_BLUE='#0057B8'
method_cfg=[('aug',RICH_ORANGE,'o','Strong Aug.'),('shiftguard',RICH_BLUE,'D','ShiftGuard')]
severity_offsets=np.linspace(-.075,.075,5)
allvals=rd.accuracy.values*100; ymin=np.floor(allvals.min()-1.5); ymax=np.ceil(allvals.max()+2.8)
for i,k in enumerate(kinds):
 vals_by_method={}
 for method,color,marker,label in method_cfg:
  sub=rd[(rd.method==method)&(rd.corruption==k)].sort_values('severity')
  vals=sub.accuracy.values*100; vals_by_method[method]=vals
  xs=i+(-.18 if method=='aug' else .18)+severity_offsets
  ax.plot(xs,vals,color=color,lw=1.35,alpha=.58,zorder=1)
  sizes=np.linspace(26,58,5)
  ax.scatter(xs,vals,s=sizes,marker=marker,color=color,edgecolor='white',linewidth=.7,alpha=1.0,zorder=3)
  mean=vals.mean(); sd=vals.std(ddof=1)
  ax.errorbar(i+(-.18 if method=='aug' else .18),mean,yerr=sd,fmt=marker,color=color,mfc='white',mec=color,mew=1.6,ms=9,capsize=3,capthick=1.2,lw=1.2,zorder=5)
 delta=vals_by_method['shiftguard'].mean()-vals_by_method['aug'].mean()
 top=max(vals_by_method['shiftguard'].max(),vals_by_method['aug'].max())+1.5
 ax.text(i,top,f'{delta:+.1f} pp',ha='center',va='bottom',fontsize=7.5,weight='bold',color=TEAL if delta>=0 else RED,bbox=dict(boxstyle='round,pad=.18',fc='white',ec='none',alpha=.9))
 if i>0: ax.axvline(i-.5,color='#D5DDE3',lw=.8,zorder=0)
ax.set_xticks(range(len(kinds)),[k.replace('_',' ').title() for k in kinds]); ax.set_ylabel('Target accuracy (%)'); ax.set_ylim(ymin,ymax); ax.set_xlim(-.55,len(kinds)-.45); ax.grid(axis='y',alpha=.22)
handles=[plt.Line2D([0],[0],marker='o',color='none',markerfacecolor=RICH_ORANGE,markeredgecolor='white',markersize=7,label='Strong Aug.'),plt.Line2D([0],[0],marker='D',color='none',markerfacecolor=RICH_BLUE,markeredgecolor='white',markersize=7,label='ShiftGuard'),plt.Line2D([0],[0],marker='o',color=DARK,markerfacecolor='white',markersize=8,label='Mean ± std')]
leg=ax.legend(handles=handles,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.12),columnspacing=1.1,handlelength=1.2)
ax.text(.99,.99,'marker size increases with severity (S1 → S5)',transform=ax.transAxes,ha='right',va='top',fontsize=7.2,color=GRAY)
save(fig,'fig4_robustness')
print('generated',sorted(str(p) for p in OUT.glob('*')))
