"""Independent re-computation of the headline DAR numbers (separate code path from dar.py):
detection counts on 22.04 and false reassurance of P vs B2 on invisible techniques (23.05, 21.03)."""
import pickle, numpy as np, pandas as pd
from scipy.stats import beta
BLK=300; R=0.5; RHO=300/(365*86400); PB=RHO/(RHO+R)
SP={'H1':['P1_B2016'],'H2':['P1_B3004'],'H3':['P2_AutoSD','P2_ManualSD']}
def eps(y):
    e=np.flatnonzero(np.diff(np.r_[0,y,0])); return list(zip(e[::2],e[1::2]))
def count(ver, h, layer, tags):
    st=pickle.load(open(f'alarms_{ver}.pkl','rb')); gt=st['gt']; k=n=0
    for name,f in st['files'].items():
        if f['y'] is None: continue
        g=gt[gt.file==name].reset_index(drop=True); on=f['A'].get((h,layer),np.array([]))
        for (s,e),(_,a) in zip(eps(f['y']),g.iterrows()):
            if set(str(a.points_eval).split(';'))&set(tags):
                n+=1; k+=bool(((on>=s)&(on<=e+60)).any())
    return k,n
def fr(ver,h,layer,tags,d_sil,d_al,alpha):
    st=pickle.load(open(f'alarms_{ver}.pkl','rb')); gt=st['gt']; tot=low=0
    for name,f in st['files'].items():
        if f['y'] is None: continue
        nb=f['n']//BLK; al=np.zeros(nb,bool); on=f['A'].get((h,layer),np.array([],int)); b=on//BLK; al[b[b<nb]]=True
        pi=PB; post=[]
        for x in al:
            pr=pi*(1-R)+(1-pi)*RHO; l1,l0=(d_al,alpha) if x else (1-d_sil,1-alpha); pi=pr*l1/(pr*l1+(1-pr)*l0); post.append(pi)
        post=np.array(post); g=gt[gt.file==name].reset_index(drop=True)
        for (s,e),(_,a) in zip(eps(f['y']),g.iterrows()):
            if set(str(a.points_eval).split(';'))&set(tags):
                seg=post[s//BLK:min(nb,(e+60)//BLK+1)]; tot+=len(seg); low+=(seg<0.5*PB).sum()
    return low/tot if tot else None
for h in ['H1','H2','H3']:
    k,n=count('2204',h,'IE_SP',SP[h]); dL=beta.ppf(0.05,1+k,1+n-k); dh=(1+k)/(2+n)
    print(h,'T0855-SP 22.04 detected',k,'of',n,'d_L %.3f'%dL, '| FR P 23.05 %.2f 21.03 %.2f | FR B2 23.05 %.2f 21.03 %.2f'%(
        fr('2305',h,'IE_SP',SP[h],dL,dh,1e-4),fr('2103',h,'IE_SP',SP[h],dL,dh,1e-4),fr('2305',h,'IE_SP',SP[h],.9,.9,1e-4),fr('2103',h,'IE_SP',SP[h],.9,.9,1e-4)))
