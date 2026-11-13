import numpy as np, itertools, sys, pandas as pd
from rlib import *
syms=['BTCUSDT','ETHUSDT','SOLUSDT']
rows=[]
for tf in [15,30,60,240]:
  for sym in syms:
    d=load(sym,tf); n=len(d['c']); o,h,l,c=d['o'],d['h'],d['l'],d['c']
    A=atr(d,20); e200=ema(c,200); e50=ema(c,50); e20=ema(c,20)
    start=300
    sigs={}
    for N in [20,55,100,200]:
      up=c>hh(h,N); dn=c<ll(l,N)
      rng=(h-l)/A
      for imp in [0,1.5]:
        for tf_ in [0,1]:
          s=np.where(up&(rng>=imp)&((c>e200)|(tf_==0)),1,np.where(dn&(rng>=imp)&((c<e200)|(tf_==0)),-1,0))
          sigs[f'don{N}_imp{imp}_tr{tf_}']=s.astype(np.int64)
    r=rsi(c,14)
    for lvl in [30,40]:
      upt=(e50>e200); dnt=(e50<e200)
      cross_up=(r>lvl)&(np.r_[100,r[:-1]]<=lvl); cross_dn=(r<100-lvl)&(np.r_[0,r[:-1]]>=100-lvl)
      sigs[f'pb_rsi{lvl}']=np.where(upt&cross_up,1,np.where(dnt&cross_dn,-1,0)).astype(np.int64)
    z=(c-pd.Series(c).rolling(20).mean().values)/pd.Series(c).rolling(20).std().values
    for zz in [2.0,2.5]:
      sigs[f'mr_z{zz}']=np.where((z<-zz)&(c>e200),1,np.where((z>zz)&(c<e200),-1,0)).astype(np.int64)
    nn=nanarr(n)
    for name,s in sigs.items():
      for sl,tp,frac,be,tra in [(1,0,1,0,3),(1.5,0,1,0,3),(2,0,1,0,4),(1,2,1,0,0),(1.5,1.5,1,0,0),(1,1,0.5,0,3),(1.5,1,0.5,1,3),(2,3,1,0,0),(1,3,1,0,0),(1.5,2,0.5,0,4),(2,2,1,1,0)]:
        tr=sim(o,h,l,c,d['fund'],A,s,sl,tp,frac,be,tra,nn,nn,0,start,TAKER,MAKER,SLIP)
        S=summarize(tr,d)
        rows.append(dict(tf=tf,sym=sym,sig=name,sl=sl,tp=tp,frac=frac,be=be,tra=tra,**{k:S[k] for k in ['n','wr','R','Rm','avgR']}))
  print('tf done',tf,file=sys.stderr)
df=pd.DataFrame(rows); df.to_pickle('/home/user/research/ex1.pkl')
# aggregate over symbols: config must work on all 3
g=df.groupby(['tf','sig','sl','tp','frac','be','tra']).agg(n=('n','sum'),wr=('wr','mean'),Rm=('Rm','sum'),minRm=('Rm','min'),avgR=('avgR','mean')).reset_index()
pd.set_option('display.width',250); pd.set_option('display.max_rows',100)
print(g.sort_values('Rm',ascending=False).head(40))
print(g[g.wr>35].sort_values('Rm',ascending=False).head(40))
