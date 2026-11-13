import numpy as np, sys, pandas as pd
from rlib import *
syms=['BTCUSDT','ETHUSDT','SOLUSDT']
SPLIT=pd.Timestamp('2023-06-01').value//10**6
rng_=np.random.default_rng(1)
rows=[]
def signal(d,A,N,imp,trend,ema_n,cache):
    c,h,l=d['c'],d['h'],d['l']
    k=('e',ema_n)
    if k not in cache: cache[k]=ema(c,ema_n)
    e=cache[k]
    k=('hh',N)
    if k not in cache: cache[k]=(hh(h,N),ll(l,N))
    H,L=cache[k]
    r=(h-l)/A
    up=(c>H)&(r>=imp); dn=(c<L)&(r>=imp)
    if trend: up&=c>e; dn&=c<e
    return np.where(up,1,np.where(dn,-1,0)).astype(np.int64)
data={}
for tf in [60,120,240,360,480]:
    for s in syms:
        d=load(s,tf); data[(s,tf)]=(d,{})
for it in range(2500):
    p=dict(tf=int(rng_.choice([60,120,240,360,480])),N=int(rng_.choice([10,15,20,30,40,55,80])),imp=float(rng_.choice([0,1,1.25,1.5,1.75,2,2.5])),
           trend=int(rng_.choice([0,1])),ema_n=int(rng_.choice([100,200,300])),atr_n=int(rng_.choice([14,20,30])),
           sl=float(rng_.choice([1,1.5,2,2.5,3])),tp=float(rng_.choice([0,0,0,2,3,4,6])),frac=float(rng_.choice([0.3,0.5,1])),
           be=float(rng_.choice([0,0,1,2])),tra=float(rng_.choice([2,3,4,5,6,8])),mb=int(rng_.choice([0,0,30,60])))
    res=[]
    for s in syms:
        d,cache=data[(s,p['tf'])]
        k=('atr',p['atr_n'])
        if k not in cache: cache[k]=atr(d,p['atr_n'])
        A=cache[k]
        sig=signal(d,A,p['N'],p['imp'],p['trend'],p['ema_n'],cache)
        nn=nanarr(len(sig))
        tr=sim(d['o'],d['h'],d['l'],d['c'],d['fund'],A,sig,p['sl'],p['tp'],p['frac'],p['be'],p['tra'],nn,nn,p['mb'],320,TAKER,MAKER,SLIP)
        ex=d['ts'][tr[:,1].astype(int)] if len(tr) else np.array([])
        a=tr[ex<SPLIT]; b=tr[ex>=SPLIT]
        res.append((len(tr),(tr[:,3]>0).mean()*100 if len(tr) else 0,tr[:,3].sum(),a[:,3].sum(),b[:,3].sum()))
    res=np.array(res)
    m_all=(d['ts'][-1]-d['ts'][320])/(30.4375*864e5)
    rows.append(dict(**p,n=res[:,0].sum(),wr=(res[:,0]*res[:,1]).sum()/max(1,res[:,0].sum()),Rm=res[:,2].sum()/m_all,
                     minsym=res[:,2].min(),RA=res[:,3].sum(),RB=res[:,4].sum()))
    if it%250==0: print(it,file=sys.stderr)
df=pd.DataFrame(rows); df.to_pickle('/home/user/research/ex2.pkl')
pd.set_option('display.width',250); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',30)
df['minAB']=df[['RA','RB']].min(axis=1)
print(df.sort_values('Rm',ascending=False).head(25))
print(df[(df.wr>35)].sort_values('minAB',ascending=False).head(25))
