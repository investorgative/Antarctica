import numpy as np, pandas as pd, multiprocessing as mp, sys
from fam import *
CUT=pd.Timestamp('2024-01-01').value//10**6
def daily(p, syms=ALL):
    rows=[]
    for s in syms:
        d,_=get(s,p['tf']); tr,ex=run(s,p)
        for k in range(len(tr)): rows.append((ex[k]+p['tf']*60000, tr[k,3]))
    if not rows: return None
    df=pd.DataFrame(rows,columns=['t','R'])
    sr=df.groupby(pd.to_datetime(df.t,unit='ms').dt.floor('D')).R.sum()
    idx=pd.date_range('2020-04-01','2026-11-12',freq='D'); return sr.reindex(idx,fill_value=0.0), df
def metrics(sr):
    cum=sr.cumsum(); dd=(cum.cummax()-cum).max()
    m=sr.resample('MS').sum()
    return dict(Rm=m.mean(), sdm=m.std(), sharpe=m.mean()/m.std() if m.std()>0 else 0, ddR=dd, mar=m.mean()/dd if dd>0 else 0, posm=(m>0).mean()*100)
def ev(p):
    x=daily(p)
    if x is None: return p,None
    sr,df=x
    a=sr[sr.index<'2024-01-01']; b=sr[sr.index>='2024-01-01']
    ta=df[df.t<CUT].R; tb=df[df.t>=CUT].R
    return p,dict(A=metrics(a),B=metrics(b),wrA=(ta>0).mean()*100,wrB=(tb>0).mean()*100,nA=len(ta),nB=len(tb))
def sample(rng):
    return dict(tf=int(rng.choice([120,180,240,360,480,720])),N=int(rng.choice([8,10,12,15,20,25,30,40,55])),imp=float(rng.choice([0,0.5,1,1.25,1.5,2,2.5])),
           trend=int(rng.choice([0,1,1])),ema_n=int(rng.choice([100,150,200,300])),atr_n=int(rng.choice([14,20,30])),
           sl=float(rng.choice([1,1.5,2,2.5,3,3.5])),tp=float(rng.choice([0,0,0,1.5,2,3])),frac=float(rng.choice([0.3,0.5,1])),be=float(rng.choice([0,0,1,2,3])),
           tra=float(rng.choice([3,4,5,6,8,10,12])),mb=int(rng.choice([0,0,10,20,30,60])),short=int(rng.choice([1,1,0])))
if __name__=='__main__':
    rng=np.random.default_rng(int(sys.argv[2]))
    ps=[sample(rng) for _ in range(int(sys.argv[1]))]
    with mp.Pool(4) as pool: res=pool.map(ev,ps,chunksize=4)
    pd.to_pickle(res,sys.argv[3])
