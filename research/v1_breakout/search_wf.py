import numpy as np, pandas as pd, sys, multiprocessing as mp
from fam import *
YEARS=list(range(2020,2027))
def yr(ex): return pd.to_datetime(ex,unit='ms').year.values
def evalp(p):
    # per-year: R sum, wins, n  (summed over coins)
    R=np.zeros(len(YEARS)); W=np.zeros(len(YEARS)); N=np.zeros(len(YEARS)); coinR=[]
    for s in ALL:
        tr,ex=run(s,p)
        if len(tr)==0: coinR.append(0); continue
        y=yr(ex)
        for j,Y in enumerate(YEARS):
            m=y==Y; R[j]+=tr[m,3].sum(); W[j]+=(tr[m,3]>0).sum(); N[j]+=m.sum()
        coinR.append(tr[:,3].sum())
    return dict(p=p,R=R,W=W,N=N,coinR=np.array(coinR))
def sample(rng):
    return dict(tf=int(rng.choice([180,240,360,480,720])),N=int(rng.choice([8,10,12,15,20,25,30,40])),imp=float(rng.choice([0,0.5,1,1.25,1.5,2])),
           trend=int(rng.choice([0,1,1])),ema_n=int(rng.choice([100,150,200,300])),atr_n=int(rng.choice([14,20,30])),
           sl=float(rng.choice([1.5,2,2.5,3,3.5])),tp=0.0,frac=1.0,be=float(rng.choice([0,0,1,2,3])),tra=float(rng.choice([4,5,6,7,8,10,12])),
           mb=int(rng.choice([0,0,20,30,60])),short=int(rng.choice([1,1,0])))
if __name__=='__main__':
    rng=np.random.default_rng(int(sys.argv[2]) if len(sys.argv)>2 else 7)
    ps=[sample(rng) for _ in range(int(sys.argv[1]))]
    ps.append(dict(tf=360,N=15,imp=1.25,trend=1,ema_n=200,atr_n=14,sl=2.0,tp=0.0,frac=1.0,be=0.0,tra=8.0,mb=30,short=1))
    with mp.Pool(4) as pool:
        res=pool.map(evalp,ps,chunksize=8)
    pd.to_pickle(res,'/home/user/research/wf_res.pkl')
    print('done',len(res))
