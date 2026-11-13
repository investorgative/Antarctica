import numpy as np, pandas as pd, multiprocessing as mp, sys
from fam import *
from psearch import sample
YEARS=list(range(2020,2027))
def ev(p):
    rows=[]
    for s in ALL:
        tr,ex=run(s,p)
        for k in range(len(tr)): rows.append((ex[k]+p['tf']*60000, tr[k,3]))
    if len(rows)<50: return p,None
    df=pd.DataFrame(rows,columns=['t','R']); dt=pd.to_datetime(df.t,unit='ms')
    m=df.groupby(dt.dt.to_period('M')).R.sum().reindex(pd.period_range('2020-04','2026-11',freq='M'),fill_value=0)
    y=df.groupby(dt.dt.year).agg(R=('R','sum'),w=('R',lambda x:(x>0).sum()),n=('R','size')).reindex(YEARS,fill_value=0)
    ysh=[ (m[m.index.year==Y].mean()/m[m.index.year==Y].std()) if m[m.index.year==Y].std()>0 else 0 for Y in YEARS]
    return p,dict(m=m.values,yR=y.R.values,yW=y.w.values,yN=y.n.values,ysh=np.array(ysh))
if __name__=='__main__':
    rng=np.random.default_rng(int(sys.argv[2]))
    ps=[sample(rng) for _ in range(int(sys.argv[1]))]
    with mp.Pool(4) as pool: res=pool.map(ev,ps,chunksize=4)
    pd.to_pickle([r for r in res if r[1] is not None],sys.argv[3])
