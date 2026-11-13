import numpy as np, pandas as pd, sys, itertools, multiprocessing as mp
from mr import *
from fam import ALL
CUT=pd.Timestamp('2024-01-01').value//10**6
SY=sys.argv[1].split(',')
def ev(args):
    tf,n,trend,k,s_sd,tpf,mb=args
    rows=[]
    for sym in SY:
        d,mid,sd,okL,okS=prep(sym,tf,n,trend,'with' if trend else 'both')
        tr=sim_mr(d['o'],d['h'],d['l'],d['c'],d['fund'],mid,sd,okL,okS,k,s_sd,tpf,mb,0.0003,TAKER,MAKER,SLIP,max(n,trend)+5)
        ex=d['ts'][tr[:,1].astype(int)]
        for x,t in zip(tr[:,3],ex): rows.append((t,x))
    df=pd.DataFrame(rows,columns=['t','R'])
    if len(df)==0: return None
    m=df.groupby(pd.to_datetime(df.t,unit='ms').dt.to_period('M')).R.sum()
    a=df[df.t<CUT]; b=df[df.t>=CUT]
    ma=m[m.index<pd.Period('2024-01')]; mb_=m[m.index>=pd.Period('2024-01')]
    return dict(tf=tf,n=n,trend=trend,k=k,s=s_sd,tpf=tpf,mb=mb,nA=len(a),wrA=(a.R>0).mean()*100,RmA=ma.mean(),shA=ma.mean()/ma.std(),
                nB=len(b),wrB=(b.R>0).mean()*100,RmB=mb_.mean(),shB=mb_.mean()/mb_.std(),avgR=df.R.mean())
grid=list(itertools.product([5,15,60],[20,50],[0,200],[2.0,2.5,3.0],[1.0,2.0],[0.5,1.0],[12,48]))
with mp.Pool(4) as p: res=[r for r in p.map(ev,grid,chunksize=2) if r]
df=pd.DataFrame(res); df.to_pickle('/home/user/research/mr1.pkl')
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(df.sort_values('shA',ascending=False).head(20).round(2))
print('positive in both halves:',((df.RmA>0)&(df.RmB>0)).sum(),'of',len(df))
