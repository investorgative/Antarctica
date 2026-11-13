import numpy as np, pandas as pd, itertools
from fam import *
CUT=pd.Timestamp('2024-01-01').value//10**6
def stats(p):
    rows=[]
    for s in ALL:
        tr,ex=run(s,p)
        for k in range(len(tr)): rows.append((ex[k],tr[k,3]))
    df=pd.DataFrame(rows,columns=['t','R']); a=df[df.t<CUT]; b=df[df.t>=CUT]
    ma=a.groupby(pd.to_datetime(a.t,unit='ms').dt.to_period('M')).R.sum(); mb=b.groupby(pd.to_datetime(b.t,unit='ms').dt.to_period('M')).R.sum()
    return (a.R.sum()/45, (a.R>0).mean()*100, ma.mean()/ma.std(), b.R.sum()/34.4, (b.R>0).mean()*100, mb.mean()/mb.std())
base=dict(tf=480,N=10,imp=1.25,trend=1,ema_n=200,atr_n=20,sl=2.0,tp=0.0,frac=1.0,be=0.0,tra=8.0,mb=20,short=1)
rows=[]
for tf,N,imp,sl,tra,mb in itertools.product([240,360,480],[8,10,15],[1.0,1.25,1.5],[2.0,2.5],[6.0,8.0,10.0],[20,30]):
    p=dict(base,tf=tf,N=N,imp=imp,sl=sl,tra=tra,mb=mb)
    r=stats(p); rows.append(dict(tf=tf,N=N,imp=imp,sl=sl,tra=tra,mb=mb,RmA=r[0],wrA=r[1],shA=r[2],RmB=r[3],wrB=r[4],shB=r[5]))
df=pd.DataFrame(rows); df.to_pickle('/home/user/research/neigh.pkl')
pd.set_option('display.width',250); pd.set_option('display.max_rows',400)
print(df.describe().round(2))
print(df.groupby('tf')[['RmA','wrA','shA','RmB','wrB','shB']].mean().round(2))
print(df.groupby('sl')[['RmA','wrA','shA','RmB','wrB','shB']].mean().round(2))
print(df.groupby('tra')[['RmA','wrA','shA','RmB','wrB','shB']].mean().round(2))
print(df.groupby('mb')[['RmA','wrA','shA','RmB','wrB','shB']].mean().round(2))
print(df.groupby('N')[['RmA','wrA','shA','RmB','wrB','shB']].mean().round(2))
print(df.groupby('imp')[['RmA','wrA','shA','RmB','wrB','shB']].mean().round(2))
