import numpy as np, pandas as pd
from fam import *
P=dict(tf=360,N=15,imp=1.5,trend=1,ema_n=200,atr_n=20,sl=2.0,tp=0.0,frac=1.0,be=0.0,tra=8.0,mb=30,short=1)
rows=[]
for s in ALL:
    d,_=get(s,P['tf']); tr,ex=run(s,P); en=d['ts'][tr[:,0].astype(int)]
    for k in range(len(tr)): rows.append((en[k],ex[k]+P['tf']*60000,tr[k,3],s))
df=pd.DataFrame(rows,columns=['en','ex','R','sym']).sort_values(['en','sym']).reset_index(drop=True)
print('trades',len(df),'wr %.1f'%((df.R>0).mean()*100),'avgR %.3f'%df.R.mean())
def port(r,heat,start,end='2026-11-13',bal0=100.0):
    s=pd.Timestamp(start).value//10**6; e=pd.Timestamp(end).value//10**6
    x=df[(df.en>=s)&(df.ex<=e)]
    ev=sorted([(t,1,i) for i,t in zip(x.index,x.en)]+[(t,0,i) for i,t in zip(x.index,x.ex)])
    bal=bal0; op={}; pts=[(s,bal)]; pk=bal; dd=0; skipped=0
    for t,typ,i in ev:
        if typ==1:
            used=sum(v[1] for v in op.values())
            rr=min(r, heat-used)
            if rr< r*0.5: skipped+=1; continue
            op[i]=(bal*rr, rr)
        elif i in op:
            bal+=op.pop(i)[0]*x.R[i]; pts.append((t,bal)); pk=max(pk,bal); dd=max(dd,(pk-bal)/pk)
    h=pd.Series([b for _,b in pts],index=pd.to_datetime([t for t,_ in pts],unit='ms'))
    m=h.resample('MS').last().ffill(); m=pd.concat([pd.Series([bal0],index=[m.index[0]-pd.Timedelta(days=1)]),m]); mr=m.pct_change().dropna()*100
    return mr.mean(),mr.median(),(mr>0).mean()*100,mr.min(),dd*100,((bal/bal0)**(1/len(mr))-1)*100,bal,skipped
for heat in [1.0,0.12,0.08,0.06]:
  for r in [0.01,0.015,0.02,0.025,0.03]:
    out=[]
    for st in ['2020-04-01','2023-11-13','2025-11-13']:
        a=port(r,heat,st); out.append(f"avg {a[0]:5.1f}% med {a[1]:5.1f}% geo {a[5]:5.1f}% green {a[2]:3.0f}% worstM {a[3]:6.1f}% DD {a[4]:4.1f}% ->${a[6]:,.0f}")
    print(f"heat {heat*100:4.0f}% risk {r*100:.1f}% | ALL {out[0]} | 3y {out[1]} | 1y {out[2]}")
