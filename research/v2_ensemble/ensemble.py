import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from robust import *
LT=[14,21,30,45,60]; LC=[3,7,14]; LX=[7,14,21,30]; HS=[18,42]
TS=sum(ts(l,h) for l in LT for h in HS)/(len(LT)*len(HS))
CA=sum(ca(l,h) for l in LC for h in HS)/(len(LC)*len(HS))
XS=sum(xs(l,h) for l in LX for h in HS)/(len(LX)*len(HS))
def renorm(W):  # each ensemble sleeve back to 20% vol (averaging lowers vol)
    r0,_=backtest(W,R,F,cost=0); lev=(0.20/(r0.rolling(60*pd_).std()*np.sqrt(bpy))).clip(upper=8).fillna(0); return W.mul(lev,axis=0)
TS,CA,XS=renorm(TS),renorm(CA),renorm(XS)
for name,W0 in [('ts',TS),('carry',CA),('xs',XS),('ts+carry',(TS+CA)/2),('all3',(TS+CA+XS)/3)]:
    r1,_=backtest(W0,R,F); r1=r1[r1.index>='2020-10-01']; yy=by_year(r1,bpy)
    print(f"{name:9s} unit sharpe {stats(r1,bpy)['sharpe']:.2f}  by year "+' '.join(f'{k}:{v:.2f}' for k,v in yy.items()))
W0=(TS+CA+XS)/3
for k in [2,3,4,5,6]:
    r,_=backtest(W0*k,R,F); r=r[r.index>='2020-10-01']
    s=stats(r,bpy); a=stats(r[r.index<'2024'],bpy); b=stats(r[r.index>='2024'],bpy); e=stats(r[r.index<'2026-09-13'],bpy)
    gross=(W0*k).abs().sum(1); gross=gross[gross.index>='2020-10-01']
    print(f"x{k}: sharpe {s['sharpe']:.2f} DD {s['dd']*100:4.1f}% avg month {s['avgm']*100:5.2f}% geo {s['geom']*100:5.2f}% green {s['posm']*100:3.0f}% worst {s['worstm']*100:5.1f}% | 20-23 avg {a['avgm']*100:4.1f}% DD {a['dd']*100:3.0f}% | 24-26 avg {b['avgm']*100:4.1f}% DD {b['dd']*100:3.0f}% sh {b['sharpe']:.2f} | ex-last-2m avg {e['avgm']*100:4.1f}% | gross lev median {gross.median():.1f}x p95 {gross.quantile(.95):.1f}x")
pd.to_pickle({'TS':TS,'CA':CA,'XS':XS},'ens_w.pkl')
