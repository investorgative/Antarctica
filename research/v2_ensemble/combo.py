import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from plib import *
tf=240; C,R,F=panel(tf); bpy=365*6; pd_=6
vol=R.rolling(30*pd_).std(); avail=C.notna()&C.shift(30*pd_).notna()
def unitW(W):
    W=W.where(avail,0.0); g=W.abs().sum(1).replace(0,np.nan); return W.div(g,axis=0).fillna(0)
def reb(W,H): return W.where(pd.Series(np.arange(len(C))%H==0,index=C.index),np.nan).ffill()
mom=C/C.shift(30*pd_)-1
W_ts=unitW(reb(np.sign(mom)/vol,42))
fa=F.rolling(7*pd_).mean(); rk=fa.where(avail).rank(axis=1); n=avail.sum(axis=1)
W_ca=unitW(reb((rk<=3).astype(float)-(rk>n.values[:,None]-3).astype(float),42))
m14=C/C.shift(14*pd_)-1; rk2=m14.where(avail).rank(axis=1)
W_xs=unitW(reb((rk2>n.values[:,None]-3).astype(float)-(rk2<=3).astype(float),42))
def sleeve(W):  # vol-normalize sleeve to 20% using past 60d frictionless returns
    r0,_=backtest(W,R,F,cost=0); lev=(0.20/(r0.rolling(60*pd_).std()*np.sqrt(bpy))).clip(upper=8).fillna(0); return W.mul(lev,axis=0)
S={'ts':sleeve(W_ts),'carry':sleeve(W_ca),'xs':sleeve(W_xs)}
rs={k:backtest(v,R,F)[0] for k,v in S.items()}
X=pd.DataFrame(rs)[lambda d:d.index>='2020-10-01']
print('daily corr:\n',X.resample('D').sum().corr().round(2))
for combo in [['ts'],['ts','carry'],['ts','carry','xs']]:
    W=sum(S[k] for k in combo)/len(combo)
    for tgt in [1,1.5,2,3,4]:
        r,_=backtest(W*tgt,R,F); r=r[r.index>='2020-10-01']
        s=stats(r,bpy); y=by_year(r,bpy)
        a=r[r.index<'2024-01-01']; b=r[r.index>='2024-01-01']
        sa,sb=stats(a,bpy),stats(b,bpy)
        print(f"{'+'.join(combo):12s} x{tgt}: sharpe {s['sharpe']:.2f} DD {s['dd']*100:4.0f}% avg month {s['avgm']*100:5.1f}% geo {s['geom']*100:5.1f}% green {s['posm']*100:3.0f}% worst {s['worstm']*100:5.1f}% | 20-23 geo {sa['geom']*100:4.1f}% DD {sa['dd']*100:3.0f}% | 24-26 geo {sb['geom']*100:4.1f}% DD {sb['dd']*100:3.0f}% sh {sb['sharpe']:.2f}")
