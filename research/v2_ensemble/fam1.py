import numpy as np, pandas as pd, itertools, sys
from plib import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
rows=[]
for tf in [240,1440]:
    C,R,F=panel(tf); bpy=365*1440/tf; per_day=1440//tf
    vol=R.rolling(30*per_day).std()  # per-coin vol (past only, at bar close)
    avail=C.notna()&C.shift(30*per_day).notna()
    def run(name,W,**kw):
        W=W.where(avail,0.0)
        # normalize to 1 unit gross exposure then vol-target the whole book to 20%/yr using past 60d realized
        g=W.abs().sum(1).replace(0,np.nan); W=W.div(g,axis=0).fillna(0)
        r0,_=backtest(W,R,F,cost=0)  # unit, frictionless just for vol estimate
        lev=(0.20/(r0.rolling(60*per_day).std()*np.sqrt(bpy))).shift(0).clip(upper=8).fillna(0)
        Wl=W.mul(lev,axis=0)
        r,turn=backtest(Wl,R,F)
        r=r[r.index>='2020-10-01']
        s=stats(r,bpy,name); y=by_year(r,bpy)
        rows.append(dict(tf=tf,name=name,**kw,sh=s['sharpe'],cagr=s['cagr'],dd=s['dd'],turn_y=turn[turn.index>='2020-10-01'].sum()/((len(r))/bpy),
                         **{f'y{k%100}':v for k,v in y.items()}))
    for L in [1,3,7,14,30,60]:
        lb=L*per_day; mom=C/C.shift(lb)-1
        for H in [1,per_day,7*per_day]:
            reb=(np.arange(len(C))%H==0)
            # cross-sectional momentum long top 3 / short bottom 3
            rk=mom.where(avail).rank(axis=1)
            n=avail.sum(1)
            W=(rk>n.values[:,None]-3).astype(float)-(rk<=3).astype(float)
            W=W.where(pd.Series(reb,index=C.index),np.nan).ffill()
            run('xs_mom',W,L=L,H=H)
            run('xs_rev',-W,L=L,H=H)
            # time-series momentum, inverse vol
            Wt=np.sign(mom)/vol
            Wt=Wt.where(pd.Series(reb,index=C.index),np.nan).ffill()
            run('ts_mom',Wt,L=L,H=H)
            run('ts_mom_long',Wt.clip(lower=0),L=L,H=H)
    # funding carry: short highest funding, long lowest (past 7d avg funding)
    fa=F.rolling(7*per_day).mean()
    for H in [per_day,7*per_day]:
        reb=pd.Series(np.arange(len(C))%H==0,index=C.index)
        rk=fa.where(avail).rank(axis=1); n=avail.sum(1)
        W=(rk<=3).astype(float)-(rk>n.values[:,None]-3).astype(float)
        run('carry',W.where(reb,np.nan).ffill(),L=7,H=H)
    print('tf',tf,'done',file=sys.stderr)
df=pd.DataFrame(rows); df.to_pickle('/home/user/research/v2/fam1.pkl')
yc=[c for c in df.columns if c.startswith('y')]
df['miny']=df[yc].min(1)
print(df.sort_values('sh',ascending=False).head(30).round(2))
print(df.sort_values('miny',ascending=False).head(15).round(2))
