import numpy as np, pandas as pd, warnings, itertools, multiprocessing as mp; warnings.filterwarnings('ignore')
from plib import *
tf=240; C,R,F=panel(tf); bpy=365*6; pd_=6
vol=R.rolling(30*pd_).std(); avail=C.notna()&C.shift(30*pd_).notna(); n=avail.sum(axis=1)
def unitW(W):
    W=W.where(avail,0.0); g=W.abs().sum(1).replace(0,np.nan); return W.div(g,axis=0).fillna(0)
def reb(W,H): return W.where(pd.Series(np.arange(len(C))%H==0,index=C.index),np.nan).ffill()
def sleeve(W):
    r0,_=backtest(W,R,F,cost=0); lev=(0.20/(r0.rolling(60*pd_).std()*np.sqrt(bpy))).clip(upper=8).fillna(0); return W.mul(lev,axis=0)
def ts(L,H): return sleeve(unitW(reb(np.sign(C/C.shift(L*pd_)-1)/vol,H)))
def ca(L,H):
    rk=F.rolling(L*pd_).mean().where(avail).rank(axis=1); return sleeve(unitW(reb((rk<=3).astype(float)-(rk>n.values[:,None]-3).astype(float),H)))
def xs(L,H):
    rk=(C/C.shift(L*pd_)-1).where(avail).rank(axis=1); return sleeve(unitW(reb((rk>n.values[:,None]-3).astype(float)-(rk<=3).astype(float),H)))
def ev(args):
    (lt,lc,lx,H)=args
    W=(ts(lt,H)+ca(lc,H)+xs(lx,H))/3*3
    r,_=backtest(W,R,F); r=r[r.index>='2020-10-01']
    out=dict(lt=lt,lc=lc,lx=lx,H=H)
    for nm,(a,b) in {'all':('2020','2027'),'A':('2020','2024'),'B':('2024','2027'),'ex2m':('2020','2026-09-13')}.items():
        x=r[(r.index>=a)&(r.index<b)]; s=stats(x,bpy); out[f'{nm}_sh']=s['sharpe']; out[f'{nm}_avgm']=s['avgm']*100; out[f'{nm}_dd']=s['dd']*100; out[f'{nm}_geo']=s['geom']*100
    yy=by_year(r,bpy); out['miny']=min(v for k,v in yy.items() if k>2020); out['y26']=yy[2026]
    x=r[r.index>='2026-09-13']; out['last2m_ret']=((1+x).prod()-1)*100
    return out
if __name__=='__main__':
    grid=list(itertools.product([14,21,30,45,60],[3,7,14],[7,14,21,30],[18,42,84]))
    with mp.Pool(4) as p: res=p.map(ev,grid)
    df=pd.DataFrame(res); df.to_pickle('robust.pkl')
    pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
    print(df.describe().loc[['mean','min','25%','50%','75%','max']].round(2).to_string())
    print('share with every year Sharpe>0:',(df.miny>0).mean().round(2))
    for c in ['lt','lc','lx','H']: print(df.groupby(c)[['all_sh','A_sh','B_sh','all_dd','all_avgm']].mean().round(2))
