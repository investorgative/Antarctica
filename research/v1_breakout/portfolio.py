import numpy as np, pandas as pd, sys
from fam import *
from filters import bases
def trades_all(p,syms=ALL):
    rows=[]
    for s in syms:
        d,_=get(s,p['tf']); tr,ex=run(s,dict(p,trend=1,tp=0.0,frac=1.0))
        en=d['ts'][tr[:,0].astype(int)]
        for k in range(len(tr)): rows.append((en[k],ex[k]+p['tf']*60000,tr[k,3],s))
    return pd.DataFrame(rows,columns=['en','ex','R','sym']).sort_values('en').reset_index(drop=True)
def port(df,risk,start='2020-01-01',end='2026-11-13',bal0=100.0,maxpos=99):
    s=pd.Timestamp(start).value//10**6; e=pd.Timestamp(end).value//10**6
    df=df[(df.en>=s)&(df.ex<e)]
    ev=[(r.en,1,i) for i,r in df.iterrows()]+[(r.ex,0,i) for i,r in df.iterrows()]
    ev.sort()
    bal=bal0; openr={}; hist=[(s,bal)]; peak=bal; dd=0; skipped=0
    for t,typ,i in ev:
        if typ==1:
            if len(openr)>=maxpos: skipped+=1; continue
            openr[i]=bal*risk
        elif i in openr:
            bal+=openr.pop(i)*df.R[i]; hist.append((t,bal)); peak=max(peak,bal); dd=max(dd,(peak-bal)/peak)
    h=pd.Series([b for _,b in hist],index=pd.to_datetime([t for t,_ in hist],unit='ms'))
    m=h.resample('MS').last().ffill(); m=pd.concat([pd.Series([bal0],index=[m.index[0]-pd.Timedelta(days=1)]),m])
    mr=m.pct_change().dropna()*100
    yrs=h.resample('YS').last().ffill(); 
    return dict(risk=risk*100,final=bal,avgm=mr.mean(),medm=mr.median(),posm=(mr>0).mean()*100,worstm=mr.min(),maxdd=dd*100,
                geo=((bal/bal0)**(1/len(mr))-1)*100,skipped=skipped,mr=mr)
def _x():
    pass
if __name__=='__main__':
    for bn in ['A_6h','B_4h','C_4h']:
        p=bases[bn]; df=trades_all(p)
        for r in [0.005,0.01,0.015,0.02,0.03]:
            for per in [('2020-01-01','2026-11-13'),('2023-11-13','2026-11-13'),('2025-11-13','2026-11-13')]:
                o=port(df,r,*per)
                print(f"{bn} risk {o['risk']:.1f}% {per[0][:7]}: avg month {o['avgm']:6.1f}%  median {o['medm']:5.1f}%  geo {o['geo']:5.1f}%  green {o['posm']:3.0f}%  worst {o['worstm']:6.1f}%  maxDD {o['maxdd']:5.1f}%  $100->{o['final']:,.0f}")
        print()
