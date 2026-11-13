import numpy as np, pandas as pd
from fam import *
CUT=pd.Timestamp('2024-01-01').value//10**6
def adx(d,n=14):
    h,l,c=d['h'],d['l'],d['c']
    up=np.diff(h,prepend=h[0]); dn=-np.diff(l,prepend=l[0])
    pdm=np.where((up>dn)&(up>0),up,0); ndm=np.where((dn>up)&(dn>0),dn,0)
    pc=np.r_[c[0],c[:-1]]; tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc)))
    a=pd.Series(tr).ewm(alpha=1/n,adjust=False).mean()
    pdi=100*pd.Series(pdm).ewm(alpha=1/n,adjust=False).mean()/a; ndi=100*pd.Series(ndm).ewm(alpha=1/n,adjust=False).mean()/a
    dx=100*abs(pdi-ndi)/(pdi+ndi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean().fillna(0).values
def runf(sym,p,f):
    d,cache=get(sym,p['tf']); c,h,l=d['c'],d['h'],d['l']
    A=atr(d,p['atr_n']); e=ema(c,p['ema_n']); H,L=hh(h,p['N']),ll(l,p['N'])
    r=(h-l)/A
    up=(c>H)&(r>=p['imp'])&(c>e); dn=(c<L)&(r>=p['imp'])&(c<e)
    if f.get('btc'):
        b,_=get('BTCUSDT',p['tf']); be=ema(b['c'],f['btc']); s=pd.Series(b['c']>be,index=b['ts']); s2=pd.Series(b['c']<be,index=b['ts'])
        bu=s.reindex(d['ts']).fillna(False).values; bd=s2.reindex(d['ts']).fillna(False).values
        up&=bu; dn&=bd
    if f.get('volp'):
        bars=int(30*1440/p['tf']); pr=pd.Series(A/c).rolling(bars).rank(pct=True).values
        up&=pr<=f['volp']; dn&=pr<=f['volp']
    if f.get('adx'):
        x=adx(d); up&=x>=f['adx']; dn&=x>=f['adx']
    if f.get('slope'):
        sl=e>np.r_[e[:f['slope']],e[:-f['slope']]]; up&=sl; dn&=~sl
    sig=np.where(up,1,np.where(dn,-1,0)).astype(np.int64); nn=nanarr(len(sig))
    tr=sim(d['o'],h,l,c,d['fund'],A,sig,p['sl'],0.0,1.0,p['be'],p['tra'],nn,nn,p['mb'],max(320,p['ema_n']+10),TAKER,MAKER,SLIP)
    ex=d['ts'][tr[:,1].astype(int)] if len(tr) else np.zeros(0,dtype=np.int64)
    return tr,ex
bases={'A_6h':dict(tf=360,N=15,imp=1.25,ema_n=200,atr_n=14,sl=2.0,be=0.0,tra=8.0,mb=30),
       'B_4h':dict(tf=240,N=8,imp=1.0,ema_n=200,atr_n=30,sl=2.0,be=0.0,tra=8.0,mb=30),
       'C_4h':dict(tf=240,N=15,imp=1.5,ema_n=150,atr_n=20,sl=2.5,be=2.0,tra=10.0,mb=30)}
filts={'none':{},'btc200':{'btc':200},'btc100':{'btc':100},'vol75':{'volp':0.75},'vol85':{'volp':0.85},'adx20':{'adx':20},'adx25':{'adx':25},'slope5':{'slope':5},'btc200+vol85':{'btc':200,'volp':0.85}}
for bn,p in bases.items():
    for fn,f in filts.items():
        a=[];b=[]
        for s in ALL:
            tr,ex=runf(s,p,f); a.append(tr[ex<CUT,3]); b.append(tr[ex>=CUT,3])
        a=np.concatenate(a); b=np.concatenate(b)
        print(f"{bn:5s} {fn:13s} | 2020-23: Rm {a.sum()/48:5.2f} wr {(a>0).mean()*100:4.1f} n {len(a):5d} | 2024-26: Rm {b.sum()/34.4:5.2f} wr {(b>0).mean()*100:4.1f} n {len(b):5d}")
