import numpy as np, pandas as pd
from rlib import *
ALL=['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','DOGEUSDT','BNBUSDT','ADAUSDT','LINKUSDT','LTCUSDT','AVAXUSDT','DOTUSDT','BCHUSDT']
_c={}
def get(sym,tf):
    k=(sym,tf)
    if k not in _c: _c[k]=(load(sym,tf),{})
    return _c[k]
def run(sym,p):
    d,cache=get(sym,p['tf'])
    c,h,l=d['c'],d['h'],d['l']
    k=('atr',p['atr_n'])
    if k not in cache: cache[k]=atr(d,p['atr_n'])
    A=cache[k]
    k=('e',p['ema_n'])
    if k not in cache: cache[k]=ema(c,p['ema_n'])
    e=cache[k]
    k=('hh',p['N'])
    if k not in cache: cache[k]=(hh(h,p['N']),ll(l,p['N']))
    H,L=cache[k]
    r=(h-l)/A
    up=(c>H)&(r>=p['imp']); dn=(c<L)&(r>=p['imp'])
    if p['trend']: up&=c>e; dn&=c<e
    if not p.get('short',1): dn&=False
    sig=np.where(up,1,np.where(dn,-1,0)).astype(np.int64)
    nn=nanarr(len(sig))
    tr=sim(d['o'],h,l,c,d['fund'],A,sig,p['sl'],p['tp'],p['frac'],p['be'],p['tra'],nn,nn,p['mb'],max(320,p['ema_n']+10),TAKER,MAKER,SLIP)
    ex=d['ts'][tr[:,1].astype(int)] if len(tr) else np.zeros(0,dtype=np.int64)
    return tr,ex
