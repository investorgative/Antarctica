import os, sys, random, shutil, time as _time
D='/home/user/simdata'; shutil.rmtree(D,ignore_errors=True); os.makedirs(D)
os.environ['ANTARCTICA_DATA']=D; os.environ['ANTARCTICA_CONFIG']=D+'/config.json'
sys.path.insert(0,'/home/user/Antarctica')
import numpy as np, antarctica_core as core
core.refresh_instruments=lambda cfg,log=None: False
import antarctica_trader as T
TF=360; tf=TF*60000
SY=core.SYMBOLS
M={}
for s in SY:
    z=np.load(f'/home/user/data/{s}_1m.npz'); M[s]={k:z[k] for k in ('ts','o','h','l','c','v')}
    f=np.load(f'/home/user/data/{s}_funding.npz'); M[s]['fts']=(f['ts']//60000*60000); M[s]['fr']=f['rate']
END=int(M['BTCUSDT']['ts'][-1])//tf*tf; START=END-120*core.DAY
NOW=[START+3*60000]
core.now_ms=lambda: NOW[0]
def agg(s,a,b):
    m=M[s]; i0=np.searchsorted(m['ts'],a); i1=np.searchsorted(m['ts'],b)
    if i1<=i0: return None
    return (int(a),float(m['o'][i0]),float(m['h'][i0:i1].max()),float(m['l'][i0:i1].min()),float(m['c'][i1-1]),float(m['v'][i0:i1].sum()))
class MockAPI:
    def __init__(self,cfg): pass
    def klines(self,sym,interval,start=None,end=None,limit=1000,tries=4):
        now=NOW[0]; cur=now//tf*tf
        if start is None: start=cur-(limit-1)*tf
        e=min(end if end is not None else cur, cur)
        out=[]; t=max(start//tf*tf, e-(limit-1)*tf)
        while t<=e:
            b=agg(sym,t,min(t+tf,now - 60000*0) if t==cur else t+tf)
            if t==cur: b=agg(sym,t,now)  # forming candle: 1m candles that already finished
            if b: out.append(b)
            t+=tf
        return out
    def klines_range(self,sym,interval,start,end,progress=None):
        return [r for r in self.klines(sym,interval,start=start,end=end,limit=100000) if r[0]+tf<=NOW[0]]
    def funding_range(self,sym,start,end):
        m=M[sym]; sel=(m['fts']>=start)&(m['fts']<=min(end,NOW[0]))
        return dict(zip(m['fts'][sel].tolist(),m['fr'][sel].tolist()))
    def tickers(self):
        return {s:{'lastPrice':agg(s,NOW[0]//tf*tf,NOW[0])[4] if agg(s,NOW[0]//tf*tf,NOW[0]) else 0} for s in SY}
core.Bybit=MockAPI; T.core.Bybit=MockAPI
cfg=core.load_config()
# pre-fill history (closed candles before START) so the warm-up needs no huge download
st=core.Store()
p=core.clean_params(cfg['strategy'])
for s in SY:
    rows=[]; t=(START-(core.warmup_bars(p)+80)*tf)//tf*tf
    while t+tf<=START:
        b=agg(s,t,t+tf); 
        if b: rows.append(b)
        t+=tf
    st.insert(s,TF,rows,closed_before=START)
random.seed(11)
tr=T.Trader(); tr.load_state(); tr.warmup(); trade_from=tr.eng.trade_from
polls=restarts=0; down=0
while NOW[0]<END-60000:
    step=random.choice([5,10,20,30,45])*60000
    if random.random()<0.004:   # downtime up to 14h
        NOW[0]+=random.randint(1,14)*3600000; down+=1
    if random.random()<0.004:   # crash + restart
        tr.save_state(); tr=T.Trader(); tr.load_state(); tr.warmup(); restarts+=1
    tr.poll(); tr.flush_events(); tr.save_state(); polls+=1
    NOW[0]=min(END-60000, NOW[0]+step)
NOW[0]=END+5*60000; tr.poll(); tr.flush_events()
live=tr.eng.trades
# backtest over the same closed candles
datas={}; fund={}
for s in SY:
    rows=[]; t=(START-(core.warmup_bars(p)+80)*tf)//tf*tf
    while t+tf<=END:
        b=agg(s,t,t+tf)
        if b: rows.append(b)
        t+=tf
    datas[s]=core.rows_to_data(rows); fund[s]=MockAPI(cfg).funding_range(s,START-core.DAY,END)
bt=core.run_backtest(datas,cfg['strategy'],cfg,start_ms=trade_from,funding=fund,close_end=False)
key=lambda t:(t['sym'],t['entry_ts'],t['exit_ts'],round(t['entry'],6),round(t['exit'],6),round(t['qty'],8),round(t['fees'],8),round(t['funding'],8),t['reason'])
L=sorted(key(t) for t in live); B=sorted(key(t) for t in bt.trades)
print(f'polls {polls} downtimes {down} restarts {restarts}')
print(f'live trades {len(L)}  backtest trades {len(B)}  identical: {L==B}')
print(f'live balance {tr.eng.balance:.6f}  backtest balance {bt.balance:.6f}')
print('open live',sorted(tr.eng.pos), 'open bt', sorted(bt.pos))
if L!=B:
    for a,b in zip(L,B):
        if a!=b: print('first diff\n live',a,'\n bt  ',b); break
