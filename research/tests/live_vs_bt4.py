import os, sys, random, shutil, warnings; warnings.filterwarnings('ignore')
D=sys.argv[1] if len(sys.argv)>1 else '/home/user/simdata4'; DAYS=int(sys.argv[2]) if len(sys.argv)>2 else 120
shutil.rmtree(D,ignore_errors=True); os.makedirs(D)
os.environ['ANTARCTICA_DATA']=D; os.environ['ANTARCTICA_CONFIG']=D+'/config.json'
sys.path.insert(0,'/home/user/Antarctica')
import numpy as np, antarctica_core as core, antarctica_strategy as strat
core.refresh_instruments=lambda cfg,log=None: False
import antarctica_trader as T
TF=240; tf=TF*60000; SY=core.SYMBOLS
M={}
for s in SY:
    z=np.load(f'/home/user/data/{s}_1m.npz'); M[s]={k:z[k] for k in ('ts','o','h','l','c','v')}
    f=np.load(f'/home/user/data/{s}_funding.npz'); M[s]['fts']=(f['ts']//60000*60000); M[s]['fr']=f['rate']
END=int(M['BTCUSDT']['ts'][-1])//tf*tf; START=END-DAYS*core.DAY
NOW=[START+3*60000]; core.now_ms=lambda: NOW[0]
def agg(s,a,b):
    m=M[s]; i0=np.searchsorted(m['ts'],a); i1=np.searchsorted(m['ts'],b)
    if i1<=i0: return None
    return (int(a),float(m['o'][i0]),float(m['h'][i0:i1].max()),float(m['l'][i0:i1].min()),float(m['c'][i1-1]),float(m['v'][i0:i1].sum()))
class MockAPI:
    def __init__(self,cfg): pass
    def klines(self,sym,interval,start=None,end=None,limit=1000,tries=4):
        now=NOW[0]; cur=now//tf*tf
        if start is None: start=cur-(limit-1)*tf
        e=min(end if end is not None else cur,cur); out=[]; t=max(start//tf*tf,e-(limit-1)*tf)
        while t<=e:
            b=agg(sym,t,now) if t==cur else agg(sym,t,t+tf)
            if b: out.append(b)
            t+=tf
        return out
    def klines_range(self,sym,interval,start,end,progress=None):
        return [r for r in self.klines(sym,interval,start=start,end=end,limit=100000) if r[0]+tf<=NOW[0]]
    def funding_range(self,sym,start,end):
        m=M[sym]; sel=(m['fts']>=start)&(m['fts']<=min(end,NOW[0])); return dict(zip(m['fts'][sel].tolist(),m['fr'][sel].tolist()))
    def tickers(self):
        out={}
        for s in SY:
            b=agg(s,NOW[0]//tf*tf,NOW[0]); out[s]={'lastPrice':b[4] if b else 0}
        return out
core.Bybit=MockAPI; T.core.Bybit=MockAPI
cfg=core.load_config(); st=core.Store(); p=core.clean_params(cfg['strategy'])
W0=START-(strat.warmup_bars(p)+120)*tf
for s in SY:
    rows=[]; t=W0//tf*tf
    while t+tf<=START:
        b=agg(s,t,t+tf)
        if b: rows.append(b)
        t+=tf
    st.insert(s,TF,rows,closed_before=START); st.insert_funding(s,MockAPI(cfg).funding_range(s,W0,START))
random.seed(21)
tr=T.Trader(); tr.load_state(); tr.warmup(); trade_from=tr.eng.trade_from
polls=restarts=down=0
while NOW[0]<END-60000:
    step=random.choice([5,10,20,30,45])*60000
    if random.random()<0.004: NOW[0]+=random.randint(1,14)*3600000; down+=1
    if random.random()<0.004: tr.save_state(); tr=T.Trader(); tr.load_state(); tr.warmup(); restarts+=1
    tr.poll(); tr.flush_events(); tr.save_state(); polls+=1
    NOW[0]=min(END-60000,NOW[0]+step)
tr.write_status(); tr.save_state()
datas={}; fund={}
for s in SY:
    rows=[]; t=W0//tf*tf
    while t+tf<=END:
        b=agg(s,t,t+tf)
        if b: rows.append(b)
        t+=tf
    datas[s]=core.rows_to_data(rows); fund[s]=MockAPI(cfg).funding_range(s,W0,END+tf)
bt=core.run_backtest(datas,cfg['strategy'],cfg,start_ms=trade_from,funding=fund,close_end=False)
key=lambda t:(t['sym'],t['entry_ts'],t['exit_ts'],float('%.9g'%t['entry']),float('%.9g'%t['exit']),round(t['fees'],6),round(t['funding'],6),round(t['pnl'],5))
L=sorted(key(t) for t in tr.eng.trades); B=sorted(key(t) for t in bt.trades)
le=tr.eng.cash; be=bt.cash
print(f'polls {polls} downtimes {down} restarts {restarts} | live closed trades {len(L)} backtest {len(B)} identical: {L==B}')
print(f'live cash {le:.6f} backtest cash {be:.6f} | open live {sorted((s,round(p["qty"],6)) for s,p in tr.eng.pos.items())==sorted((s,round(p["qty"],6)) for s,p in bt.pos.items())} | fees {tr.eng.fees_total:.6f} vs {bt.fees_total:.6f} funding {tr.eng.funding_total:.6f} vs {bt.funding_total:.6f}')
if L!=B:
    for a,b in zip(L,B):
        if a!=b: print('first diff\n live',a,'\n bt  ',b); break
