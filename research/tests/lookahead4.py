import os, sys, random
os.environ['ANTARCTICA_DATA']='/home/user/devdata4'; os.environ['ANTARCTICA_CONFIG']='/home/user/devdata4/config.json'
sys.path.insert(0,'/home/user/Antarctica')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, antarctica_core as core
cfg=core.load_config(); st=core.Store(); tf=core.tf_ms(cfg)
end=st.bounds('BTCUSDT',240)[1]+tf; start=(end-1000*core.DAY)//tf*tf
datas={s:st.load(s,240,0,end) for s in cfg['symbols']}; fund={s:st.load_funding(s,0,end) for s in cfg['symbols']}
full=core.run_backtest(datas,cfg['strategy'],cfg,start_ms=start,funding=fund,close_end=False)
key=lambda t:(t['sym'],t['entry_ts'],t['exit_ts'],round(t['entry'],8),round(t['exit'],8),round(t['pnl'],8))
F=[key(t) for t in full.trades]; FC={c[0]:c[1] for c in full.curve}
random.seed(3); bad=0; ntr=0; npt=0
for k in range(20):
    cut=random.randrange(start+100*core.DAY,end-core.DAY)//tf*tf
    dd={s:{kk:v[d['ts']<cut] for kk,v in d.items()} for s,d in datas.items()}
    fd={s:{t:r for t,r in f.items() if t<cut} for s,f in fund.items()}
    e=core.run_backtest(dd,cfg['strategy'],cfg,start_ms=start,funding=fd,close_end=False)
    ref=[x for x in F if x[2]<cut]; got=[x for x in (key(t) for t in e.trades) if x[2]<cut]
    cv=[(c[0],c[1]) for c in e.curve]; okc=all(abs(FC[t]-v)<1e-6 for t,v in cv)
    ntr+=len(ref); npt+=len(cv)
    if ref!=got or not okc: bad+=1; print('MISMATCH cut',cut,len(ref),len(got),okc)
print(f'20 cuts: {ntr} closed trades and {npt} equity points compared, mismatching cuts: {bad}')
