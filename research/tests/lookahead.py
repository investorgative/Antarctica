import os, sys, random, json
os.environ['ANTARCTICA_DATA']='/home/user/devdata'; os.environ['ANTARCTICA_CONFIG']='/home/user/devdata/config.json'
sys.path.insert(0,'/home/user/Antarctica')
import numpy as np, antarctica_core as core
cfg=core.load_config(); p=core.clean_params(cfg['strategy']); st=core.Store(); tf=core.tf_ms(cfg)
end=st.bounds('BTCUSDT',360)[1]+tf; start=end-730*core.DAY//tf*tf; start=start//tf*tf
lead=start-(core.warmup_bars(p)+5)*tf
datas={s:st.load(s,360,lead,end) for s in cfg['symbols']}; fund={s:st.load_funding(s,lead,end) for s in cfg['symbols']}
full=core.run_backtest(datas,p,cfg,start_ms=start,funding=fund,close_end=False)
key=lambda t:(t['sym'],t['entry_ts'],t['exit_ts'],round(t['entry'],8),round(t['exit'],8),round(t['qty'],8),round(t['pnl'],8),t['reason'])
F=[key(t) for t in full.trades]
print('full run trades',len(F))
random.seed(5); checked=0; bad=0
for k in range(25):
    cut=random.randrange(start+60*core.DAY, end-core.DAY)//tf*tf
    dd={s:{kk:(v[d['ts']<cut]) for kk,v in d.items()} for s,d in datas.items()}
    e=core.run_backtest(dd,p,cfg,start_ms=start,funding=fund,close_end=False)
    T=[key(t) for t in e.trades]
    ref=[x for x in F if x[2]<=cut-tf]  # trades closed before the last candle of the truncated data
    got=[x for x in T if x[2]<=cut-tf]
    checked+=len(ref)
    if ref!=got:
        bad+=1; print('MISMATCH at cut',cut,len(ref),len(got))
print(f'cuts 25, compared {checked} closed trades, mismatching cuts: {bad}')
