import os, sys, json, hmac, hashlib, threading, time, shutil
D='/home/user/execdata'; shutil.rmtree(D,ignore_errors=True); os.makedirs(D)
os.environ['ANTARCTICA_DATA']=D; os.environ['ANTARCTICA_CONFIG']=D+'/config.json'
sys.path.insert(0,'/home/user/Antarctica')
from flask import Flask, request, jsonify
import antarctica_core as core, antarctica_exec as ex
KEY,SEC='testkey123456','testsecret'
app=Flask('mock'); POS={}; CALLS=[]
def chk():
    ts=request.headers['X-BAPI-TIMESTAMP']; payload=request.query_string.decode() if request.method=='GET' else request.get_data(as_text=True)
    exp=hmac.new(SEC.encode(),f"{ts}{KEY}{ex.RECV}{payload}".encode(),hashlib.sha256).hexdigest()
    assert request.headers['X-BAPI-SIGN']==exp and request.headers['X-BAPI-API-KEY']==KEY, 'bad signature'
@app.get('/v5/market/time')
def t(): return jsonify({'retCode':0,'result':{'timeNano':str(time.time_ns())}})
@app.get('/v5/account/wallet-balance')
def wb(): chk(); return jsonify({'retCode':0,'result':{'list':[{'totalEquity':'10000','totalAvailableBalance':'9000'}]}})
@app.get('/v5/position/list')
def pl(): chk(); return jsonify({'retCode':0,'result':{'list':list(POS.values())}})
@app.post('/v5/position/set-leverage')
def sl(): chk(); CALLS.append(('lev',request.json)); return jsonify({'retCode':0,'result':{}})
@app.post('/v5/order/create')
def oc():
    chk(); j=request.json; CALLS.append(('order',j)); s=j['symbol']
    if j.get('reduceOnly'): POS.pop(s,None)
    else: POS[s]={'symbol':s,'side':j['side'],'size':j['qty'],'avgPrice':'100','stopLoss':j.get('stopLoss','0'),'unrealisedPnl':'0'}
    return jsonify({'retCode':0,'result':{'orderId':'o1'}})
@app.post('/v5/position/trading-stop')
def ts_(): chk(); j=request.json; CALLS.append(('stop',j)); POS[j['symbol']]['stopLoss']=j['stopLoss']; return jsonify({'retCode':0,'result':{}})
threading.Thread(target=lambda: app.run(port=8799),daemon=True).start(); time.sleep(1)
ex.HOSTS['demo']='http://127.0.0.1:8799'
ex.save_keys('demo',KEY,SEC); ex.set_control('demo',armed=True)
cfg=core.load_config()
class FakeTrader: pass
tr=FakeTrader(); tr.cfg=cfg; tr.syms=cfg['symbols']; tr.eng=core.Engine(cfg['strategy'],cfg)
e=ex.Executor('demo',tr)
print('test_connection', ex.test_connection('demo'))
pos={'id':1,'sym':'ETHUSDT','dir':1,'qty':0.5,'entry':3000.0,'stop':2900.0,'dist':100.0}
e.on_event({'type':'open','sym':'ETHUSDT','ts':0,'pos':pos})
e.on_event({'type':'trail','sym':'ETHUSDT','ts':0,'id':1,'new':2950.123456,'old':2900})
print('position after open/trail:', POS.get('ETHUSDT'))
e.on_event({'type':'close','sym':'ETHUSDT','ts':0,'trade':{'reason':'trail'}})
print('position after close:', POS.get('ETHUSDT'))
e.on_event({'type':'open','sym':'BTCUSDT','ts':0,'pos':dict(pos,sym='BTCUSDT',entry=80000.0,stop=78000.0,dist=2000.0)})
ex.set_control('demo',flatten=True); e.last_sync=0; e.tick(); print('after flatten:', POS, 'armed:', e.armed())
for c in CALLS: print(c)
# live must never trade
ex.save_keys('live',KEY,SEC); ex.set_control('live',armed=True); L=ex.Executor('live',tr); n=len(CALLS)
L.on_event({'type':'open','sym':'ETHUSDT','ts':0,'pos':pos}); print('live calls made while LIVE_ENABLED=False:', len(CALLS)-n)
