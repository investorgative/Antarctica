import os, sys, numpy as np
os.environ['ANTARCTICA_DATA']='/home/user/devdata'; os.environ['ANTARCTICA_CONFIG']='/home/user/devdata/config.json'
sys.path.insert(0,'/home/user/Antarctica'); sys.path.insert(0,'/home/user/research')
import antarctica_core as core
from rlib import load
st=core.Store()
for s in core.SYMBOLS:
    d=load(s,360)
    rows=list(zip(d['ts'].tolist(),d['o'].tolist(),d['h'].tolist(),d['l'].tolist(),d['c'].tolist(),d['v'].tolist()))
    print(s, st.insert(s,360,rows))
    f=np.load(f'/home/user/data/{s}_funding.npz'); st.insert_funding(s,dict(zip(f['ts'].tolist(),f['rate'].tolist())))
