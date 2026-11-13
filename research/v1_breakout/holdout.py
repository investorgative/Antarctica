import numpy as np, pandas as pd, multiprocessing as mp, sys
from fam import *
from search_wf import sample
CUT=pd.Timestamp('2024-01-01').value//10**6
def ev(p):
    out={}
    for s in ALL:
        tr,ex=run(s,p)
        a=tr[ex<CUT]; b=tr[ex>=CUT]
        out[s]=(a[:,3].sum(),(a[:,3]>0).sum(),len(a),b[:,3].sum(),(b[:,3]>0).sum(),len(b), a[:,3][pd.to_datetime(ex[ex<CUT],unit='ms').year.values==2022].sum() if len(a) else 0)
    return p,out
if __name__=='__main__':
    rng=np.random.default_rng(11)
    ps=[sample(rng) for _ in range(3000)]
    with mp.Pool(4) as pool: res=pool.map(ev,ps,chunksize=8)
    pd.to_pickle(res,'/home/user/research/holdout.pkl')
    TRAIN=['BTCUSDT','ETHUSDT','SOLUSDT']
    rows=[]
    for p,o in res:
        tR=sum(o[s][0] for s in TRAIN); tW=sum(o[s][1] for s in TRAIN); tN=sum(o[s][2] for s in TRAIN)
        t22=sum(o[s][6] for s in TRAIN); tmin=min(o[s][0] for s in TRAIN)
        hR=sum(o[s][3] for s in ALL); hW=sum(o[s][4] for s in ALL); hN=sum(o[s][5] for s in ALL)
        rows.append(dict(**p,train_Rm=tR/48,train_wr=tW/max(tN,1)*100,train_2022=t22,train_mincoin=tmin,hold_Rm=hR/34.4,hold_wr=hW/max(hN,1)*100,hold_n=hN,hold_neg=sum(o[s][3]<0 for s in ALL)))
    df=pd.DataFrame(rows)
    pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
    sel=df[(df.train_wr>=37)&(df.train_2022>0)&(df.train_mincoin>0)].sort_values('train_Rm',ascending=False)
    print('selected on train only (top 15), with holdout results:'); print(sel.head(15))
    print('\nholdout stats of top-10 train picks: mean Rm %.2f, mean wr %.1f'%(sel.head(10).hold_Rm.mean(),sel.head(10).hold_wr.mean()))
    print('all configs holdout: median Rm %.2f; share profitable %.0f%%'%(df.hold_Rm.median(),(df.hold_Rm>0).mean()*100))
