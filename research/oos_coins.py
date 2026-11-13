import pandas as pd, numpy as np
from fam import *
NEW=ALL[3:]
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
for f in ['ex2.pkl','ex3.pkl']:
    df=pd.read_pickle(f); df['minAB']=df[['RA','RB']].min(axis=1)
    top=pd.concat([df[df.wr>35].sort_values('minAB',ascending=False).head(12), df.sort_values('minAB',ascending=False).head(6)])
    for _,p in top.iterrows():
        p=p.to_dict(); 
        for k in ['tf','N','trend','ema_n','atr_n','mb']: p[k]=int(p[k])
        R=[];W=[];n=0;months=0
        per=[]
        for s in NEW:
            tr,ex=run(s,p)
            d,_=get(s,p['tf']); m=(d['ts'][-1]-d['ts'][0])/(30.4375*864e5)
            per.append(tr[:,3].sum()/m)
            R.append(tr[:,3]); 
        R=np.concatenate(R)
        print(f"{f} tf{p['tf']} N{p['N']} imp{p['imp']} tr{p['trend']} sl{p['sl']} tp{p['tp']} fr{p['frac']} be{p['be']} tra{p['tra']} | in-sample 3c: wr {p['wr']:.1f} Rm {p['Rm']:.2f} | NEW 9c: n {len(R)} wr {(R>0).mean()*100:.1f} Rm {sum(per):.2f} neg-coins {sum(x<0 for x in per)}")
