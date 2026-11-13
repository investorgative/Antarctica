import numpy as np, pandas as pd, sys
res=pd.read_pickle(sys.argv[1] if len(sys.argv)>1 else '/home/user/research/wf_res.pkl')
YEARS=list(range(2020,2027)); MONTHS=np.array([12]*6+[10.4])
R=np.array([r['R'] for r in res]); W=np.array([r['W'] for r in res]); N=np.array([r['N'] for r in res])
def pick(mask_years, wrmin, obj='rm'):
    m=np.array(mask_years)
    Rm=R[:,m].sum(1)/MONTHS[m].sum(); wr=W[:,m].sum(1)/np.maximum(1,N[:,m].sum(1))*100
    miny=(R[:,m]/MONTHS[m]).min(1)
    score=np.where((wr>=wrmin)&(miny>0), Rm + 0.5*miny, -1e9)
    return int(np.argmax(score))
for wrmin in [35,37,40]:
    tot=0;w=0;n=0;rows=[]
    for j,Y in enumerate(YEARS):
        if Y<2022: continue
        i=pick([y<Y for y in YEARS],wrmin)
        tot+=R[i,j]; w+=W[i,j]; n+=N[i,j]
        p=res[i]['p']
        rows.append(f"  {Y}: R/mo {R[i,j]/MONTHS[j]:6.2f}  wr {W[i,j]/max(1,N[i,j])*100:5.1f}  n {int(N[i,j])}  <- {p}")
    print(f"WALK-FORWARD wrmin={wrmin}: OOS R/month {tot/MONTHS[2:].sum():.2f}, OOS winrate {w/n*100:.1f}%, trades {int(n)}")
    print("\n".join(rows))
# full period table
Rm=R.sum(1)/MONTHS.sum(); wr=W.sum(1)/N.sum(1)*100; miny=(R/MONTHS).min(1)
df=pd.DataFrame([r['p'] for r in res]); df['Rm']=Rm; df['wr']=wr; df['miny']=miny; df['n']=N.sum(1); df['negcoins']=[(r['coinR']<0).sum() for r in res]
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(df[(df.wr>=36)].sort_values('Rm',ascending=False).head(20))
print('reference cfg:'); print(df.tail(1))
