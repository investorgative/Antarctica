import numpy as np, pandas as pd, sys
res=pd.read_pickle(sys.argv[1]); YEARS=list(range(2020,2027))
P=[r[0] for r in res]; M=np.array([r[1]['m'] for r in res]); YR=np.array([r[1]['yR'] for r in res]); YW=np.array([r[1]['yW'] for r in res]); YN=np.array([r[1]['yN'] for r in res])
per=pd.period_range('2020-04','2026-11',freq='M'); yrs=np.array([p.year for p in per])
def score(mask_m, mask_y, wrmin):
    m=M[:,mask_m]; mu=m.mean(1); sd=m.std(1)+1e-9
    wr=YW[:,mask_y].sum(1)/np.maximum(1,YN[:,mask_y].sum(1))*100
    # yearly sharpe consistency
    ys=[]
    for Y in np.array(YEARS)[mask_y]:
        mm=M[:,yrs==Y]; ys.append(mm.mean(1)/(mm.std(1)+1e-9))
    ys=np.array(ys).T
    sc=mu/sd + 0.5*ys.min(1)
    return np.where(wr>=wrmin, sc, -9), wr, mu/sd
for wrmin in [36,40,45]:
    oos=[]; wins=0; n=0; lines=[]
    for Y in [2022,2023,2024,2025,2026]:
        sc,_,_=score(yrs<Y, np.array(YEARS)<Y, wrmin); i=int(np.argmax(sc))
        mm=M[i,yrs==Y]; oos.extend(mm.tolist()); j=YEARS.index(Y); wins+=YW[i,j]; n+=YN[i,j]
        lines.append(f"   {Y}: R/mo {mm.mean():5.2f} sharpe {mm.mean()/mm.std():5.2f} wr {YW[i,j]/max(1,YN[i,j])*100:4.1f}  {P[i]}")
    oos=np.array(oos)
    print(f"WF wrmin {wrmin}: OOS R/month {oos.mean():.2f}  monthly sharpe {oos.mean()/oos.std():.2f}  green months {(oos>0).mean()*100:.0f}%  winrate {wins/n*100:.1f}%  n {n}")
    print("\n".join(lines))
sc,wr,sh=score(np.ones(len(per),bool), np.ones(7,bool), 40)
df=pd.DataFrame(P); df['score']=sc; df['wr']=wr; df['sh']=sh; df['Rm']=M.mean(1); df['ddR']=[(np.maximum.accumulate(np.cumsum(m))-np.cumsum(m)).max() for m in M]
df['minyR']=YR.min(1)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(df.sort_values('score',ascending=False).head(25).round(2))
