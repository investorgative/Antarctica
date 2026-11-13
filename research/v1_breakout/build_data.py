import zipfile, glob, io, numpy as np, pandas as pd, os
RAW='/home/user/data/raw'; OUT='/home/user/data'
import sys
for sym in sys.argv[1:]:
    frames=[]
    for f in sorted(glob.glob(f'{RAW}/{sym}-1m-*.zip')):
        with zipfile.ZipFile(f) as z:
            raw=z.read(z.namelist()[0]).decode()
        first=raw.split('\n',1)[0]
        df=pd.read_csv(io.StringIO(raw), header=0 if 'open' in first else None, usecols=[0,1,2,3,4,5])
        df.columns=['ts','o','h','l','c','v']
        frames.append(df)
    d=pd.concat(frames).drop_duplicates('ts').sort_values('ts')
    d['ts']=d['ts'].astype('int64')
    d=d[d.ts<4e12]  # guard against microsecond timestamps
    np.savez(f'{OUT}/{sym}_1m.npz', **{k:d[k].values for k in d.columns})
    fr=[]
    for f in sorted(glob.glob(f'{RAW}/{sym}-fundingRate-*.zip')):
        with zipfile.ZipFile(f) as z: raw=z.read(z.namelist()[0]).decode()
        first=raw.split('\n',1)[0]
        x=pd.read_csv(io.StringIO(raw), header=0 if 'calc' in first or 'funding' in first.lower() else None)
        x=x.iloc[:, [0,2]]; x.columns=['ts','rate']; fr.append(x)
    fr=pd.concat(fr); fr['ts']=(fr.ts//60000*60000).astype('int64')
    np.savez(f'{OUT}/{sym}_funding.npz', ts=fr.ts.values, rate=fr.rate.values.astype(float))
    gaps=np.diff(d.ts.values); 
    print(sym, len(d), pd.to_datetime(d.ts.iloc[0],unit='ms'), pd.to_datetime(d.ts.iloc[-1],unit='ms'), 'gaps>1m:', (gaps>60000).sum(), 'funding', len(fr))
