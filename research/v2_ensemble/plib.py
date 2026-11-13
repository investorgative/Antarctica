"""Vectorized portfolio research (v2). Weights decided at the CLOSE of bar t are held during bar t+1.
pnl_t+1 = sum_i w_i,t * r_i,t+1 - cost * sum_i |w_i,t - w_i,t-1 drifted| - funding.
Weights are fractions of equity (1.0 = 100% notional). Costs: taker 0.055% + slippage 0.01% per unit turnover."""
import numpy as np, pandas as pd
D = '/home/user/data'
ALL = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'BNBUSDT', 'ADAUSDT', 'LINKUSDT', 'LTCUSDT', 'AVAXUSDT',
       'DOTUSDT', 'BCHUSDT']
COST = 0.00055 + 0.0001
_c = {}


def panel(tf_min=240, syms=ALL):
    k = (tf_min, tuple(syms))
    if k in _c:
        return _c[k]
    C, F = {}, {}
    for s in syms:
        z = np.load(f'{D}/{s}_1m.npz')
        c = pd.Series(z['c'], index=pd.to_datetime(z['ts'], unit='ms'))
        C[s] = c.resample(f'{tf_min}min').last()
        f = np.load(f'{D}/{s}_funding.npz')
        fr = pd.Series(f['rate'], index=pd.to_datetime(f['ts'], unit='ms').floor('min'))
        fr = fr[~fr.index.duplicated()]
        F[s] = fr.resample(f'{tf_min}min').sum()  # funding charged during the bar (rate * notional)
    C = pd.DataFrame(C)
    F = pd.DataFrame(F).reindex(C.index).fillna(0.0)
    R = C.pct_change()
    _c[k] = (C, R, F)
    return C, R, F


def backtest(W, R, F, cost=COST):
    """W: target weights at bar close (index aligned with R). returns per-bar net returns"""
    W = W.fillna(0.0)
    Wp = W.shift(1).fillna(0.0)  # held during the bar
    gross = (Wp * R.fillna(0.0)).sum(1)
    # turnover: change from drifted previous weights to new target
    drift = Wp * (1 + R.fillna(0.0))
    drift = drift.div((1 + gross), axis=0)
    turn = (W - drift).abs().sum(1)
    fund = (Wp * F).sum(1)  # longs pay positive funding
    return gross - cost * turn - fund, turn


def stats(r, bars_per_year, label=''):
    r = r.dropna()
    eq = (1 + r).cumprod()
    dd = (1 - eq / eq.cummax()).max()
    m = (1 + r).groupby(r.index.to_period('M')).prod() - 1
    yrs = len(r) / bars_per_year
    sh = r.mean() / r.std() * np.sqrt(bars_per_year) if r.std() > 0 else 0
    return dict(label=label, sharpe=sh, cagr=eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0, dd=dd, avgm=m.mean(),
                geom=eq.iloc[-1] ** (1 / len(m)) - 1, posm=(m > 0).mean(), worstm=m.min())


def by_year(r, bpy):
    out = {}
    for y, g in r.groupby(r.index.year):
        out[y] = g.mean() / g.std() * np.sqrt(bpy) if g.std() > 0 else 0
    return out


def voltarget(r_unit, target_ann, bpy, lookback):
    """scale a unit strategy return stream to a target annual vol using PAST realized vol only"""
    vol = r_unit.rolling(lookback).std().shift(1) * np.sqrt(bpy)
    lev = (target_ann / vol).clip(upper=10)
    return lev
