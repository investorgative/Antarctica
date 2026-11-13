"""Research library: fast, causal backtests of many strategy variants.
Conventions: signal computed at the CLOSE of bar i, entry at OPEN of bar i+1. Stops/TP checked intrabar,
stop assumed first when both are touched in the same bar (conservative)."""
import numpy as np, pandas as pd
from numba import njit

D = '/home/user/data'
TAKER, MAKER, SLIP = 0.00055, 0.0002, 0.0001
_cache = {}


def load(sym, tf_min):
    key = (sym, tf_min)
    if key in _cache:
        return _cache[key]
    z = np.load(f'{D}/{sym}_1m.npz')
    df = pd.DataFrame({k: z[k] for k in ('ts', 'o', 'h', 'l', 'c', 'v')})
    df['g'] = df.ts // (tf_min * 60000)
    a = df.groupby('g').agg(ts=('ts', 'first'), o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'),
                            v=('v', 'sum'))
    a['ts'] = (a.index.values * tf_min * 60000).astype(np.int64)
    f = np.load(f'{D}/{sym}_funding.npz')
    fr = pd.Series(f['rate'], index=f['ts'])
    fr = fr[~fr.index.duplicated()]
    # funding rate charged at the bar whose open time == funding time (ts multiple of 8h normally)
    a['fund'] = a.ts.map(fr).fillna(0.0).values
    out = {k: a[k].values.astype(np.float64) if k != 'ts' else a[k].values for k in a.columns}
    _cache[key] = out
    return out


# ------------------------------------------------------------------ indicators (causal)
def atr(d, n):
    h, l, c = d['h'], d['l'], d['c']
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().values


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def hh(x, n):  # highest of the PREVIOUS n bars (excludes current)
    return pd.Series(x).rolling(n).max().shift(1).values


def ll(x, n):
    return pd.Series(x).rolling(n).min().shift(1).values


def rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).values


# ------------------------------------------------------------------ simulator
@njit(cache=True)
def sim(o, h, l, c, fund, atr_, sig, sl_atr, tp_r, tp_frac, be_r, tr_atr, tr_lo, tr_hi, max_bars, start_i,
        taker, maker, slip):
    """returns trades array: [entry_i, exit_i, dir, R_net, win_flag, entry, stop_dist, fees_R]
    sig[i] in {-1,0,1}: decided at close of i. tr_lo/tr_hi: optional structure trail levels (nan to disable)
    tp_frac: fraction closed at TP (1 = full). be_r: move stop to entry(+fees) once MFE >= be_r R (0=off)
    tr_atr: chandelier trail distance in ATR from best extreme since entry (0=off)"""
    n = len(o)
    out = np.zeros((n // 2 + 10, 8))
    k = 0
    pos = 0
    i = start_i
    entry = 0.0; stop = 0.0; dist = 0.0; best = 0.0; tp = 0.0; rem = 1.0; realized = 0.0; feesR = 0.0
    ei = 0
    while i < n - 1:
        if pos == 0:
            s = sig[i]
            if s != 0 and atr_[i] == atr_[i] and atr_[i] > 0:
                pos = int(s); ei = i + 1
                entry = o[ei] * (1 + slip * pos)
                dist = sl_atr * atr_[i]
                stop = entry - pos * dist
                tp = entry + pos * tp_r * dist if tp_r > 0 else 0.0
                best = entry; rem = 1.0; realized = 0.0
                feesR = taker * entry / dist
                realized = -feesR
                i = ei
                # process the entry bar itself (no gap check on open)
            else:
                i += 1
                continue
        # in position at bar i
        exited = False
        # funding at the open of bar i (if held from a previous bar)
        if i > ei and fund[i] != 0.0:
            fr = fund[i] * o[i] / dist * pos * rem
            realized -= fr
        # stop check
        px = 0.0
        if pos == 1:
            if i != ei and o[i] <= stop:
                px = o[i] * (1 - slip)
            elif l[i] <= stop:
                px = stop * (1 - slip)
        else:
            if i != ei and o[i] >= stop:
                px = o[i] * (1 + slip)
            elif h[i] >= stop:
                px = stop * (1 + slip)
        if px != 0.0:
            realized += rem * ((px - entry) * pos / dist - taker * px / dist)
            exited = True
        else:
            # take profit (limit, maker)
            if tp_r > 0 and rem > 0:
                hit = (pos == 1 and h[i] >= tp) or (pos == -1 and l[i] <= tp)
                if hit:
                    tpx = tp
                    if i != ei and ((pos == 1 and o[i] > tp) or (pos == -1 and o[i] < tp)):
                        tpx = o[i]
                    q = tp_frac if rem >= 1.0 else rem
                    realized += q * ((tpx - entry) * pos / dist - maker * tpx / dist)
                    rem -= q
                    if rem <= 1e-9:
                        exited = True
                        px = tpx
            if not exited:
                # update extremes at close, trail
                if pos == 1:
                    if h[i] > best: best = h[i]
                else:
                    if l[i] < best: best = l[i]
                ns = stop
                if be_r > 0 and (best - entry) * pos >= be_r * dist:
                    b = entry + pos * (2 * taker * entry)
                    if (b - ns) * pos > 0: ns = b
                if tr_atr > 0 and atr_[i] == atr_[i]:
                    t = best - pos * tr_atr * atr_[i]
                    if (t - ns) * pos > 0: ns = t
                if pos == 1 and tr_lo[i] == tr_lo[i] and tr_lo[i] > ns and tr_lo[i] < c[i]: ns = tr_lo[i]
                if pos == -1 and tr_hi[i] == tr_hi[i] and tr_hi[i] < ns and tr_hi[i] > c[i]: ns = tr_hi[i]
                # never trail through the close
                if (c[i] - ns) * pos > 0:
                    stop = ns
                if max_bars > 0 and i - ei + 1 >= max_bars:
                    px = c[i] * (1 - slip * pos)
                    realized += rem * ((px - entry) * pos / dist - taker * px / dist)
                    exited = True
        if exited:
            out[k, 0] = ei; out[k, 1] = i; out[k, 2] = pos; out[k, 3] = realized
            out[k, 4] = 1.0 if realized > 0 else 0.0; out[k, 5] = entry; out[k, 6] = dist; out[k, 7] = feesR
            k += 1
            pos = 0
            # a new signal can be taken on the close of the exit bar
        i += 1
    return out[:k]


def nanarr(n):
    return np.full(n, np.nan)


def summarize(tr, d, start_ts=None, end_ts=None, risk=0.02, label=''):
    """tr from sim; returns dict with R stats and compounded monthly returns at fixed risk"""
    ts = d['ts']
    if len(tr) == 0:
        return dict(n=0, wr=0, R=0, Rm=0, avgm=0, geom=0, posm=0, dd=0, label=label)
    ex = ts[tr[:, 1].astype(int)]
    R = tr[:, 3]
    s0 = start_ts if start_ts is not None else ts[0]
    e0 = end_ts if end_ts is not None else ts[-1]
    months = (e0 - s0) / (30.4375 * 86400000)
    # compounded equity at fixed fractional risk
    eq = np.cumprod(1 + risk * R)
    peak = np.maximum.accumulate(np.r_[1, eq])
    dd = ((peak[1:] - eq) / peak[1:]).max() if len(eq) else 0
    m = pd.Series(1 + risk * R, index=pd.to_datetime(ex, unit='ms')).groupby(pd.Grouper(freq='MS')).prod()
    allm = pd.date_range(pd.to_datetime(s0, unit='ms').to_period('M').to_timestamp(),
                         pd.to_datetime(e0, unit='ms'), freq='MS')
    m = m.reindex(allm, fill_value=1.0) - 1
    return dict(label=label, n=len(R), wr=(R > 0).mean() * 100, R=R.sum(), Rm=R.sum() / months,
                avgR=R.mean(), avgm=m.mean() * 100, medm=m.median() * 100, geom=(eq[-1] ** (1 / months) - 1) * 100,
                posm=(m > 0).mean() * 100, dd=dd * 100, worstm=m.min() * 100)
