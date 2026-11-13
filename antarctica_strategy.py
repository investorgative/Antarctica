#!/usr/bin/env python3
"""
ANTARCTICA strategy "Aurora Ensemble" - target portfolio weights for a 12-coin Bybit perpetual book (4h candles).

Three sleeves that earn money for different reasons, each run as an ENSEMBLE of many parameter sets
(no single "best" parameter set is chosen - that was the main over-fitting trap found in the research):

  TREND  (time-series momentum)   long coins whose price is above its price L days ago, short the others,
                                  sized by inverse volatility.            L in {14,21,30,45,60} days
  CARRY  (funding)                long the 3 coins with the lowest average funding rate, short the 3 with
                                  the highest (shorts collect funding).   L in {3,7,14} days
  XS-MOM (cross-sectional)        long the 3 strongest coins, short the 3 weakest.  L in {7,14,21,30} days

Every parameter set is re-balanced on a fixed clock (every 18 or 42 candles = 3 or 7 days), normalised to 1x gross,
and scaled to 20 % annual volatility using ONLY past returns. The sets are averaged per sleeve, each sleeve is
scaled to 20 % volatility again, the three sleeves are averaged and the book is multiplied by `scale` (default 4).

NO LOOK-AHEAD: the weight row t only uses candles and funding up to the close of candle t; it is traded at the
open of candle t+1. Every rolling statistic is computed per window (sliding windows), so the value at candle t
is identical no matter where the data array starts (live == backtest).
"""
import numpy as np
import pandas as pd

BARS_PER_DAY = 6
BPY = 365 * BARS_PER_DAY

DEFAULT_PARAMS = {
    "lt": [14, 21, 30, 45, 60],   # trend look-backs (days)
    "lc": [3, 7, 14],             # carry funding averages (days)
    "lx": [7, 14, 21, 30],        # cross-sectional momentum look-backs (days)
    "hs": [18, 42],               # re-balance clocks (candles)
    "phases": 3,                  # each clock is run with this many staggered start offsets (removes timing luck)
    "n_xs": 3,                    # coins per side in carry / xs sleeves
    "w_ts": 1.0, "w_carry": 1.0, "w_xs": 1.0,   # sleeve weights
    "vol_days": 30,               # per-coin volatility window (trend sizing, availability)
    "sleeve_vol": 0.20,           # annual vol target per parameter set and per sleeve
    "sleeve_vol_days": 60,        # window of the vol estimate
    "lev_cap": 8.0,               # max scale-up of a unit set/sleeve
    "scale": 4.0,                 # final book multiplier (4x = the chosen risk level)
}


def clean_params(raw=None, base=None):
    p = dict(base or DEFAULT_PARAMS)
    raw = raw or {}

    def lst(k, lo, hi):
        v = raw.get(k)
        if v is None:
            return
        if isinstance(v, str):
            v = [x for x in v.replace(";", ",").split(",") if x.strip()]
        try:
            out = sorted({int(min(hi, max(lo, float(x)))) for x in v})
            if out:
                p[k] = out
        except Exception:
            pass

    def num(k, lo, hi):
        if raw.get(k) is not None:
            try:
                p[k] = float(min(hi, max(lo, float(raw[k]))))
            except Exception:
                pass

    lst("lt", 2, 365)
    lst("lc", 1, 60)
    lst("lx", 2, 180)
    lst("hs", 1, 500)
    num("phases", 1, 12)
    p["phases"] = int(p["phases"])
    num("n_xs", 1, 6)
    p["n_xs"] = int(p["n_xs"])
    for k in ("w_ts", "w_carry", "w_xs"):
        num(k, 0, 10)
    num("vol_days", 5, 120)
    num("sleeve_vol", 0.02, 2)
    num("sleeve_vol_days", 10, 365)
    num("lev_cap", 0.5, 50)
    num("scale", 0, 20)
    return {k: p[k] for k in DEFAULT_PARAMS}


def warmup_bars(p):
    p = clean_params(p)
    longest = max(max(p["lt"]), max(p["lx"]), max(p["lc"])) * BARS_PER_DAY
    return int(longest + 2 * p["vol_days"] * BARS_PER_DAY + 2 * p["sleeve_vol_days"] * BARS_PER_DAY + max(p["hs"]) + 20)


# ---------------------------------------------------------------- exact rolling helpers
def roll_std(a, n):
    """sample std over [i-n+1 .. i] per column; nan if the window has a nan or is not full"""
    a = np.asarray(a, float)
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n, axis=0)
        out[n - 1:] = np.std(w, axis=-1, ddof=1)
    return out


def roll_mean(a, n):
    a = np.asarray(a, float)
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n, axis=0)
        out[n - 1:] = np.mean(w, axis=-1)
    return out


def shift(a, k):
    out = np.full(a.shape, np.nan)
    if k < a.shape[0]:
        out[k:] = a[:-k] if k else a
    return out


def _hold(raw, reb):
    """value of `raw` at the last re-balance candle (nan values at a re-balance candle keep the old value)"""
    df = pd.DataFrame(np.where(reb[:, None], raw, np.nan))
    return df.ffill().to_numpy()


def _unit(W, avail):
    W = np.where(avail, W, 0.0)
    g = np.nansum(np.abs(W), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = W / np.where(g > 0, g, np.nan)[:, None]
    return np.nan_to_num(out, nan=0.0)


def _vol_scale(W, R0, p):
    """scale a unit weight book to the annual vol target using its own PAST frictionless returns"""
    Wp = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    r0 = np.sum(Wp * R0, axis=1)
    sd = roll_std(r0, int(p["sleeve_vol_days"] * BARS_PER_DAY)) * np.sqrt(BPY)
    with np.errstate(invalid="ignore", divide="ignore"):
        lev = np.minimum(p["sleeve_vol"] / sd, p["lev_cap"])
    lev = np.nan_to_num(lev, nan=0.0, posinf=0.0)
    return W * lev[:, None]


def _rank(x, avail):
    """row-wise rank (1 = smallest) among available coins, average ranks for ties; nan where not available"""
    df = pd.DataFrame(np.where(avail, x, np.nan))
    return df.rank(axis=1, method="average").to_numpy()


def compute(T, C, FR, p, tf_ms, detail=False):
    """T: candle open times (n), C: closes (n x k, nan = no data), FR: funding rate sum per candle (n x k).
    Returns target weights (n x k) decided at the close of each candle (and per-sleeve weights if detail)."""
    p = clean_params(p)
    C = np.asarray(C, float)
    FR = np.nan_to_num(np.asarray(FR, float), nan=0.0)
    n, k = C.shape
    R = C / shift(C, 1) - 1
    R0 = np.nan_to_num(R, nan=0.0)
    vb = int(p["vol_days"] * BARS_PER_DAY)
    vol = roll_std(R, vb)
    avail = ~np.isnan(C) & ~np.isnan(shift(C, vb))
    cnt = avail.sum(axis=1)
    clock = (np.asarray(T, np.int64) // int(tf_ms))
    nx = p["n_xs"]

    def ensemble(raws):
        acc = np.zeros((n, k))
        m = 0
        for raw, H in raws:
            for ph in range(p["phases"]):
                reb = ((clock + (ph * H) // p["phases"]) % H) == 0
                acc += _vol_scale(_unit(_hold(raw, reb), avail), R0, p)
                m += 1
        acc /= max(1, m)
        return _vol_scale(acc, R0, p)

    sleeves = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        if p["w_ts"] > 0:
            raws = []
            for L in p["lt"]:
                mom = C / shift(C, L * BARS_PER_DAY) - 1
                raw = np.sign(mom) / vol
                raws += [(raw, H) for H in p["hs"]]
            sleeves["ts"] = ensemble(raws) * p["w_ts"]
        if p["w_carry"] > 0:
            raws = []
            for L in p["lc"]:
                fa = roll_mean(FR, L * BARS_PER_DAY)
                rk = _rank(fa, avail)
                raw = (rk <= nx).astype(float) - (rk > (cnt[:, None] - nx)).astype(float)
                raws += [(raw, H) for H in p["hs"]]
            sleeves["carry"] = ensemble(raws) * p["w_carry"]
        if p["w_xs"] > 0:
            raws = []
            for L in p["lx"]:
                mom = C / shift(C, L * BARS_PER_DAY) - 1
                rk = _rank(mom, avail)
                raw = (rk > (cnt[:, None] - nx)).astype(float) - (rk <= nx).astype(float)
                raws += [(raw, H) for H in p["hs"]]
            sleeves["xs"] = ensemble(raws) * p["w_xs"]
    tot_w = p["w_ts"] + p["w_carry"] + p["w_xs"]
    W = sum(sleeves.values()) / tot_w * p["scale"] if sleeves and tot_w > 0 else np.zeros((n, k))
    W = np.nan_to_num(W, nan=0.0)
    if detail:
        return W, {s: np.nan_to_num(v / tot_w * p["scale"]) for s, v in sleeves.items()}
    return W


def panel(datas, funding, symbols, tf_ms, start=None, end=None):
    """align candle dicts {sym: {ts,o,h,l,c,v}} on one time grid; funding {sym: {ts: rate}} summed per candle"""
    ts_all = sorted(set().union(*[set(datas[s]["ts"].tolist()) for s in symbols if s in datas]))
    if start is not None:
        ts_all = [t for t in ts_all if t >= start]
    if end is not None:
        ts_all = [t for t in ts_all if t <= end]
    T = np.array(ts_all, dtype=np.int64)
    n, k = len(T), len(symbols)
    O = np.full((n, k), np.nan)
    C = np.full((n, k), np.nan)
    H = np.full((n, k), np.nan)
    Lo = np.full((n, k), np.nan)
    FR = np.zeros((n, k))
    if n == 0:
        return T, O, H, Lo, C, FR
    pos = {t: i for i, t in enumerate(ts_all)}
    for j, s in enumerate(symbols):
        d = datas.get(s)
        if d is None:
            continue
        for i0, t in enumerate(d["ts"].tolist()):
            i = pos.get(t)
            if i is not None:
                O[i, j], H[i, j], Lo[i, j], C[i, j] = d["o"][i0], d["h"][i0], d["l"][i0], d["c"][i0]
        for ft, r in (funding.get(s) or {}).items():
            i = (ft - T[0]) // tf_ms
            if 0 <= i < n and T[i] <= ft < T[i] + tf_ms:
                FR[i, j] += r
            elif ft >= T[0]:
                ii = int(np.searchsorted(T, ft, side="right")) - 1
                if 0 <= ii < n and ft < T[ii] + tf_ms:
                    FR[ii, j] += r
    return T, O, H, Lo, C, FR
