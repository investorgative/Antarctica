"""Limit-order mean reversion research. Decision at close of bar i: place limit buy at mid - k*sd (and/or sell at mid + k*sd)
valid for bar i+1 only (re-placed every bar). Fill only if price trades THROUGH the limit by `thru` (fraction).
Exit: limit TP at the band middle computed at entry decision (maker), stop at entry -/+ s*sd (taker, stop first if both
touched in one bar), time stop after mb bars at close (taker)."""
import numpy as np, pandas as pd, sys, multiprocessing as mp
from numba import njit
from rlib import load, ema, TAKER, MAKER, SLIP


@njit(cache=True)
def sim_mr(o, h, l, c, fund, mid, sd, okL, okS, k, s_sd, tp_frac_mid, mb, thru, taker, maker, slip, start):
    n = len(o)
    out = np.zeros((n // 2 + 10, 5))
    m = 0
    pos = 0; entry = 0.0; stop = 0.0; tp = 0.0; dist = 0.0; ei = 0
    i = start
    while i < n - 1:
        if pos == 0:
            if sd[i] != sd[i] or sd[i] <= 0:
                i += 1; continue
            bl = mid[i] - k * sd[i]; bs = mid[i] + k * sd[i]
            j = i + 1
            filled = 0
            if okL[i] and l[j] < bl * (1 - thru) and o[j] > bl:
                filled = 1; entry = bl
            elif okS[i] and h[j] > bs * (1 + thru) and o[j] < bs:
                filled = -1; entry = bs
            if filled == 0:
                i += 1; continue
            pos = filled; ei = j
            dist = s_sd * sd[i]
            stop = entry - pos * dist
            tp = entry + pos * (mid[i] - entry) * pos * tp_frac_mid  # fraction of the way back to the mid
            if (tp - entry) * pos <= 0:
                tp = entry + pos * 0.1 * dist
            realized = -maker * entry / dist
            # entry bar: after fill, assume worst case: stop checked first, TP only if close beyond tp
            if (pos == 1 and l[j] <= stop) or (pos == -1 and h[j] >= stop):
                px = stop * (1 - slip * pos)
                realized += (px - entry) * pos / dist - taker * px / dist
                out[m, 0] = ei; out[m, 1] = j; out[m, 2] = pos; out[m, 3] = realized; out[m, 4] = dist / entry
                m += 1; pos = 0; i = j; continue
            if (pos == 1 and c[j] > tp) or (pos == -1 and c[j] < tp):
                realized += (tp - entry) * pos / dist - maker * tp / dist
                out[m, 0] = ei; out[m, 1] = j; out[m, 2] = pos; out[m, 3] = realized; out[m, 4] = dist / entry
                m += 1; pos = 0; i = j; continue
            i = j + 1
            cur = realized
            continue
        # in position at bar i (> entry bar)
        if fund[i] != 0.0:
            cur -= fund[i] * o[i] / dist * pos
        px = 0.0
        if pos == 1:
            if o[i] <= stop: px = o[i] * (1 - slip)
            elif l[i] <= stop: px = stop * (1 - slip)
        else:
            if o[i] >= stop: px = o[i] * (1 + slip)
            elif h[i] >= stop: px = stop * (1 + slip)
        done = False
        if px != 0.0:
            cur += (px - entry) * pos / dist - taker * px / dist; done = True
        elif (pos == 1 and h[i] > tp * (1 + thru)) or (pos == -1 and l[i] < tp * (1 - thru)):
            tpx = tp
            if (pos == 1 and o[i] > tp) or (pos == -1 and o[i] < tp):
                tpx = o[i]
            cur += (tpx - entry) * pos / dist - maker * tpx / dist; done = True
        elif mb > 0 and i - ei >= mb:
            px = c[i] * (1 - slip * pos)
            cur += (px - entry) * pos / dist - taker * px / dist; done = True
        if done:
            out[m, 0] = ei; out[m, 1] = i; out[m, 2] = pos; out[m, 3] = cur; out[m, 4] = dist / entry
            m += 1; pos = 0
        i += 1
    return out[:m]


def prep(sym, tf, n, trend_n, mode):
    d = load(sym, tf)
    c = d['c']
    s = pd.Series(c)
    mid = s.rolling(n).mean().values
    sd = s.rolling(n).std().values
    if trend_n > 0:
        e = ema(c, trend_n)
        if mode == 'with':  # buy dips in uptrend, sell rips in downtrend
            okL, okS = c > e, c < e
        else:  # both sides regardless
            okL = np.ones(len(c), bool); okS = okL
    else:
        okL = np.ones(len(c), bool); okS = okL
    return d, mid, sd, okL, okS
