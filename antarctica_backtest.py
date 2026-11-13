#!/usr/bin/env python3
"""
ANTARCTICA backtester + optimizer (command line and website).

  backtest   the paper trader's own Engine, candle by candle: the 12-coin long/short book, re-balanced at every
             4h open to the weights decided at the previous close, Bybit taker fee + slippage on every fill,
             real funding, lot sizes, minimum order value, leverage cap, no-trade band.
  optimizer  random search over the strategy settings with a train / test split in time: ranked on TRAIN only,
             TEST shows how each set did on data the ranking never saw.
  walkforward  choose on everything before a block, trade the block, repeat (only traded blocks are reported).

NO LOOK-AHEAD: the weights of candle t use candles and funding up to the close of t and are filled at the open
of t+1 (identical protocol in the live trader).

  python antarctica_backtest.py --days 365 --balance 1000 --scale 4
  python antarctica_backtest.py --days 1460 --kelly
  python antarctica_backtest.py --optimize --days 1460 --combos 40
  python antarctica_backtest.py --walkforward --days 2190
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

import antarctica_core as core
import antarctica_strategy as strat

log = core.setup_logging("backtest")


def load_data(cfg, start_ms, end_ms, params_list, progress=None, download=True):
    tf = int(cfg["timeframe"])
    download = download and not os.environ.get("ANTARCTICA_OFFLINE")
    warm = max(strat.warmup_bars(p) for p in params_list) + 10
    lead = start_ms - warm * tf * core.MIN
    store = core.Store(core.Bybit(cfg) if download else None)
    datas, funding = {}, {}
    syms = cfg["symbols"]
    for k, s in enumerate(syms):
        if progress:
            progress(k / len(syms) * 0.6, f"candles + funding {s}")
        if download:
            try:
                store.ensure(s, tf, lead, end_ms)
                if cfg["fees"].get("include_funding", True):
                    store.ensure_funding(s, lead, end_ms)
            except Exception as e:
                log.warning(f"{s}: download failed ({e}) - using cached data only")
        d = store.load(s, tf, lead, end_ms - tf * core.MIN)
        if len(d["ts"]):
            datas[s] = d
            funding[s] = store.load_funding(s, lead, end_ms)
    if download:
        core.refresh_instruments(cfg, log)
    return datas, funding


def _cfg_with(cfg, balance=None, scale=None, kelly=None, band=None):
    cfg = json.loads(json.dumps(cfg))
    if balance is not None:
        cfg["account"]["start_balance"] = float(balance)
    if scale is not None:
        cfg["strategy"]["scale"] = float(scale)
    if kelly is not None:
        cfg["account"]["kelly_sizing"] = bool(kelly)
    if band is not None:
        cfg["account"]["band_pct"] = float(band)
    return cfg


def backtest(cfg, days=365, balance=None, scale=None, kelly=None, band=None, params=None, end_ms=None, progress=None,
             download=True, datas=None, funding=None):
    cfg = _cfg_with(cfg, balance, scale, kelly, band)
    p = core.clean_params(params or cfg["strategy"])
    if scale is not None:
        p["scale"] = float(scale)
    tf = core.tf_ms(cfg)
    end_ms = int(end_ms or (core.now_ms() // tf * tf))
    start_ms = int(end_ms - float(days) * core.DAY) // tf * tf
    if datas is None:
        datas, funding = load_data(cfg, start_ms, end_ms, [p], progress, download)
    if not datas:
        raise RuntimeError("no candles available - is the Bybit API reachable?")
    if progress:
        progress(0.65, "computing weights + simulating")
    eng = core.run_backtest(datas, p, cfg, start_ms=start_ms, funding=funding, end_ms=end_ms - tf,
                            instr=core.instruments(cfg),
                            progress=(lambda f: progress(0.65 + 0.3 * f, "simulating")) if progress else None)
    prices = {s: float(d["c"][-1]) for s, d in datas.items()}
    b = core.bundle(eng, cfg, start_ms, eng.end_ts, prices=prices)
    b["kind"] = "backtest"
    b["request"] = {"days": days, "balance": cfg["account"]["start_balance"], "scale": p["scale"],
                    "kelly": cfg["account"]["kelly_sizing"], "band": cfg["account"]["band_pct"]}
    b["data_from"] = {s: int(d["ts"][0]) for s, d in datas.items()}
    if progress:
        progress(1.0, "done")
    return b


# ---------------------------------------------------------------- optimizer
_G = {}


def _score(st):
    """risk-adjusted: Sharpe, penalised for drawdowns beyond 30 %"""
    if st["months"] < 3:
        return -1e9
    return st["sharpe"] - max(0.0, st["max_dd_pct"] - 30) / 20


def _summ(st):
    return {k: st[k] for k in ("trades", "winrate", "avg_month", "geo_month", "max_dd_pct", "sharpe", "net_pct",
                               "pos_months", "months", "profit_factor")}


def _eval(p):
    cfg, datas, funding, t0, t1, t2 = _G["cfg"], _G["datas"], _G["funding"], _G["t0"], _G["t1"], _G["t2"]
    tf = core.tf_ms(cfg)
    T, O, H, L, C, FR = core.load_panel(datas, funding, cfg)
    W = strat.compute(T, C, FR, p, tf, detail=True)
    out = {"params": p}
    for name, a, b in (("train", t0, t1), ("test", t1, t2)):
        eng = core.run_backtest(datas, p, cfg, start_ms=a, funding=funding, end_ms=b - tf, instr=_G["instr"],
                                weights=_cut(W, T, b - tf))
        st = core.compute_stats(eng, cfg, a, eng.end_ts)
        out[name] = _summ(st)
    out["score"] = _score(out["train"])
    return out


def _cut(W, T, end):
    n = int(np.searchsorted(T, end, side="right"))
    return W[0][:n], {k: v[:n] for k, v in W[1].items()}


def _init(g):
    _G.update(g)


def sample_params(space, rng, base):
    p = dict(base)
    for k, vals in space.items():
        if vals:
            p[k] = rng.choice(vals)
    return core.clean_params(p)


def optimize(cfg, days=None, combos=None, train_frac=None, progress=None, download=True, seed=None):
    o = cfg["optimizer"]
    days = float(days or o["days"])
    combos = int(combos or o["combos"])
    train_frac = float(train_frac or o["train_frac"])
    tf = core.tf_ms(cfg)
    end = core.now_ms() // tf * tf
    start = int(end - days * core.DAY) // tf * tf
    split = int(start + (end - start) * train_frac) // tf * tf
    rng = random.Random(seed if seed is not None else int(time.time()))
    base = core.clean_params(cfg["strategy"])
    seen, cands = set(), []
    for p in [base] + [sample_params(o["space"], rng, base) for _ in range(combos * 3)]:
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key)
            cands.append(p)
        if len(cands) >= combos:
            break
    big = dict(base, lt=[max(sum(o["space"].get("lt", [[60]]), []) + base["lt"])])
    datas, funding = load_data(cfg, start, end, [big, base], progress, download)
    g = {"cfg": cfg, "datas": datas, "funding": funding if cfg["fees"].get("include_funding", True) else {},
         "t0": start, "t1": split, "t2": end, "instr": core.instruments(cfg)}
    workers = int(o.get("workers") or 0) or max(1, (os.cpu_count() or 2) - 1)
    res, t_start = [], time.time()
    if workers > 1:
        with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(g,)) as pool:
            for k, r in enumerate(pool.imap_unordered(_eval, cands)):
                res.append(r)
                if progress:
                    progress(0.6 + 0.4 * (k + 1) / len(cands), f"{k + 1}/{len(cands)} combos")
    else:
        _init(g)
        for k, p in enumerate(cands):
            res.append(_eval(p))
            if progress:
                progress(0.6 + 0.4 * (k + 1) / len(cands), f"{k + 1}/{len(cands)} combos")
    res.sort(key=lambda r: r["score"], reverse=True)
    for k, r in enumerate(res):
        r["rank"] = k + 1
        r["is_current"] = r["params"] == base
    top = res[: max(1, len(res) // 5)]
    summary = {"combos": len(res), "train": [start, split], "test": [split, end], "seconds": time.time() - t_start,
               "test_profitable_share": sum(1 for r in res if r["test"]["net_pct"] > 0) / len(res) * 100,
               "top_test_avg_month": float(np.mean([r["test"]["avg_month"] for r in top])),
               "top_test_sharpe": float(np.mean([r["test"]["sharpe"] for r in top])),
               "all_test_avg_month_median": float(np.median([r["test"]["avg_month"] for r in res]))}
    return {"summary": summary, "results": res}


def walkforward(cfg, days=2190, block_days=365, combos=20, progress=None, download=True, seed=1):
    o = cfg["optimizer"]
    tf = core.tf_ms(cfg)
    end = core.now_ms() // tf * tf
    start = int(end - days * core.DAY) // tf * tf
    rng = random.Random(seed)
    base = core.clean_params(cfg["strategy"])
    cands = [base] + [sample_params(o["space"], rng, base) for _ in range(combos - 1)]
    datas, funding = load_data(cfg, start, end, cands, progress, download)
    instr = core.instruments(cfg)
    T, O, H, L, C, FR = core.load_panel(datas, funding, cfg)
    Ws = [strat.compute(T, C, FR, p, tf, detail=True) for p in cands]
    out = []
    t = start + int(block_days * core.DAY)
    while t < end:
        b = min(end, t + int(block_days * core.DAY))
        best, best_s = 0, -1e18
        for i, p in enumerate(cands):
            eng = core.run_backtest(datas, p, cfg, start_ms=start, funding=funding, end_ms=t - tf, instr=instr,
                                    weights=_cut(Ws[i], T, t - tf))
            s = _score(core.compute_stats(eng, cfg, start, eng.end_ts))
            if s > best_s:
                best, best_s = i, s
        eng = core.run_backtest(datas, cands[best], cfg, start_ms=t, funding=funding, end_ms=b - tf, instr=instr,
                                weights=_cut(Ws[best], T, b - tf))
        st = core.compute_stats(eng, cfg, t, eng.end_ts)
        out.append({"from": t, "to": b, "params": cands[best], **_summ(st)})
        if progress:
            progress(min(1, (b - start) / (end - start)), f"block ending {core.fmt_ts(b, cfg, '%Y-%m-%d')}")
        t = b
    return out


def run_job(jid):
    jd = os.path.join(core.DATA_DIR, "jobs")
    req = core.read_json(os.path.join(jd, f"{jid}.req.json"), {}) or {}
    out_path = os.path.join(jd, f"{jid}.out.json")
    last = [0.0]

    def prog(f, m=""):
        if time.time() - last[0] > 0.7 or f >= 1:
            last[0] = time.time()
            core.save_json(out_path, {"state": "running", "progress": f, "msg": m})

    try:
        cfg = core.load_config()
        q = req.get("params", {})
        if req.get("kind") == "backtest":
            res = backtest(cfg, days=q["days"], balance=q["balance"], scale=q["scale"], kelly=q["kelly"],
                           band=q.get("band"), params=q.get("params"), progress=prog)
        elif req.get("kind") == "optimize":
            res = optimize(cfg, days=q["days"], combos=q["combos"], train_frac=q["train_frac"], progress=prog)
        else:
            raise RuntimeError("unknown job")
        core.save_json(out_path, {"state": "done", "progress": 1, "result": res})
    except Exception as e:
        log.error(f"job {jid} ({req.get('kind')}) failed: {e}", exc_info=True)
        core.save_json(out_path, {"state": "error", "progress": 1, "error": str(e)})


def _print_stats(st):
    keys = [("trades", "{:.0f}"), ("winrate", "{:.1f} %"), ("net_pct", "{:+.1f} %"), ("avg_month", "{:+.2f} %"),
            ("geo_month", "{:+.2f} %"), ("pos_months", "{:.0f}"), ("months", "{:.0f}"), ("sharpe", "{:.2f}"),
            ("max_dd_pct", "{:.1f} %"), ("max_daily_dd", "{:.1f} %"), ("profit_factor", "{:.2f}"), ("fees", "${:,.2f}"),
            ("funding", "${:,.2f}"), ("avg_lev", "{:.2f}x"), ("rebalances", "{:.0f}"), ("kelly_scale_full", "{:.2f}x")]
    for k, f in keys:
        v = st.get(k)
        print(f"  {k:18s} " + (f.format(v) if v is not None else "-"))
    print("  per coin:", ", ".join(f"{s.replace('USDT', '')} ${x['pnl']:+,.0f}" for s, x in sorted(st["per_sym"].items())))


def main():
    ap = argparse.ArgumentParser(description="ANTARCTICA backtester / optimizer")
    ap.add_argument("--job", help=argparse.SUPPRESS)
    ap.add_argument("--days", type=float, default=365)
    ap.add_argument("--balance", type=float)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--band", type=float)
    ap.add_argument("--kelly", action="store_true")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--combos", type=int)
    ap.add_argument("--offline", action="store_true", help="cached data only (no download)")
    a = ap.parse_args()
    if a.job:
        return run_job(a.job)
    cfg = core.load_config()
    pr = lambda f, m="": print(f"\r{f * 100:5.1f}% {m}          ", end="", flush=True)
    if a.optimize:
        r = optimize(cfg, days=a.days, combos=a.combos, download=not a.offline, progress=pr)
        print()
        s = r["summary"]
        print(f"{s['combos']} combos in {s['seconds']:.0f}s - test profitable for {s['test_profitable_share']:.0f}%; "
              f"best 20% by TRAIN -> TEST avg month {s['top_test_avg_month']:+.2f}%, Sharpe {s['top_test_sharpe']:.2f}")
        for x in r["results"][:15]:
            tr, te = x["train"], x["test"]
            print(f"#{x['rank']:3d}{'*' if x['is_current'] else ' '} train {tr['avg_month']:+6.2f}%/m sh {tr['sharpe']:4.2f} dd {tr['max_dd_pct']:4.1f}% | "
                  f"test {te['avg_month']:+6.2f}%/m sh {te['sharpe']:4.2f} dd {te['max_dd_pct']:4.1f}% | {x['params']}")
        return
    if a.walkforward:
        out = walkforward(cfg, days=max(a.days, 730), combos=a.combos or 20, download=not a.offline, progress=pr)
        print()
        for b in out:
            print(f"{core.fmt_ts(b['from'], cfg, '%Y-%m-%d')} -> {core.fmt_ts(b['to'], cfg, '%Y-%m-%d')}: avg month {b['avg_month']:+.2f}%, "
                  f"Sharpe {b['sharpe']:.2f}, maxDD {b['max_dd_pct']:.1f}%, win rate {b['winrate']:.1f}%")
        return
    b = backtest(cfg, days=a.days, balance=a.balance, scale=a.scale, kelly=True if a.kelly else None, band=a.band,
                 download=not a.offline, progress=pr)
    print()
    st = b["stats"]
    print(f"ANTARCTICA backtest {core.fmt_ts(b['start_ms'], cfg)} -> {core.fmt_ts(b['end_ms'], cfg)}  "
          f"${st['start_balance']:,.2f} -> ${st['equity']:,.2f}  (scale {b['request']['scale']}x)")
    _print_stats(st)


if __name__ == "__main__":
    main()
