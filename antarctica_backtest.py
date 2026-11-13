#!/usr/bin/env python3
"""
ANTARCTICA backtester + optimizer (command line and website).

  backtest   the paper trader's own Engine, candle by candle, all 12 coins sharing one balance,
             Bybit taker/maker fees on every fill, slippage, real funding, lot sizes, leverage cap, heat cap.
  optimizer  random search over the strategy parameters with a train / test split in time:
             parameters are ranked on the TRAIN part only, the TEST part shows how they did on data the
             ranking never saw. A walk-forward run (choose on the past, trade the next block) is included.

NO LOOK-AHEAD: indicators at candle i only use candles <= i, signals are decided at the close and filled at
the next open, and every candle is processed with the same two-phase protocol as the live trader.

  python antarctica_backtest.py --days 365 --risk 2 --balance 100
  python antarctica_backtest.py --days 1095 --kelly
  python antarctica_backtest.py --optimize --days 1095 --combos 150
  python antarctica_backtest.py --walkforward --days 1825
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

import antarctica_core as core

log = core.setup_logging("backtest")


def load_data(cfg, start_ms, end_ms, params_list, progress=None, download=True, symbols=None):
    """candles (with warm-up before start_ms) and funding for every symbol"""
    tf = int(cfg["timeframe"])
    syms = symbols or cfg["symbols"]
    download = download and not os.environ.get("ANTARCTICA_OFFLINE")
    warm = max(core.warmup_bars(core.clean_params(p)) for p in params_list) + 2
    lead = start_ms - warm * tf * core.MIN
    store = core.Store(core.Bybit(cfg) if download else None)
    datas, funding = {}, {}
    for k, s in enumerate(syms):
        if progress:
            progress(k / len(syms) * 0.9, f"candles {s}")
        if download:
            try:
                store.ensure(s, tf, lead, end_ms)
                if cfg["fees"].get("include_funding", True):
                    store.ensure_funding(s, lead, end_ms)
            except Exception as e:
                log.warning(f"{s}: download failed ({e}) - using cached candles only")
        d = store.load(s, tf, lead, end_ms - tf * core.MIN)
        if len(d["ts"]):
            datas[s] = d
            funding[s] = store.load_funding(s, lead, end_ms)
    if download:
        core.refresh_instruments(cfg, log)
    return datas, funding


def backtest(cfg, days=365, risk=None, balance=None, heat=None, kelly=None, params=None, end_ms=None, progress=None,
             download=True, datas=None, funding=None):
    cfg = json.loads(json.dumps(cfg))
    if risk is not None:
        cfg["account"]["risk_pct"] = float(risk)
    if balance is not None:
        cfg["account"]["start_balance"] = float(balance)
    if heat is not None:
        cfg["account"]["heat_pct"] = float(heat)
    if kelly is not None:
        cfg["account"]["kelly_sizing"] = bool(kelly)
    p = core.clean_params(params or cfg["strategy"])
    tf = core.tf_ms(cfg)
    end_ms = int(end_ms or (core.now_ms() // tf * tf))
    start_ms = int(end_ms - float(days) * core.DAY) // tf * tf
    if datas is None:
        datas, funding = load_data(cfg, start_ms, end_ms, [p], progress, download)
    if progress:
        progress(0.92, "simulating")
    if not datas:
        raise RuntimeError("no candles available - is the Bybit API reachable?")
    eng = core.run_backtest(datas, p, cfg, start_ms=start_ms, funding=funding if cfg["fees"].get("include_funding", True) else {},
                            end_ms=end_ms - tf, instr=core.instruments(cfg))
    b = core.bundle(eng, cfg, start_ms, eng.end_ts, prices={s: float(d["c"][-1]) for s, d in datas.items()})
    b["kind"] = "backtest"
    b["request"] = {"days": days, "risk": cfg["account"]["risk_pct"], "balance": cfg["account"]["start_balance"],
                    "heat": cfg["account"]["heat_pct"], "kelly": cfg["account"]["kelly_sizing"]}
    b["data_from"] = {s: int(d["ts"][0]) for s, d in datas.items()}
    if progress:
        progress(1.0, "done")
    return b


# ---------------------------------------------------------------- optimizer
_G = {}


def _score(st, min_trades):
    """portfolio return / drawdown ratio (MAR-like), penalised below min_trades"""
    if st["trades"] < max(5, min_trades):
        return -1e9
    return st["avg_month"] / max(5.0, st["max_dd_pct"]) * 10 + st["winrate"] / 100.0


def _eval(p):
    cfg, datas, funding, t0, t1, t2 = _G["cfg"], _G["datas"], _G["funding"], _G["t0"], _G["t1"], _G["t2"]
    out = {"params": p}
    for name, a, b in (("train", t0, t1), ("test", t1, t2)):
        eng = core.run_backtest(datas, p, cfg, start_ms=a, funding=funding, end_ms=b - core.tf_ms(cfg),
                                instr=_G["instr"], caches=_G["caches"])
        st = core.compute_stats(eng.trades, cfg, eng.start_balance, a, eng.end_ts, days_eq=eng.days)
        out[name] = {k: st[k] for k in ("trades", "winrate", "cum_r", "avg_month", "geo_month", "max_dd_pct", "max_dd_r",
                                        "profit_factor", "net_pct", "pos_months", "months", "avg_r")}
    out["score"] = _score(out["train"], _G["min_trades"] * (t1 - t0) / max(1, t2 - t0))
    return out


def _init(g):
    _G.update(g)


def sample_params(space, rng, base):
    p = dict(base)
    for k, vals in space.items():
        if k in p and vals:
            p[k] = rng.choice(vals)
    return core.clean_params(p)


def optimize(cfg, days=None, combos=None, train_frac=None, progress=None, download=True, seed=None, datas=None, funding=None):
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
    cands = [base] + [sample_params(o["space"], rng, base) for _ in range(combos - 1)]
    big = dict(base, ema_n=max(o["space"].get("ema_n", [base["ema_n"]]) + [base["ema_n"]]),
               entry_n=max(o["space"].get("entry_n", [base["entry_n"]]) + [base["entry_n"]]))
    if datas is None:
        datas, funding = load_data(cfg, start, end, [big], progress, download)
    g = {"cfg": cfg, "datas": datas, "funding": funding if cfg["fees"].get("include_funding", True) else {},
         "t0": start, "t1": split, "t2": end, "instr": core.instruments(cfg), "caches": {},
         "min_trades": int(o.get("min_trades", 60))}
    workers = int(o.get("workers") or 0) or max(1, (os.cpu_count() or 2) - 1)
    res = []
    t_start = time.time()
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init, initargs=(g,)) as pool:
            for k, r in enumerate(pool.imap_unordered(_eval, cands)):
                res.append(r)
                if progress:
                    progress(0.1 + 0.9 * (k + 1) / len(cands), f"{k + 1}/{len(cands)} combos")
    else:
        _init(g)
        for k, p in enumerate(cands):
            res.append(_eval(p))
            if progress:
                progress(0.1 + 0.9 * (k + 1) / len(cands), f"{k + 1}/{len(cands)} combos")
    res.sort(key=lambda r: r["score"], reverse=True)
    for k, r in enumerate(res):
        r["rank"] = k + 1
        r["is_current"] = r["params"] == base
    prof_test = sum(1 for r in res if r["test"]["cum_r"] > 0)
    top = res[: max(1, len(res) // 10)]
    summary = {"combos": len(res), "train": [start, split], "test": [split, end], "seconds": time.time() - t_start,
               "test_profitable_share": prof_test / len(res) * 100,
               "top10pct_test_avg_month": float(np.mean([r["test"]["avg_month"] for r in top])),
               "top10pct_test_winrate": float(np.mean([r["test"]["winrate"] for r in top])),
               "all_test_avg_month_median": float(np.median([r["test"]["avg_month"] for r in res]))}
    return {"summary": summary, "results": res}


def walkforward(cfg, days=1825, block_days=365, combos=60, progress=None, download=True, seed=1):
    """anchored walk-forward: for every block, choose the best parameters on everything before it
    (optimizer score), then trade the block with them. Only the traded blocks are reported."""
    o = cfg["optimizer"]
    tf = core.tf_ms(cfg)
    end = core.now_ms() // tf * tf
    start = int(end - days * core.DAY) // tf * tf
    rng = random.Random(seed)
    base = core.clean_params(cfg["strategy"])
    cands = [base] + [sample_params(o["space"], rng, base) for _ in range(combos - 1)]
    big = dict(base, ema_n=max(o["space"]["ema_n"] + [base["ema_n"]]), entry_n=max(o["space"]["entry_n"] + [base["entry_n"]]))
    datas, funding = load_data(cfg, start, end, [big], progress, download)
    instr = core.instruments(cfg)
    blocks = []
    t = start + int(block_days * core.DAY)
    while t < end:
        blocks.append((t, min(end, t + int(block_days * core.DAY))))
        t += int(block_days * core.DAY)
    out = []
    for bi, (a, b) in enumerate(blocks):
        best, best_s = None, -1e18
        for p in cands:
            eng = core.run_backtest(datas, p, cfg, start_ms=start, funding=funding, end_ms=a - tf, instr=instr)
            st = core.compute_stats(eng.trades, cfg, eng.start_balance, start, eng.end_ts, days_eq=eng.days)
            s = _score(st, 30)
            if s > best_s:
                best, best_s = p, s
        eng = core.run_backtest(datas, best, cfg, start_ms=a, funding=funding, end_ms=b - tf, instr=instr)
        st = core.compute_stats(eng.trades, cfg, eng.start_balance, a, eng.end_ts, days_eq=eng.days)
        out.append({"from": a, "to": b, "params": best, "trades": st["trades"], "winrate": st["winrate"],
                    "cum_r": st["cum_r"], "avg_month": st["avg_month"], "max_dd_pct": st["max_dd_pct"]})
        if progress:
            progress((bi + 1) / len(blocks), f"block {bi + 1}/{len(blocks)}")
    return out


def _print_stats(st):
    keys = [("trades", "{:.0f}"), ("winrate", "{:.1f} %"), ("cum_r", "{:+.1f} R"), ("avg_r", "{:+.3f} R"),
            ("net_pct", "{:+.1f} %"), ("avg_month", "{:+.2f} %"), ("geo_month", "{:+.2f} %"), ("pos_months", "{:.0f}"),
            ("months", "{:.0f}"), ("max_dd_pct", "{:.1f} %"), ("max_dd_r", "{:.1f} R"), ("profit_factor", "{:.2f}"),
            ("fees", "${:.2f}"), ("funding", "${:.2f}"), ("avg_lev", "{:.2f}x"), ("avg_hold_h", "{:.1f} h"),
            ("max_consec_losses", "{:.0f}"), ("kelly_full", "{:.2f} %"), ("kelly_half", "{:.2f} %")]
    for k, f in keys:
        v = st.get(k)
        print(f"  {k:18s} " + (f.format(v) if v is not None else "-"))
    print("  per coin:", ", ".join(f"{s.replace('USDT', '')} {x['r']:+.1f}R/{x['n']}" for s, x in sorted(st["per_sym"].items())))


def run_job(jid):
    """website job: data/jobs/<id>.req.json -> progress + result in data/jobs/<id>.out.json"""
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
            res = backtest(cfg, days=q["days"], risk=q["risk"], balance=q["balance"], heat=q["heat"], kelly=q["kelly"],
                           params=q.get("params"), progress=prog)
        elif req.get("kind") == "optimize":
            cfg["account"]["risk_pct"], cfg["account"]["heat_pct"] = q["risk"], q["heat"]
            cfg["optimizer"]["workers"] = cfg["optimizer"].get("workers") or 0
            res = optimize(cfg, days=q["days"], combos=q["combos"], train_frac=q["train_frac"], progress=prog)
        else:
            raise RuntimeError("unknown job")
        core.save_json(out_path, {"state": "done", "progress": 1, "result": res})
    except Exception as e:
        log.error(f"job {jid} ({req.get('kind')}) failed: {e}", exc_info=True)
        core.save_json(out_path, {"state": "error", "progress": 1, "error": str(e)})


def main():
    ap = argparse.ArgumentParser(description="ANTARCTICA backtester / optimizer")
    ap.add_argument("--job", help=argparse.SUPPRESS)
    ap.add_argument("--days", type=float, default=365)
    ap.add_argument("--risk", type=float)
    ap.add_argument("--balance", type=float)
    ap.add_argument("--heat", type=float)
    ap.add_argument("--kelly", action="store_true")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--combos", type=int)
    ap.add_argument("--offline", action="store_true", help="use cached candles only (no download)")
    a = ap.parse_args()
    if a.job:
        return run_job(a.job)
    cfg = core.load_config()
    if a.optimize:
        r = optimize(cfg, days=a.days, combos=a.combos, download=not a.offline,
                     progress=lambda f, m="": print(f"\r{f * 100:5.1f}% {m}   ", end="", flush=True))
        print()
        s = r["summary"]
        print(f"{s['combos']} combos in {s['seconds']:.0f}s - test part profitable for {s['test_profitable_share']:.0f}% of them")
        print(f"best 10% by TRAIN score -> TEST avg month {s['top10pct_test_avg_month']:+.2f}%, win rate {s['top10pct_test_winrate']:.1f}%")
        for x in r["results"][:15]:
            tr, te = x["train"], x["test"]
            print(f"#{x['rank']:3d} {'*' if x['is_current'] else ' '} train {tr['avg_month']:+6.2f}%/m wr {tr['winrate']:4.1f} dd {tr['max_dd_pct']:4.1f}% | "
                  f"test {te['avg_month']:+6.2f}%/m wr {te['winrate']:4.1f} dd {te['max_dd_pct']:4.1f}% | {x['params']}")
        return
    if a.walkforward:
        out = walkforward(cfg, days=a.days if a.days > 365 else 1825, combos=a.combos or 40, download=not a.offline,
                          progress=lambda f, m="": print(f"\r{f * 100:5.1f}% {m}   ", end="", flush=True))
        print()
        for b in out:
            print(f"{core.fmt_ts(b['from'], cfg, '%Y-%m-%d')} -> {core.fmt_ts(b['to'], cfg, '%Y-%m-%d')}: {b['trades']} trades, "
                  f"wr {b['winrate']:.1f}%, {b['cum_r']:+.1f} R, avg month {b['avg_month']:+.2f}%, maxDD {b['max_dd_pct']:.1f}%")
        return
    b = backtest(cfg, days=a.days, risk=a.risk, balance=a.balance, heat=a.heat, kelly=True if a.kelly else None,
                 download=not a.offline, progress=lambda f, m="": print(f"\r{f * 100:5.1f}% {m}   ", end="", flush=True))
    print()
    st = b["stats"]
    print(f"ANTARCTICA backtest {core.fmt_ts(b['start_ms'], cfg)} -> {core.fmt_ts(b['end_ms'], cfg)}  "
          f"${st['start_balance']:.2f} -> ${st['equity']:.2f}  (risk {b['request']['risk']}%, heat {b['request']['heat']}%)")
    _print_stats(st)


if __name__ == "__main__":
    main()
