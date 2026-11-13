#!/usr/bin/env python3
"""
ANTARCTICA live paper trader (tmux session "Antarctica").

Every `poll_seconds` it pulls the latest 4h candles of all 12 coins from Bybit. When a candle has closed for
EVERY coin it runs exactly the backtester's protocol:
  close of candle t : funding of candle t, mark-to-market, target weights from candles/funding up to t
  open of candle t+1: re-balance the paper book to those weights at the open price (taker fee + slippage)
After downtime it replays the missed candles in order, so the paper account stays identical to a backtest of
the same period.

  python antarctica_trader.py            run
  python antarctica_trader.py --reset    wipe the paper account (start again with the configured balance)
"""
import argparse
import os
import signal
import time

import numpy as np

import antarctica_core as core
import antarctica_strategy as strat

log = core.setup_logging("trader")

try:
    import antarctica_exec as execmod
except Exception as _e:  # pragma: no cover
    execmod = None
    log.error(f"executor module not loaded: {_e}")


class Trader:
    def __init__(self):
        self.cfg = core.load_config()
        self.cfg_mtime = self._mtime()
        self.api = core.Bybit(self.cfg)
        self.store = core.Store(self.api)
        self.tf = core.tf_ms(self.cfg)
        self.tfm = int(self.cfg["timeframe"])
        self.syms = list(self.cfg["symbols"])
        self.eng = None
        self.started = None
        self.forming = {}
        self.tickers = {}
        self.last_poll = 0
        self.poll_errors = 0
        self.stop = False
        self.wait_since = {}
        self.executors = {}

    def _mtime(self):
        try:
            return os.path.getmtime(core.CONFIG_PATH)
        except Exception:
            return 0

    def _funding_src(self, sym, a, b):
        if not self.cfg["fees"].get("include_funding", True):
            return {}
        return self.store.load_funding(sym, a, b)

    def _new_engine(self):
        last_closed = (core.now_ms() // self.tf - 1) * self.tf
        eng = core.Engine(self.cfg["strategy"], self.cfg, funding=self._funding_src, trade_from=last_closed + self.tf)
        eng.last_close_ts = last_closed - self.tf  # the last closed candle sets the first targets
        eng.last_open_ts = last_closed
        return eng

    def load_state(self):
        st = core.read_json(core.STATE_PATH, None)
        self.eng = self._new_engine()
        if st and st.get("engine") and st.get("kind") == "ensemble":
            self.eng.load(st["engine"])
            self.started = st.get("started")
            log.info(f"state loaded: equity ${self.eng.equity():,.2f}, {len(self.eng.trades)} closed trades, "
                     f"{len(self.eng.pos)} open positions")
        else:
            if st:
                log.warning("old state file (different strategy) ignored - starting a new paper account")
            self.started = core.now_ms()
            log.info(f"new paper account: ${self.eng.cash:,.2f}")
        self.eng.funding = self._funding_src
        self.apply_config(initial=True)

    def save_state(self):
        e = self.eng
        if len(e.curve) > 6000:  # keep the live curve compact: thin old points to one per day
            old, new = e.curve[:-1500], e.curve[-1500:]
            keep, seen = [], set()
            for c in old:
                d = c[0] // core.DAY
                if d not in seen:
                    seen.add(d)
                    keep.append(c)
            e.curve = keep + new
        core.save_json(core.STATE_PATH, {"engine": e.export(), "started": self.started, "version": core.VERSION,
                                         "kind": "ensemble", "symbols": self.syms, "timeframe": self.tfm,
                                         "saved": core.now_ms()})

    def apply_config(self, initial=False):
        cfg = core.load_config()
        self.cfg = cfg
        e = self.eng
        a = cfg["account"]
        e.p = core.clean_params(cfg["strategy"])
        e.max_lev = float(a.get("max_leverage", 8))
        e.band = float(a.get("band_pct", 2)) / 100
        e.kelly = bool(a.get("kelly_sizing", False))
        e.kelly_frac = float(a.get("kelly_fraction", 0.5))
        e.kelly_max = float(a.get("kelly_max_scale", 6))
        e.instr = core.instruments(cfg)
        e.taker, e.maker, e.slip = cfg["fees"]["taker"], cfg["fees"]["maker"], cfg["fees"].get("slippage", 0.0)
        if not initial:
            log.info(f"config reloaded: scale {e.p['scale']}x band {e.band * 100:.1f}% params {e.p}")

    def warmup(self):
        need = strat.warmup_bars(self.cfg["strategy"]) + 80
        now = core.now_ms()
        for s in self.syms:
            for attempt in range(5):
                try:
                    self.store.ensure(s, self.tfm, now - need * self.tf, now)
                    self.store.ensure_funding(s, now - need * self.tf, now)
                    break
                except Exception as ex:
                    log.warning(f"{s}: history download failed ({ex}), retry {attempt + 1}/5")
                    time.sleep(5 * (attempt + 1))
        core.refresh_instruments(self.cfg, log)
        self.eng.instr = core.instruments(self.cfg)

    # ---------------------------------------------------------------- candle processing
    def _weights_for(self, ts_list):
        """target weights (and sleeves) for the closed candles in ts_list, from the stored history"""
        need = strat.warmup_bars(self.eng.p) + 20
        lo = ts_list[0] - need * self.tf
        datas, funding = {}, {}
        for s in self.syms:
            d = self.store.load(s, self.tfm, lo, ts_list[-1])
            if len(d["ts"]):
                datas[s] = d
            funding[s] = self.store.load_funding(s, lo, ts_list[-1] + self.tf - 1)
        T, O, H, L, C, FR = core.load_panel(datas, funding, self.cfg)
        W, SL = strat.compute(T, C, FR, self.eng.p, self.tf, detail=True)
        out = {}
        for t in ts_list:
            i = int(np.searchsorted(T, t))
            if i < len(T) and T[i] == t:
                out[t] = ({s: float(W[i, j]) for j, s in enumerate(self.syms) if C[i, j] == C[i, j]},
                          {s: float(O[i, j]) for j, s in enumerate(self.syms) if O[i, j] == O[i, j]},
                          {s: float(C[i, j]) for j, s in enumerate(self.syms) if C[i, j] == C[i, j]},
                          {k: {s: float(v[i, j]) for j, s in enumerate(self.syms)} for k, v in SL.items()})
        return out

    def poll(self):
        now = core.now_ms()
        cur = now // self.tf * self.tf
        target = cur - self.tf
        for s in self.syms:
            try:
                rows = self.api.klines(s, self.tfm, limit=4, tries=2)
            except Exception as ex:
                log.warning(f"{s}: kline poll failed: {ex}")
                continue
            self.store.insert(s, self.tfm, rows, closed_before=now)
            fr = [r for r in rows if r[0] == cur]
            if fr:
                self.forming[s] = fr[0]
        try:
            self.tickers = {k: v for k, v in self.api.tickers().items() if k in self.syms}
        except Exception as ex:
            log.warning(f"ticker poll failed: {ex}")
        e = self.eng
        if target - e.last_close_ts > 2 * self.tf:  # downtime: fill the whole gap, not only the newest candles
            for s in self.syms:
                try:
                    self.store.insert(s, self.tfm, self.api.klines_range(s, self.tfm, e.last_close_ts - self.tf, now),
                                      closed_before=now)
                except Exception as ex:
                    log.warning(f"{s}: gap fill failed: {ex}")
        if e.last_close_ts < target:
            missing = [s for s in self.syms if (self.store.bounds(s, self.tfm)[1] or 0) < target]
            if missing:
                for s in missing:
                    try:
                        self.store.ensure(s, self.tfm, e.last_close_ts - self.tf, now)
                    except Exception as ex:
                        log.warning(f"{s}: catch-up download failed: {ex}")
                missing = [s for s in self.syms if (self.store.bounds(s, self.tfm)[1] or 0) < target]
            first = self.wait_since.setdefault(target, now)
            if missing and now - first < 180_000:
                return  # wait until every coin has the closed candle
            if missing:
                log.warning(f"candle {core.fmt_ts(target, self.cfg)} missing for {missing} after 3 min - continuing without")
            # funding of the candles to process must be known first
            for s in self.syms:
                try:
                    self.store.ensure_funding(s, e.last_close_ts, now)
                except Exception as ex:
                    log.warning(f"{s}: funding download failed: {ex}")
            ts_list = list(range(e.last_close_ts + self.tf, target + 1, self.tf))
            wmap = self._weights_for(ts_list)
            for t in ts_list:
                if t not in wmap:
                    e.last_close_ts = t
                    continue
                tg, opens, closes, sl = wmap[t]
                e.on_open(t, opens)
                e.on_close(t, opens, closes, tg, sl)
            self.wait_since = {}
        # open of the forming candle: re-balance at its open price
        opens = {s: fr[1] for s, fr in self.forming.items() if fr[0] == cur}
        need = set(e.targets) | set(e.pos)
        if e.last_close_ts == target and e.last_open_ts < cur and opens:
            lacking = sorted(need - set(opens))
            first = self.wait_since.setdefault(("open", cur), now)
            if not lacking or now - first > 180_000:
                if lacking:
                    log.warning(f"open prices missing for {lacking} after 3 min - re-balancing without them")
                e.on_open(cur, opens)
        for s, fr in self.forming.items():
            if fr[0] == cur:
                e.last_px[s] = fr[4]
        self.last_poll = now

    # ---------------------------------------------------------------- outputs
    def flush_events(self):
        evs, self.eng.events = self.eng.events, []
        for ev in evs:
            core.push_event(ev)
            if ev["type"] in ("open", "adjust", "close"):
                msg = core.fmt_discord(ev, self.cfg)
                url = self.cfg["discord"].get(ev.get("sym", ""), "")
                if msg and url:
                    core.discord_send(url, msg)
            if ev["type"] == "rebalance":
                log.info(f"rebalanced {len(ev['syms'])} coins, equity ${ev['equity']:,.2f}")
                if self.cfg["discord"].get("rebalance_summary", True):
                    core.discord_send(self.cfg["discord"].get("errors", ""), core.fmt_rebalance(ev, self.cfg))
            if ev["type"] == "close":
                t = ev["trade"]
                log.info(f"CLOSE {ev['sym']} {'LONG' if t['dir'] > 0 else 'SHORT'} pnl ${t['pnl']:+.2f} ({t['r']:+.2f} R)")
            if execmod:
                for kind, ex in self.executors.items():
                    try:
                        ex.on_event(ev)
                    except Exception as err:
                        log.error(f"{kind} executor: {err}")

    def write_status(self):
        e = self.eng
        sy = {}
        prices = {}
        for s in self.syms:
            t = self.tickers.get(s) or {}
            px = float(t.get("lastPrice") or (self.forming.get(s) or [0, 0, 0, 0, 0])[4] or 0)
            prices[s] = px
            sy[s] = {"price": px, "chg24": float(t.get("price24hPcnt") or 0) * 100,
                     "funding_rate": float(t.get("fundingRate") or 0), "forming": self.forming.get(s),
                     "target": e.targets.get(s, 0.0),
                     "sleeves": {k: v.get(s, 0.0) for k, v in (e.sleeves or {}).items()}}
        eq = e.equity({k: v for k, v in prices.items() if v})
        for s in self.syms:
            ps = e.pos.get(s)
            sy[s]["weight"] = ps["qty"] * prices[s] / eq if ps and eq > 0 and prices[s] else 0.0
        execs = {}
        for kind, ex in self.executors.items():
            try:
                execs[kind] = ex.status()
            except Exception:
                pass
        core.save_json(core.STATUS_PATH, {
            "heartbeat": core.now_ms(), "last_poll": self.last_poll, "started": self.started, "version": core.VERSION,
            "timeframe": self.tfm, "next_close": (core.now_ms() // self.tf + 1) * self.tf, "symbols": sy,
            "cash": e.cash, "equity": eq, "gross": e.gross({k: v for k, v in prices.items() if v}) / eq if eq > 0 else 0,
            "scale": e.cur_scale, "kelly_scale": e.kelly_scale(), "params": e.p, "target_ts": e.target_ts,
            "last_close_ts": e.last_close_ts, "executors": execs, "pid": os.getpid()})

    def setup_executors(self):
        if not execmod:
            return
        for kind in ("demo", "live"):
            try:
                self.executors[kind] = execmod.Executor(kind, self)
            except Exception as ex:
                log.error(f"{kind} executor not available: {ex}")

    def run(self):
        signal.signal(signal.SIGTERM, lambda *a: setattr(self, "stop", True))
        log.info(f"ANTARCTICA paper trader v{core.VERSION} starting - {len(self.syms)} coins, {self.tfm // 60}h candles")
        self.load_state()
        self.warmup()
        self.setup_executors()
        if self.cfg["discord"].get("notify_start", True):
            core.discord_send(self.cfg["discord"].get("errors", ""),
                              f"🧊 **ANTARCTICA** paper trader started · equity ${self.eng.equity():,.2f} · "
                              f"{len(self.eng.pos)} open positions · scale {self.eng.p['scale']}x")
        while not self.stop:
            t0 = time.time()
            try:
                m = self._mtime()
                if m != self.cfg_mtime:
                    self.cfg_mtime = m
                    self.apply_config()
                self.poll()
                self.flush_events()
                self.save_state()
                for ex in self.executors.values():
                    try:
                        ex.tick()
                    except Exception as err:
                        log.error(f"executor tick: {err}")
                self.poll_errors = 0
            except Exception as ex:
                self.poll_errors += 1
                log.error(f"poll failed ({self.poll_errors}x): {ex}", exc_info=self.poll_errors <= 2)
            try:
                self.write_status()
            except Exception as ex:
                log.error(f"status write failed: {ex}")
            time.sleep(max(1.0, float(self.cfg.get("poll_seconds", 15)) - (time.time() - t0)))
        self.save_state()
        log.info("trader stopped")


def main():
    ap = argparse.ArgumentParser(description="ANTARCTICA paper trader")
    ap.add_argument("--reset", action="store_true", help="wipe the paper account")
    a = ap.parse_args()
    if a.reset:
        for pth in (core.STATE_PATH, core.EVENTS_PATH, core.STATUS_PATH):
            if os.path.exists(pth):
                os.remove(pth)
        print("paper account wiped - it starts again with the configured balance on the next start")
        return
    Trader().run()


if __name__ == "__main__":
    main()
