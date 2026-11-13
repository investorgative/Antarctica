#!/usr/bin/env python3
"""
ANTARCTICA live paper trader (tmux session "Antarctica").

Every `poll_seconds` it pulls the latest 6h candles of all 12 coins from Bybit, and
  * while a candle is forming:   checks the protective stops against the forming high/low (paper fill at the stop)
  * when a candle closes:        waits until EVERY coin has that closed candle, then runs the same two-phase
                                 protocol as the backtester (phase 1 fills at the open for all coins, phase 2
                                 stops / funding / trailing / time stop / new signals for all coins)
After a crash or downtime it replays the missed closed candles in order, so the paper account stays identical
to what the backtester computes for the same period.

  python antarctica_trader.py            run
  python antarctica_trader.py --reset    wipe the paper account (start again with the configured balance)
"""
import argparse
import os
import signal
import sys
import time

import antarctica_core as core

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

    # ---------------------------------------------------------------- setup
    def _mtime(self):
        try:
            return os.path.getmtime(core.CONFIG_PATH)
        except Exception:
            return 0

    def _funding(self, sym, a, b):
        if not self.cfg["fees"].get("include_funding", True):
            return {}
        try:
            self.store.ensure_funding(sym, a - 8 * core.HOUR, b)
        except Exception as e:
            log.warning(f"{sym}: funding download failed: {e}")
        return self.store.load_funding(sym, a, b)

    def _new_engine(self):
        last_closed = (core.now_ms() // self.tf - 1) * self.tf
        eng = core.Engine(self.cfg["strategy"], self.cfg, funding=self._funding, trade_from=last_closed + self.tf)
        for s in self.syms:  # nothing to replay for a brand-new account
            eng.seen_ts[s] = last_closed
            eng.closed_ts[s] = last_closed
        return eng

    def load_state(self):
        st = core.read_json(core.STATE_PATH, None)
        self.eng = self._new_engine()
        if st and st.get("engine"):
            self.eng.load(st["engine"])
            self.started = st.get("started")
            log.info(f"state loaded: balance ${self.eng.balance:.2f}, {len(self.eng.trades)} trades, "
                     f"{len(self.eng.pos)} open positions")
        else:
            self.started = core.now_ms()
            log.info(f"new paper account: ${self.eng.balance:.2f}")
        self.apply_config(initial=True)

    def save_state(self):
        core.save_json(core.STATE_PATH, {"engine": self.eng.export(), "started": self.started, "version": core.VERSION,
                                         "symbols": self.syms, "timeframe": self.tfm, "saved": core.now_ms()})

    def apply_config(self, initial=False):
        """strategy / risk changes from the website take effect for new signals"""
        cfg = core.load_config()
        self.cfg = cfg
        e = self.eng
        a = cfg["account"]
        e.p = core.clean_params(cfg["strategy"])
        e.risk_pct = float(a["risk_pct"])
        e.heat_pct = float(a.get("heat_pct", 12))
        e.max_lev = float(a.get("max_leverage", 10))
        e.kelly = bool(a.get("kelly_sizing", False))
        e.kelly_frac = float(a.get("kelly_fraction", 0.5))
        e.kelly_max = float(a.get("kelly_max_risk_pct", 4.0))
        e.minqty_mult = float(a.get("minqty_risk_mult", 2.0))
        e.instr = core.instruments(cfg)
        e.taker, e.maker, e.slip = cfg["fees"]["taker"], cfg["fees"]["maker"], cfg["fees"].get("slippage", 0.0)
        if not initial:
            log.info(f"config reloaded: risk {e.risk_pct}% heat {e.heat_pct}% params {e.p}")

    def warmup(self):
        p = core.clean_params(self.cfg["strategy"])
        need = core.warmup_bars(p) + 60
        now = core.now_ms()
        for s in self.syms:
            for attempt in range(5):
                try:
                    self.store.ensure(s, self.tfm, now - need * self.tf, now)
                    break
                except Exception as e:
                    log.warning(f"{s}: history download failed ({e}), retry {attempt + 1}/5")
                    time.sleep(5 * (attempt + 1))
        core.refresh_instruments(self.cfg, log)
        self.eng.instr = core.instruments(self.cfg)

    # ---------------------------------------------------------------- per candle
    def _ind_at(self, sym, ts):
        p = self.eng.p
        n = core.warmup_bars(p) + 5
        d = self.store.tail(sym, self.tfm, n, end=ts)
        if not len(d["ts"]) or int(d["ts"][-1]) != ts:
            return None, None, None
        ind = core.indicators(d, p)
        i = len(d["ts"]) - 1
        bar = (int(d["ts"][i]), float(d["o"][i]), float(d["h"][i]), float(d["l"][i]), float(d["c"][i]), float(d["v"][i]))
        N = p["entry_n"]
        nxt = (float(d["h"][-N:].max()), float(d["l"][-N:].min())) if len(d["ts"]) >= N else None
        return bar, tuple(float(ind[k][i]) for k in core.IND_KEYS), nxt

    def process_closed(self, t):
        """two-phase protocol for closed candle time t (all symbols)"""
        bars = {}
        for s in self.syms:
            if t <= self.eng.closed_ts.get(s, 0):
                continue
            bar, ind, nxt = self._ind_at(s, t)
            if bar:
                bars[s] = (bar, ind, nxt)
        for s in self.syms:
            if s in bars:
                self.eng.open_bar(s, t, bars[s][0][1])
        for s in self.syms:
            if s in bars:
                self.eng.process(s, bars[s][0], bars[s][1], True)
                self.eng.last[s + "#next"] = bars[s][2]
        for s in self.syms:  # a symbol without this candle (exchange gap) must not block the others forever
            if s not in bars and t > self.eng.closed_ts.get(s, 0):
                self.eng.closed_ts[s] = t
                self.eng.seen_ts[s] = max(self.eng.seen_ts.get(s, 0), t)

    def poll(self):
        now = core.now_ms()
        cur = now // self.tf * self.tf  # open time of the forming candle
        target = cur - self.tf  # last closed candle
        got = {}
        for s in self.syms:
            try:
                rows = self.api.klines(s, self.tfm, limit=4, tries=2)
            except Exception as e:
                log.warning(f"{s}: kline poll failed: {e}")
                continue
            self.store.insert(s, self.tfm, rows, closed_before=now)
            got[s] = rows
            fr = [r for r in rows if r[0] == cur]
            if fr:
                self.forming[s] = fr[0]
        try:
            self.tickers = {k: v for k, v in self.api.tickers().items() if k in self.syms}
        except Exception as e:
            log.warning(f"ticker poll failed: {e}")
        # ---- closed candles not processed yet (replays downtime too)
        todo = set()
        mn_closed = min(self.eng.closed_ts.get(s, 0) for s in self.syms)
        if mn_closed < target:
            missing = [s for s in self.syms if self.store.bounds(s, self.tfm)[1] is None
                       or self.store.bounds(s, self.tfm)[1] < target]
            if missing:
                # catch up older history for a symbol (downtime) and wait a little for the newest candle
                for s in missing:
                    try:
                        self.store.ensure(s, self.tfm, self.eng.closed_ts.get(s, target - self.tf), now)
                    except Exception as e:
                        log.warning(f"{s}: catch-up download failed: {e}")
                missing = [s for s in self.syms if (self.store.bounds(s, self.tfm)[1] or 0) < target]
            first = self.wait_since.setdefault(target, now)
            if missing and now - first < 180_000:
                return  # wait until every coin has the closed candle (keeps the two-phase order exact)
            if missing:
                log.warning(f"candle {core.fmt_ts(target, self.cfg)} missing for {missing} after 3 min - continuing without")
            for s in self.syms:
                d = self.store.load(s, self.tfm, self.eng.closed_ts.get(s, 0) + 1, target)
                todo.update(int(x) for x in d["ts"])
            for t in sorted(todo):
                self.process_closed(t)
            self.wait_since = {}
        # ---- forming candle: fills at its open, then stops on the forming high/low
        for s in self.syms:
            fr = self.forming.get(s)
            if fr and fr[0] == cur:
                self.eng.open_bar(s, cur, fr[1])
        for s in self.syms:
            fr = self.forming.get(s)
            if fr and fr[0] == cur:
                self.eng.process(s, fr, None, False)
        self.last_poll = now

    # ---------------------------------------------------------------- outputs
    def flush_events(self):
        evs, self.eng.events = self.eng.events, []
        for ev in evs:
            core.push_event(ev)
            if ev["type"] in ("open", "trail", "close", "skip"):
                msg = core.fmt_discord(ev, self.cfg)
                url = self.cfg["discord"].get(ev.get("sym", ""), "")
                if msg and url:
                    core.discord_send(url, msg)
                if ev["type"] in ("open", "close"):
                    log.info(f"{ev['type'].upper()} {ev.get('sym')} " + (
                        f"{'LONG' if ev['pos']['dir'] == 1 else 'SHORT'} qty {ev['pos']['qty']:g} @ {core.fmt_px(ev['pos']['entry'])} stop {core.fmt_px(ev['pos']['stop'])}"
                        if ev["type"] == "open" else
                        f"{ev['trade']['reason']} @ {core.fmt_px(ev['trade']['exit'])} pnl ${ev['trade']['pnl']:+.2f} ({ev['trade']['r']:+.2f} R)"))
            if execmod:
                for kind, ex in self.executors.items():
                    try:
                        ex.on_event(ev)
                    except Exception as e:
                        log.error(f"{kind} executor: {e}")

    def write_status(self):
        e = self.eng
        sy = {}
        for s in self.syms:
            last = e.last.get(s)
            sy[s] = core.signal_status(s, e, last[0] if last else None, last[1] if last else None,
                                       self.forming.get(s), e.last.get(s + "#next"))
            sy[s]["forming"] = self.forming.get(s)
            t = self.tickers.get(s) or {}
            sy[s]["price"] = float(t.get("lastPrice") or (self.forming.get(s) or [0, 0, 0, 0, 0])[4] or 0)
            sy[s]["chg24"] = float(t.get("price24hPcnt") or 0) * 100
            sy[s]["funding_rate"] = float(t.get("fundingRate") or 0)
        execs = {}
        for kind, ex in self.executors.items():
            try:
                execs[kind] = ex.status()
            except Exception:
                pass
        core.save_json(core.STATUS_PATH, {
            "heartbeat": core.now_ms(), "last_poll": self.last_poll, "started": self.started, "version": core.VERSION,
            "timeframe": self.tfm, "next_close": (core.now_ms() // self.tf + 1) * self.tf, "symbols": sy,
            "balance": e.balance, "equity": e.equity({s: v["price"] for s, v in sy.items() if v.get("price")}),
            "open_heat": e.open_heat(), "heat_pct": e.heat_pct, "risk_pct": e.current_risk_pct(),
            "params": e.p, "pending": e.pending, "executors": execs, "pid": os.getpid()})

    def setup_executors(self):
        if not execmod:
            return
        for kind in ("demo", "live"):
            try:
                self.executors[kind] = execmod.Executor(kind, self)
            except Exception as ex:
                log.error(f"{kind} executor not available: {ex}")

    # ---------------------------------------------------------------- main loop
    def run(self):
        signal.signal(signal.SIGTERM, lambda *a: setattr(self, "stop", True))
        log.info(f"ANTARCTICA paper trader v{core.VERSION} starting - {len(self.syms)} coins, {self.tfm // 60}h candles")
        self.load_state()
        self.warmup()
        self.setup_executors()
        if self.cfg["discord"].get("notify_start", True):
            url = self.cfg["discord"].get("errors", "")
            core.discord_send(url, f"🧊 **ANTARCTICA** paper trader started · balance ${self.eng.balance:,.2f} · "
                                   f"{len(self.eng.pos)} open positions")
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
                    except Exception as e:
                        log.error(f"executor tick: {e}")
                self.poll_errors = 0
            except Exception as e:
                self.poll_errors += 1
                log.error(f"poll failed ({self.poll_errors}x): {e}", exc_info=self.poll_errors <= 2)
            try:
                self.write_status()
            except Exception as e:
                log.error(f"status write failed: {e}")
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
