#!/usr/bin/env python3
"""
ANTARCTICA core - shared by the paper trader, the backtester, the optimizer and the website.

Strategy "Aurora Breakout" - a 12-coin Bybit USDT-perpetual portfolio on 6h candles
  * ENTRY   a 6h candle CLOSES above the highest high (below the lowest low) of the previous `entry_n` candles,
            that candle is an impulse (high - low >= `impulse` x ATR) and it agrees with the trend
            (close above / below the EMA(`ema_n`)).  -> market order at the OPEN of the next candle.
  * STOP    entry -/+ `stop_atr` x ATR. Position size = risk$ / stop distance (Bybit lot size, leverage cap).
  * EXIT    chandelier trail: best price since entry -/+ `trail_atr` x ATR (moved only at candle closes),
            optional break-even at `be_r` R, and a time stop after `max_bars` candles (exit at the close).
  * PORTFOLIO  every coin shares one balance. New positions are only opened while the total open risk
            ("heat") stays below `heat_pct` % of the balance - the coins are strongly correlated.

NO LOOK-AHEAD: every decision is taken on a closed candle and only uses that candle and older ones.
Orders created by a decision are filled on a LATER candle. The paper trader and the backtester run
the very same Engine class with the same two-phase candle protocol (phase 1: all fills at the open,
phase 2: stops / closes), so a paper trade and a backtest trade come from the same code.
"""
import json
import logging
import math
import os
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

import numpy as np
import requests

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

VERSION = "1.0.0"
NAME = "ANTARCTICA"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("ANTARCTICA_DATA") or os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
CONFIG_PATH = os.environ.get("ANTARCTICA_CONFIG") or os.path.join(BASE_DIR, "config.json")
DB_PATH = os.path.join(DATA_DIR, "antarctica.db")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
STATUS_PATH = os.path.join(DATA_DIR, "status.json")
EVENTS_PATH = os.path.join(DATA_DIR, "events.json")
SECRETS_PATH = os.path.join(DATA_DIR, "secrets.json")
INSTR_PATH = os.path.join(DATA_DIR, "instruments.json")
LOG_PATH = os.path.join(LOG_DIR, "antarctica.log")
os.makedirs(LOG_DIR, exist_ok=True)

MIN = 60_000
HOUR = 3_600_000
DAY = 86_400_000

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
           "ADAUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT", "DOTUSDT", "BCHUSDT"]

# Bybit linear-perpetual lot sizes (fallback only - the trader/backtester refresh them from
# /v5/market/instruments-info when the API is reachable).
INSTRUMENTS = {
    "BTCUSDT": {"qty_step": 0.001, "min_qty": 0.001, "tick": 0.1},
    "ETHUSDT": {"qty_step": 0.01, "min_qty": 0.01, "tick": 0.01},
    "SOLUSDT": {"qty_step": 0.1, "min_qty": 0.1, "tick": 0.01},
    "XRPUSDT": {"qty_step": 1, "min_qty": 1, "tick": 0.0001},
    "DOGEUSDT": {"qty_step": 1, "min_qty": 1, "tick": 0.00001},
    "BNBUSDT": {"qty_step": 0.01, "min_qty": 0.01, "tick": 0.01},
    "ADAUSDT": {"qty_step": 1, "min_qty": 1, "tick": 0.0001},
    "LINKUSDT": {"qty_step": 0.1, "min_qty": 0.1, "tick": 0.001},
    "LTCUSDT": {"qty_step": 0.1, "min_qty": 0.1, "tick": 0.01},
    "AVAXUSDT": {"qty_step": 0.1, "min_qty": 0.1, "tick": 0.001},
    "DOTUSDT": {"qty_step": 0.1, "min_qty": 0.1, "tick": 0.001},
    "BCHUSDT": {"qty_step": 0.01, "min_qty": 0.01, "tick": 0.05},
}

DEFAULT_CONFIG = {
    "symbols": list(SYMBOLS),
    "category": "linear",
    "api_base": "https://api.bybit.com",
    "timezone": "Europe/Berlin",
    "timeframe": 360,
    "poll_seconds": 15,
    "account": {
        "start_balance": 100.0,
        "risk_pct": 2.0,
        "heat_pct": 12.0,
        "max_leverage": 10,
        "lot_rounding": True,
        "min_notional": 5.0,
        "minqty_risk_mult": 2.0,
        "kelly_sizing": False,
        "kelly_fraction": 0.5,
        "kelly_min_trades": 40,
        "kelly_max_risk_pct": 4.0,
    },
    "fees": {
        "maker": 0.0002,
        "taker": 0.00055,
        "slippage": 0.0001,
        "include_funding": True,
    },
    "strategy": {
        "entry_n": 15,
        "impulse": 1.5,
        "ema_n": 200,
        "atr_n": 20,
        "stop_atr": 2.0,
        "trail_atr": 8.0,
        "max_bars": 30,
        "be_r": 0.0,
        "trend_filter": True,
        "allow_long": True,
        "allow_short": True,
    },
    "optimizer": {
        "days": 1095,
        "combos": 120,
        "train_frac": 0.6,
        "min_trades": 60,
        "workers": 0,
        "space": {
            "entry_n": [8, 10, 12, 15, 20, 25],
            "impulse": [0.0, 1.0, 1.25, 1.5, 2.0],
            "ema_n": [100, 150, 200, 300],
            "atr_n": [14, 20, 30],
            "stop_atr": [1.5, 2.0, 2.5, 3.0],
            "trail_atr": [5.0, 6.0, 8.0, 10.0, 12.0],
            "max_bars": [0, 20, 30, 40],
            "be_r": [0.0, 0.0, 1.0, 2.0],
        },
    },
    "discord": {"errors": "", "notify_start": True, **{s: "" for s in SYMBOLS}},
    "web": {"port": 8760, "passcode": "", "autolock_minutes": 10},
    "demo": {"risk_pct": 2.0, "max_leverage": 10},
    "live": {"risk_pct": 1.0, "max_leverage": 5},
}

STRATEGY_KEYS = ("entry_n", "impulse", "ema_n", "atr_n", "stop_atr", "trail_atr", "max_bars", "be_r",
                 "trend_filter", "allow_long", "allow_short")


# ---------------------------------------------------------------- small helpers
def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def clean_json(o):
    """numpy / nan safe JSON"""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    return o


def save_json(path, obj):
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean_json(obj), f, separators=(",", ":"))
    os.replace(tmp, path)


def read_json(path, default=None):
    for _ in range(3):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except Exception:
            time.sleep(0.05)
    return default


def _write_config(raw):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    os.replace(tmp, CONFIG_PATH)
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass


def load_config():
    """config.json (created with defaults on first run) merged over the defaults."""
    raw = read_json(CONFIG_PATH, None)
    if raw is None:
        raw = {k: DEFAULT_CONFIG[k] for k in ("symbols", "timezone", "account", "fees", "strategy", "discord", "web")}
        _write_config(raw)
    cfg = deep_merge(DEFAULT_CONFIG, raw)
    cfg["symbols"] = [s for s in cfg.get("symbols") or SYMBOLS if isinstance(s, str)] or list(SYMBOLS)
    return cfg


def update_config(patch):
    """merge `patch` into config.json on disk (keeps everything else)."""
    raw = read_json(CONFIG_PATH, {}) or {}
    raw = deep_merge(raw, patch)
    _write_config(raw)
    return deep_merge(DEFAULT_CONFIG, raw)


def clean_params(raw, base=None):
    """validated strategy parameters"""
    p = dict(base or DEFAULT_CONFIG["strategy"])
    raw = raw or {}

    def num(k, lo, hi, cast):
        if k in raw and raw[k] is not None:
            try:
                p[k] = cast(min(hi, max(lo, float(raw[k]))))
            except Exception:
                pass

    num("entry_n", 3, 500, int)
    num("impulse", 0, 10, float)
    num("ema_n", 10, 1000, int)
    num("atr_n", 5, 200, int)
    num("stop_atr", 0.3, 10, float)
    num("trail_atr", 0, 50, float)
    num("max_bars", 0, 1000, int)
    num("be_r", 0, 20, float)
    for k in ("trend_filter", "allow_long", "allow_short"):
        if k in raw:
            p[k] = bool(raw[k])
    return {k: p[k] for k in STRATEGY_KEYS}


def now_ms():
    return int(time.time() * 1000)


def tz(cfg):
    try:
        return ZoneInfo(cfg.get("timezone", "Europe/Berlin")) if ZoneInfo else timezone.utc
    except Exception:
        return timezone.utc


def tf_ms(cfg):
    return int(cfg.get("timeframe", 360)) * MIN


def fmt_ts(ms, cfg, f="%Y-%m-%d %H:%M"):
    return datetime.fromtimestamp(ms / 1000, tz(cfg)).strftime(f)


def instruments(cfg=None):
    """lot sizes per symbol: cached Bybit values over the built-in table"""
    cached = read_json(INSTR_PATH, {}) or {}
    out = {}
    for s in (cfg or DEFAULT_CONFIG)["symbols"]:
        out[s] = dict(INSTRUMENTS.get(s, {"qty_step": 0.001, "min_qty": 0.001, "tick": 0.0001}))
        out[s].update(cached.get(s, {}))
    return out


# ---------------------------------------------------------------- logging + discord
def _discord_post(url, payload):
    for attempt in range(3):
        try:
            if isinstance(payload, str):
                payload = {"content": payload[:1990]}
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 429:
                time.sleep(min(10.0, float((r.json() or {}).get("retry_after", 2))))
                continue
            return
        except Exception:
            time.sleep(2)


def discord_send(url, payload):
    if url and str(url).startswith("http"):
        threading.Thread(target=_discord_post, args=(url, payload), daemon=True).start()


class DiscordHandler(logging.Handler):
    """Every ERROR/CRITICAL record goes to the error webhook (identical messages max once / 5 min)."""

    def __init__(self, source):
        super().__init__(level=logging.ERROR)
        self.source, self._recent, self._lock = source, {}, threading.Lock()

    def emit(self, record):
        try:
            url = (read_json(CONFIG_PATH, {}) or {}).get("discord", {}).get("errors", "")
            if not url:
                return
            msg = record.getMessage()
            key = msg[:200]
            with self._lock:
                if time.time() - self._recent.get(key, 0) < 300:
                    return
                self._recent[key] = time.time()
            tb = "".join(traceback.format_exception(*record.exc_info)) if record.exc_info else ""
            body = msg + ("\n\n" + tb if tb else "")
            if len(body) > 1700:
                body = body[:500] + "\n...\n" + body[-1150:]
            discord_send(url, f"**🧊 ANTARCTICA {self.source} | {record.levelname}** "
                              f"`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n```{body}```")
        except Exception:
            pass


def setup_logging(source):
    log = logging.getLogger("antarctica")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [" + source + "] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.addHandler(DiscordHandler(source))

    def hook(t, v, tb):
        log.critical("uncaught exception", exc_info=(t, v, tb))

    sys.excepthook = hook

    def thook(a):
        log.critical(f"uncaught exception in thread {a.thread.name if a.thread else '?'}",
                     exc_info=(a.exc_type, a.exc_value, a.exc_traceback))

    threading.excepthook = thook
    return log


def push_event(ev, keep=500):
    """append to the small event feed shown on the website"""
    evs = read_json(EVENTS_PATH, []) or []
    evs.append(ev)
    save_json(EVENTS_PATH, evs[-keep:])


# ---------------------------------------------------------------- Bybit public data
class Bybit:
    def __init__(self, cfg):
        self.base = cfg["api_base"].rstrip("/")
        self.cat = cfg["category"]
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"antarctica-paper/{VERSION}"

    def _get(self, path, params, tries=4):
        err = None
        for attempt in range(tries):
            try:
                r = self.s.get(self.base + path, params=params, timeout=15)
                r.raise_for_status()
                j = r.json()
                if j.get("retCode") != 0:
                    raise RuntimeError(f"retCode={j.get('retCode')} {j.get('retMsg')}")
                return j["result"]
            except Exception as e:
                err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Bybit request failed {path}: {err}")

    def klines(self, sym, interval, start=None, end=None, limit=1000, tries=4):
        p = {"category": self.cat, "symbol": sym, "interval": str(interval), "limit": limit}
        if start is not None:
            p["start"] = int(start)
        if end is not None:
            p["end"] = int(end)
        rows = self._get("/v5/market/kline", p, tries=tries).get("list", [])
        out = [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]
        out.sort()
        return out

    def klines_range(self, sym, interval, start, end, progress=None):
        res, cur_end, total = {}, int(end), max(1, int(end) - int(start))
        while True:
            rows = self.klines(sym, interval, start=start, end=cur_end)
            if not rows:
                break
            for r in rows:
                res[r[0]] = r
            first = rows[0][0]
            if progress:
                progress(min(1.0, (end - first) / total))
            if first <= start or len(rows) < 1000:
                break
            cur_end = first - 1
            time.sleep(0.05)
        return [res[k] for k in sorted(res) if start <= k <= end]

    def funding_range(self, sym, start, end):
        res, cur_end = {}, int(end)
        while True:
            rows = self._get("/v5/market/funding/history",
                             {"category": self.cat, "symbol": sym, "startTime": int(start), "endTime": cur_end,
                              "limit": 200}).get("list", [])
            if not rows:
                break
            for r in rows:
                res[int(r["fundingRateTimestamp"])] = float(r["fundingRate"])
            oldest = min(int(r["fundingRateTimestamp"]) for r in rows)
            if oldest <= start or len(rows) < 200:
                break
            cur_end = oldest - 1
            time.sleep(0.05)
        return res

    def tickers(self):
        rows = self._get("/v5/market/tickers", {"category": self.cat}).get("list", [])
        return {r["symbol"]: r for r in rows}

    def instruments(self, symbols):
        out = {}
        for s in symbols:
            try:
                rows = self._get("/v5/market/instruments-info", {"category": self.cat, "symbol": s}).get("list", [])
                if rows:
                    lf, pf = rows[0].get("lotSizeFilter", {}), rows[0].get("priceFilter", {})
                    out[s] = {"qty_step": float(lf.get("qtyStep")), "min_qty": float(lf.get("minOrderQty")),
                              "tick": float(pf.get("tickSize")),
                              "min_notional": float(lf.get("minNotionalValue") or 5)}
            except Exception:
                pass
        return out


def refresh_instruments(cfg, log=None):
    try:
        got = Bybit(cfg).instruments(cfg["symbols"])
        if got:
            cur = read_json(INSTR_PATH, {}) or {}
            cur.update(got)
            save_json(INSTR_PATH, cur)
        return True
    except Exception as e:
        if log:
            log.warning(f"instrument info not refreshed: {e}")
        return False


class Store:
    """sqlite cache of CLOSED candles (any timeframe) and funding rates."""

    def __init__(self, api=None, path=None):
        self.api = api
        self.path = path or DB_PATH
        self.lock = threading.Lock()
        con = self._con()
        con.execute("CREATE TABLE IF NOT EXISTS candles (sym TEXT, tf INTEGER, ts INTEGER, o REAL, h REAL, l REAL, "
                    "c REAL, v REAL, PRIMARY KEY(sym, tf, ts))")
        con.execute("CREATE TABLE IF NOT EXISTS funding (sym TEXT, ts INTEGER, rate REAL, PRIMARY KEY(sym, ts))")
        con.commit()
        con.close()

    def _con(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def insert(self, sym, tf, rows, closed_before=None):
        lim = now_ms() if closed_before is None else closed_before
        rows = [(sym, int(tf)) + tuple(r) for r in rows if r[0] + tf * MIN <= lim]  # closed candles only
        if rows:
            with self.lock:
                con = self._con()
                con.executemany("INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?,?)", rows)
                con.commit()
                con.close()
        return len(rows)

    def bounds(self, sym, tf):
        con = self._con()
        r = con.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM candles WHERE sym=? AND tf=?", (sym, int(tf))).fetchone()
        con.close()
        return r

    def ensure(self, sym, tf, start, end, progress=None):
        """download whatever is missing between start and end (needs the api)"""
        end = min(int(end), now_ms())
        mn, mx, _ = self.bounds(sym, tf)
        segs = []
        if mn is None:
            segs.append((start, end))
        else:
            if start < mn - tf * MIN:
                segs.append((start, mn - 1))
            if end > mx + 2 * tf * MIN:
                segs.append((mx, end))
        for s, e in segs:
            self.insert(sym, tf, self.api.klines_range(sym, tf, s, e, progress=progress))

    def load(self, sym, tf, start, end):
        con = self._con()
        rows = con.execute("SELECT ts,o,h,l,c,v FROM candles WHERE sym=? AND tf=? AND ts>=? AND ts<=? ORDER BY ts",
                           (sym, int(tf), int(start), int(end))).fetchall()
        con.close()
        return rows_to_data(rows)

    def tail(self, sym, tf, n, end=None):
        con = self._con()
        rows = con.execute("SELECT ts,o,h,l,c,v FROM candles WHERE sym=? AND tf=? AND ts<=? ORDER BY ts DESC LIMIT ?",
                           (sym, int(tf), int(end if end is not None else 1 << 62), int(n))).fetchall()
        con.close()
        return rows_to_data(rows[::-1])

    def insert_funding(self, sym, d):
        if d:
            with self.lock:
                con = self._con()
                con.executemany("INSERT OR REPLACE INTO funding VALUES(?,?,?)",
                                [(sym, int(t), float(r)) for t, r in d.items()])
                con.commit()
                con.close()

    def ensure_funding(self, sym, start, end):
        con = self._con()
        mn, mx = con.execute("SELECT MIN(ts), MAX(ts) FROM funding WHERE sym=?", (sym,)).fetchone()
        con.close()
        segs = []
        if mn is None:
            segs.append((start, end))
        else:
            if start < mn - 8 * HOUR:
                segs.append((start, mn - 1))
            if end > mx + HOUR:
                segs.append((mx + 1, end))
        for s, e in segs:
            self.insert_funding(sym, self.api.funding_range(sym, s, e))

    def load_funding(self, sym, start, end):
        con = self._con()
        rows = con.execute("SELECT ts, rate FROM funding WHERE sym=? AND ts>=? AND ts<=? ORDER BY ts",
                           (sym, int(start), int(end))).fetchall()
        con.close()
        return {int(t): float(r) for t, r in rows}


def rows_to_data(rows):
    if not rows:
        return {"ts": np.array([], dtype=np.int64), **{k: np.array([]) for k in ("o", "h", "l", "c", "v")}}
    a = np.array(rows, dtype=float)
    return {"ts": a[:, 0].astype(np.int64), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": a[:, 5]}


# ---------------------------------------------------------------- indicators (causal, fixed windows)
def _roll(a, n, fn):
    """fn over the window [i-n+1 .. i]; nan while the window is not full. Each window is computed on its own,
    so the value at candle i is bit-identical no matter where the array starts (live == backtest)."""
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        w = np.lib.stride_tricks.sliding_window_view(a, n)
        out[n - 1:] = fn(w, axis=1)
    return out


def _shift1(a):
    out = np.full(len(a), np.nan)
    out[1:] = a[:-1]
    return out


def ema_window(c, n, mult=5):
    """EMA(n) computed over a fixed window of mult*n candles, seeded with the first candle of the window.
    Truncation vs. an infinite EMA is < 0.005 %, and the value is identical wherever the array starts."""
    W = int(mult * n)
    out = np.full(len(c), np.nan)
    if len(c) < W:
        return out
    a = 2.0 / (n + 1.0)
    k = a * (1.0 - a) ** np.arange(W - 1)  # weights of c[i], c[i-1], ... c[i-W+2]
    seed_w = (1.0 - a) ** (W - 1)
    w = np.lib.stride_tricks.sliding_window_view(c, W)  # rows: c[i-W+1 .. i]
    out[W - 1:] = w[:, 1:][:, ::-1] @ k + seed_w * w[:, 0]
    return out


def indicators(d, p, cache=None):
    """All indicator arrays for candle data d. Index i only uses candles <= i."""
    h, l, c = d["h"], d["l"], d["c"]

    def memo(key, fn):
        if cache is None:
            return fn()
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    def _tr():
        pc = _shift1(c)
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        if len(tr):
            tr[0] = h[0] - l[0]
        return tr

    tr = memo("tr", _tr)
    atr = memo(("atr", p["atr_n"]), lambda: _roll(tr, p["atr_n"], np.mean))
    hh = memo(("hh", p["entry_n"]), lambda: _shift1(_roll(h, p["entry_n"], np.max)))
    ll = memo(("ll", p["entry_n"]), lambda: _shift1(_roll(l, p["entry_n"], np.min)))
    ema = memo(("ema", p["ema_n"]), lambda: ema_window(c, p["ema_n"]))
    return {"atr": atr, "hh": hh, "ll": ll, "ema": ema}


IND_KEYS = ("atr", "hh", "ll", "ema")


def warmup_bars(p):
    return int(max(p["entry_n"] + 1, p["atr_n"] + 1, 5 * p["ema_n"]) + 5)


# ---------------------------------------------------------------- the engine
class Engine:
    """One account, many instruments, at most one position per instrument. Fed candle by candle with a
    two-phase protocol per candle time T (identical in the paper trader and the backtester):

      open_bar(sym, ts, o)              phase 1, all symbols: fill entries decided at the previous close
      process(sym, bar, ind, final)     phase 2, all symbols: protective stop; when final=True also funding,
                                        trailing, time stop and new signals (decided at the close)
    For final=False the bar is still forming (live only): only the protective stop is checked. The live result
    of a candle is identical to what the backtester computes from the closed candle afterwards.
    """

    def __init__(self, params, cfg, start_balance=None, risk_pct=None, funding=None, trade_from=0, instr=None):
        a, f = cfg["account"], cfg["fees"]
        self.p = clean_params(params)
        self.cfg_symbols = list(cfg["symbols"])
        self.tf = tf_ms(cfg)
        self.taker, self.maker, self.slip = f["taker"], f["maker"], f.get("slippage", 0.0)
        self.use_funding = bool(f.get("include_funding", True))
        self.start_balance = float(a["start_balance"] if start_balance is None else start_balance)
        self.risk_pct = float(a["risk_pct"] if risk_pct is None else risk_pct)
        self.heat_pct = float(a.get("heat_pct", 12.0))
        self.max_lev = float(a.get("max_leverage", 10))
        self.lot = bool(a.get("lot_rounding", True))
        self.min_notional = float(a.get("min_notional", 5.0))
        self.minqty_mult = float(a.get("minqty_risk_mult", 2.0))
        self.instr = instr or instruments(cfg)
        self.kelly = bool(a.get("kelly_sizing", False))
        self.kelly_frac = float(a.get("kelly_fraction", 0.5))
        self.kelly_min = int(a.get("kelly_min_trades", 40))
        self.kelly_max = float(a.get("kelly_max_risk_pct", 4.0))
        self.funding = funding  # {sym: {ts: rate}} (backtest) or a callable(sym, start, end) -> {ts: rate} (live)
        self.trade_from = int(trade_from)  # no new signals before this time (warm-up)
        self.balance = self.start_balance
        self.pos = {}
        self.pending = {}
        self.trades = []
        self.seen_ts = {}
        self.closed_ts = {}
        self.seq = 0
        self.days = []  # [day_start_ms_utc, mark-to-market equity at the first candle open of that day]
        self.last_px = {}
        self.events = []
        self._rs = []
        self.last = {}  # sym -> (bar, ind) of the last closed candle (status display)

    # ---- persistence
    KEYS = ("balance", "pos", "pending", "trades", "seen_ts", "closed_ts", "seq", "days", "last_px",
            "start_balance", "trade_from")

    def export(self):
        return {k: getattr(self, k) for k in self.KEYS}

    def load(self, st):
        for k in self.KEYS:
            if k in st:
                setattr(self, k, st[k])
        self._rs = [t["r"] for t in self.trades]

    # ---- sizing helpers
    def current_risk_pct(self):
        if self.kelly and len(self._rs) >= self.kelly_min:
            k = kelly(self._rs)["full"]
            return max(0.1, min(self.kelly_max, k * 100 * self.kelly_frac)) if k > 0 else 0.1
        return self.risk_pct

    def open_heat(self):
        return sum(p["risk_pct"] for p in self.pos.values())

    def _ev(self, typ, ts, **kw):
        self.events.append({"type": typ, "ts": int(ts), **kw})

    def upnl(self, sym, price):
        ps = self.pos.get(sym)
        if not ps:
            return 0.0
        return (price - ps["entry"]) * ps["dir"] * ps["qty"] - ps["fee_in"] - ps["funding"]

    def equity(self, prices=None):
        prices = prices or self.last_px
        return self.balance + sum(self.upnl(s, prices.get(s, p["entry"])) for s, p in self.pos.items())

    # ---- phase 1
    def open_bar(self, sym, ts, o):
        """first look at candle `ts` of `sym`: day mark + fill a pending entry at the open"""
        if ts <= self.seen_ts.get(sym, 0):
            return
        if not self.days or ts // DAY > self.days[-1][0] // DAY:
            self.days.append([int(ts // DAY * DAY), self.equity()])
        self.last_px[sym] = o
        pd_ = self.pending.pop(sym, None)
        if pd_:
            if ts == pd_["signal_ts"] + self.tf:
                self._open(sym, pd_, ts, o)
            else:
                self._ev("skip", ts, sym=sym, reason="entry candle missed (data gap or downtime)", dir=pd_["dir"])
        self.seen_ts[sym] = int(ts)

    # ---- phase 2
    def process(self, sym, bar, ind=None, final=True):
        ts, o, h, l, c, v = bar
        if ts > self.seen_ts.get(sym, 0):
            self.open_bar(sym, ts, o)
        ps = self.pos.get(sym)
        if ps:  # ---------- protective stop
            d, st = ps["dir"], ps["stop"]
            px = None
            if d == 1:
                if ps["entry_ts"] != ts and o <= st:
                    px = o * (1 - self.slip)
                elif l <= st:
                    px = st * (1 - self.slip)
            else:
                if ps["entry_ts"] != ts and o >= st:
                    px = o * (1 + self.slip)
                elif h >= st:
                    px = st * (1 + self.slip)
            if px is not None:
                self._close(sym, ts, px, "trail" if ps["trails"] else "stop")
                ps = None
        if not final or ts <= self.closed_ts.get(sym, 0):
            if not final:
                self.last_px[sym] = c
            return
        # ---------- the candle is closed
        self.closed_ts[sym] = int(ts)
        self.last_px[sym] = c
        atr, hh, ll, ema = ind
        self.last[sym] = (bar, ind)
        p = self.p
        ps = self.pos.get(sym)
        if ps:
            if self.use_funding:
                self._funding(sym, ps, ts, o)
            d = ps["dir"]
            ext = h if d == 1 else l  # best price of this candle in the trade's direction
            if (ext - ps["best"]) * d > 0:
                ps["best"] = ext
            if (ext - ps["entry"]) * d > ps["mfe"]:
                ps["mfe"] = (ext - ps["entry"]) * d
            adv = ((l if d == 1 else h) - ps["entry"]) * d
            if adv < ps["mae"]:
                ps["mae"] = adv
            ps["bars"] += 1
            ns = ps["stop"]
            if p["be_r"] > 0 and (ps["best"] - ps["entry"]) * d >= p["be_r"] * ps["dist"]:
                b = ps["entry"] + d * (2 * self.taker * ps["entry"])
                if (b - ns) * d > 0:
                    ns = b
            if p["trail_atr"] > 0 and atr == atr:
                t = ps["best"] - d * p["trail_atr"] * atr
                if (t - ns) * d > 0:
                    ns = t
            if (c - ns) * d > 0 and ns != ps["stop"]:
                self._trail(sym, ts, ns)
            if p["max_bars"] > 0 and ps["bars"] >= p["max_bars"]:
                self._close(sym, ts + self.tf, c * (1 - self.slip * d), "time")
            return
        if sym in self.pending or ts < self.trade_from:
            return
        if atr != atr or hh != hh or ll != ll or atr <= 0 or (p["trend_filter"] and ema != ema):
            return
        d = 1 if c > hh else (-1 if c < ll else 0)
        if not d or (d == 1 and not p["allow_long"]) or (d == -1 and not p["allow_short"]):
            return
        rng = (h - l) / atr
        if rng < p["impulse"]:
            return
        if p["trend_filter"] and not ((d == 1 and c > ema) or (d == -1 and c < ema)):
            return
        self.pending[sym] = {"dir": d, "signal_ts": int(ts), "atr": float(atr), "dist": float(p["stop_atr"] * atr),
                             "level": float(hh if d == 1 else ll), "close": float(c), "rng": float(rng),
                             "ema": float(ema) if ema == ema else None}
        self._ev("signal", ts, sym=sym, dir=d, level=self.pending[sym]["level"], close=float(c), rng=float(rng))

    # ---- internals
    def _open(self, sym, pd_, ts, o):
        d = pd_["dir"]
        entry = o * (1 + self.slip * d)
        dist = pd_["dist"]
        want = self.current_risk_pct()
        room = self.heat_pct - self.open_heat()
        risk_pct = min(want, room)
        if risk_pct < want * 0.5 - 1e-9:
            self._ev("skip", ts, sym=sym, dir=d, reason=f"portfolio heat full ({self.open_heat():.1f}% of {self.heat_pct:.1f}% in use)")
            return
        if self.balance <= 0 or dist <= 0:
            self._ev("skip", ts, sym=sym, dir=d, reason="no balance")
            return
        risk_usd = self.balance * risk_pct / 100.0
        qty = risk_usd / dist
        lev_room = self.balance * self.max_lev - sum(p["qty"] * p["entry"] for p in self.pos.values())
        capped = False
        if qty * entry > lev_room:
            qty = max(0.0, lev_room / entry)
            capped = True
        if self.lot:
            ins = self.instr.get(sym, {"qty_step": 0.001, "min_qty": 0.001})
            step, mn = float(ins["qty_step"]), float(ins["min_qty"])
            min_val = max(mn, math.ceil(self.min_notional / entry / step - 1e-9) * step)
            qty = math.floor(qty / step + 1e-9) * step
            if qty < min_val - 1e-12:
                if min_val * dist <= self.minqty_mult * risk_usd and min_val * entry <= lev_room:
                    qty = min_val
                else:
                    self._ev("skip", ts, sym=sym, dir=d,
                             reason=f"minimum order {min_val:g} would risk ${min_val * dist:.2f} (> {self.minqty_mult:g}x ${risk_usd:.2f})")
                    return
            qty = round(qty, 10)
        if qty <= 0:
            self._ev("skip", ts, sym=sym, dir=d, reason="leverage limit reached")
            return
        self.seq += 1
        stop = entry - d * dist
        real_risk = qty * dist
        self.pos[sym] = {"id": self.seq, "sym": sym, "dir": d, "qty": qty, "entry": entry, "entry_ts": int(ts),
                         "signal_ts": pd_["signal_ts"], "stop": stop, "init_stop": stop, "dist": dist,
                         "risk_usd": real_risk, "risk_pct": real_risk / self.balance * 100,
                         "fee_in": qty * entry * self.taker, "funding": 0.0, "atr": pd_["atr"], "level": pd_["level"],
                         "bal_before": self.balance, "trails": 0, "mfe": 0.0, "mae": 0.0, "lev_capped": capped,
                         "best": entry, "bars": 0, "stops": [[int(ts), stop]]}
        self._ev("open", ts, sym=sym, pos=dict(self.pos[sym]))

    def _trail(self, sym, ts, new_stop):
        ps = self.pos[sym]
        old = ps["stop"]
        ps["stop"] = float(new_stop)
        ps["trails"] += 1
        ps["stops"].append([int(ts + self.tf), float(new_stop)])
        self._ev("trail", ts, sym=sym, id=ps["id"], dir=ps["dir"], old=old, new=float(new_stop), entry=ps["entry"],
                 qty=ps["qty"], locked_r=(new_stop - ps["entry"]) * ps["dir"] * ps["qty"] / ps["risk_usd"])

    def _funding(self, sym, ps, ts, o):
        """funding events inside candle ts charged at the candle's close if the position is still open"""
        src = self.funding
        if src is None:
            return
        rates = src(sym, ts, ts + self.tf - 1) if callable(src) else {
            k: r for k, r in (src.get(sym) or {}).items() if ts <= k < ts + self.tf}
        for fts in sorted(rates):
            if fts <= ps["entry_ts"] or fts in ps.setdefault("fund_ts", []):
                continue
            cost = ps["qty"] * o * ps["dir"] * rates[fts]
            ps["funding"] += cost
            ps["fund_ts"].append(int(fts))
            self._ev("funding", fts, sym=sym, id=ps["id"], rate=rates[fts], cost=cost)

    def _close(self, sym, ts, px, reason):
        ps = self.pos.pop(sym)
        d, q = ps["dir"], ps["qty"]
        gross = (px - ps["entry"]) * d * q
        fee_out = q * px * self.taker
        fees = ps["fee_in"] + fee_out
        pnl = gross - fees - ps["funding"]
        self.balance += pnl
        r = pnl / ps["risk_usd"] if ps["risk_usd"] else 0.0
        t = {"id": ps["id"], "sym": sym, "dir": d, "qty": q, "entry": ps["entry"], "exit": px, "entry_ts": ps["entry_ts"],
             "exit_ts": int(ts), "signal_ts": ps["signal_ts"], "init_stop": ps["init_stop"], "stop": ps["stop"],
             "risk_usd": ps["risk_usd"], "risk_pct": ps["risk_pct"], "gross": gross, "fees": fees,
             "funding": ps["funding"], "pnl": pnl, "r": r, "reason": reason, "bal_before": ps["bal_before"],
             "bal_after": self.balance, "notional": q * ps["entry"],
             "lev": q * ps["entry"] / ps["bal_before"] if ps["bal_before"] > 0 else 0.0,
             "sl_pct": abs(ps["entry"] - ps["init_stop"]) / ps["entry"] * 100,
             "mfe_r": ps["mfe"] * q / ps["risk_usd"] if ps["risk_usd"] else 0.0,
             "mae_r": ps["mae"] * q / ps["risk_usd"] if ps["risk_usd"] else 0.0, "trails": ps["trails"],
             "bars": ps["bars"], "stops": ps["stops"]}
        self.trades.append(t)
        self._rs.append(r)
        self._ev("close", ts, sym=sym, trade=t)

    def force_close(self, sym, ts, price, reason="end"):
        if sym in self.pos:
            self._close(sym, ts, price * (1 - self.slip * self.pos[sym]["dir"]), reason)


def build_indicators(datas, params, caches=None):
    out = {}
    for s, d in datas.items():
        ind = indicators(d, params, None if caches is None else caches.setdefault(s, {}))
        out[s] = [ind[k].tolist() for k in IND_KEYS]
    return out


def run_backtest(datas, params, cfg, start_ms=0, funding=None, balance=None, risk_pct=None, caches=None,
                 close_end=True, end_ms=None, instr=None):
    """datas = {sym: candle arrays} (must include the warm-up candles before start_ms).
    Two-phase candle protocol, exactly like the live trader. Returns the finished Engine."""
    params = clean_params(params)
    syms = [s for s in cfg["symbols"] if s in datas and len(datas[s]["ts"])]
    inds = build_indicators({s: datas[s] for s in syms}, params, caches)
    eng = Engine(params, cfg, start_balance=balance, risk_pct=risk_pct,
                 funding=funding if funding is not None else {}, trade_from=start_ms, instr=instr)
    # timeline: every candle time of every symbol from (start - 1 candle) on
    cols = {}
    for s in syms:
        d = datas[s]
        i0 = max(0, int(np.searchsorted(d["ts"], start_ms)) - 1) if start_ms else 0
        i1 = len(d["ts"]) if end_ms is None else int(np.searchsorted(d["ts"], end_ms, side="right"))
        cols[s] = (i0, i1, d["ts"].tolist(), d["o"].tolist(), d["h"].tolist(), d["l"].tolist(), d["c"].tolist(),
                   d["v"].tolist())
    events = {}
    for s in syms:
        i0, i1, ts = cols[s][0], cols[s][1], cols[s][2]
        for i in range(i0, i1):
            events.setdefault(ts[i], []).append((s, i))
    order = {s: k for k, s in enumerate(syms)}
    last_ts = 0
    for t in sorted(events):
        lst = sorted(events[t], key=lambda x: order[x[0]])
        for s, i in lst:  # phase 1: fills at the open
            eng.open_bar(s, t, cols[s][3][i])
        for s, i in lst:  # phase 2: stops, closes, signals
            c = cols[s]
            ia = inds[s]
            eng.process(s, (t, c[3][i], c[4][i], c[5][i], c[6][i], c[7][i]), (ia[0][i], ia[1][i], ia[2][i], ia[3][i]), True)
        last_ts = t
    eng.events = []
    eng.end_ts = last_ts + eng.tf if events else start_ms
    if close_end:
        for s in list(eng.pos):
            c = cols[s]
            eng.force_close(s, eng.end_ts, c[6][c[1] - 1], "end of test")
            eng.trades[-1]["open_at_end"] = True
    return eng


# ---------------------------------------------------------------- statistics
def kelly(rs):
    """Kelly fraction of equity to risk per trade, from a list of R multiples"""
    rs = [r for r in rs if r == r]
    wins = [r for r in rs if r > 0]
    losses = [-r for r in rs if r <= 0]
    if not wins or not losses:
        return {"full": 0.0, "half": 0.0, "p": len(wins) / len(rs) if rs else 0.0, "b": 0.0, "n": len(rs)}
    p = len(wins) / len(rs)
    aw, al = sum(wins) / len(wins), sum(losses) / len(losses)
    b = aw / al if al > 0 else 0.0
    f = (p - (1 - p) / b) / al if b > 0 else 0.0  # fraction of equity risked (1 R) per trade
    f = max(0.0, f)
    return {"full": f, "half": f / 2, "p": p, "b": b, "n": len(rs)}


def _streaks(flags):
    bw = bl = cw = cl = 0
    for w in flags:
        if w:
            cw += 1
            cl = 0
        else:
            cl += 1
            cw = 0
        bw, bl = max(bw, cw), max(bl, cl)
    return bw, bl, cw, cl


def compute_stats(trades, cfg, start_balance, start_ms, end_ms, days_eq=None, open_pos=None, prices=None,
                  risk_pct=None, balance=None):
    """every number on the overview. trades = closed trades in exit order; open_pos = {sym: pos}."""
    n = len(trades)
    bal = balance if balance is not None else (trades[-1]["bal_after"] if trades else start_balance)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    rs = [t["r"] for t in trades]
    gp, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    cum, peak, dd_r = 0.0, 0.0, 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        dd_r = max(dd_r, peak - cum)
    curve = [[int(start_ms), start_balance]] + [[t["exit_ts"], t["bal_after"]] for t in trades]
    upnl = 0.0
    for s, ps in (open_pos or {}).items():
        px = (prices or {}).get(s)
        if px:
            upnl += (px - ps["entry"]) * ps["dir"] * ps["qty"] - ps["fee_in"] - ps["funding"]
    equity = bal + upnl
    curve.append([int(end_ms), equity])
    pk, dd_pct, dd_usd = start_balance, 0.0, 0.0
    for _, e in curve:
        pk = max(pk, e)
        if pk > 0 and (pk - e) / pk * 100 > dd_pct:
            dd_pct, dd_usd = (pk - e) / pk * 100, pk - e
    daily_dd = 0.0
    ds = [e for e in (days_eq or []) if e[0] >= start_ms - DAY] + [[int(end_ms), equity]]
    pk2 = start_balance
    for i in range(1, len(ds)):
        if ds[i - 1][1] > 0:
            daily_dd = max(daily_dd, (ds[i - 1][1] - ds[i][1]) / ds[i - 1][1] * 100)
        pk2 = max(pk2, ds[i - 1][1], ds[i][1])
        if pk2 > 0:
            dd_pct = max(dd_pct, (pk2 - ds[i][1]) / pk2 * 100)
    days = max((end_ms - start_ms) / DAY, 1e-9)
    growth = equity / start_balance if start_balance > 0 else 1.0
    cagr = (growth ** (365.0 / days) - 1) * 100 if days >= 1 and growth > 0 else 0.0
    geo_month = (growth ** (30.4375 / days) - 1) * 100 if days >= 1 and growth > 0 else 0.0
    months = month_returns(trades, cfg, start_balance, start_ms, end_ms)
    avg_month = sum(m["pct"] for m in months) / len(months) if months else 0.0
    bw, bl, cw, cl = _streaks([t["pnl"] > 0 for t in trades])
    aw_r = sum(t["r"] for t in wins) / len(wins) if wins else 0.0
    al_r = -sum(t["r"] for t in losses) / len(losses) if losses else 0.0
    fees = sum(t["fees"] for t in trades)
    fund = sum(t["funding"] for t in trades)
    k = kelly(rs)
    f = cfg["fees"]
    per_sym = {}
    for t in trades:
        x = per_sym.setdefault(t["sym"], {"n": 0, "wins": 0, "r": 0.0, "pnl": 0.0})
        x["n"] += 1
        x["wins"] += 1 if t["pnl"] > 0 else 0
        x["r"] += t["r"]
        x["pnl"] += t["pnl"]
    return {
        "trades": n, "wins": len(wins), "losses": len(losses), "winrate": len(wins) / n * 100 if n else 0.0,
        "cum_r": sum(rs), "max_dd_r": dd_r, "avg_r": sum(rs) / n if n else 0.0,
        "start_balance": start_balance, "balance": bal, "equity": equity, "upnl": upnl,
        "risk_pct": risk_pct if risk_pct is not None else cfg["account"]["risk_pct"],
        "heat_pct": cfg["account"].get("heat_pct", 12.0),
        "fee_maker": f["maker"] * 100, "fee_taker": f["taker"] * 100, "slippage": f.get("slippage", 0) * 100,
        "days": days, "net_pnl": equity - start_balance, "net_pct": (growth - 1) * 100,
        "profit_factor": gp / gl if gl > 0 else (None if gp == 0 else 999.0),
        "annual_return": cagr, "avg_month": avg_month, "geo_month": geo_month, "months": len(months),
        "pos_months": sum(1 for m in months if m["pct"] > 0),
        "best_month": max((m["pct"] for m in months), default=0.0),
        "worst_month": min((m["pct"] for m in months), default=0.0),
        "avg_win": gp / len(wins) if wins else 0.0, "avg_loss": -gl / len(losses) if losses else 0.0,
        "avg_win_r": aw_r, "avg_loss_r": -al_r, "avg_rtr": aw_r / al_r if al_r > 0 else 0.0,
        "best_r": max(rs, default=0.0), "worst_r": min(rs, default=0.0),
        "max_dd_pct": dd_pct, "max_dd_usd": dd_usd, "max_daily_dd": daily_dd,
        "max_consec_wins": bw, "max_consec_losses": bl, "cur_wins": cw, "cur_losses": cl,
        "fees": fees, "funding": fund, "avg_fee": fees / n if n else 0.0,
        "avg_size": sum(t["notional"] for t in trades) / n if n else 0.0,
        "avg_lev": sum(t["lev"] for t in trades) / n if n else 0.0,
        "avg_hold_h": sum(t["exit_ts"] - t["entry_ts"] for t in trades) / n / HOUR if n else 0.0,
        "avg_hold_win_h": sum(t["exit_ts"] - t["entry_ts"] for t in wins) / len(wins) / HOUR if wins else 0.0,
        "avg_hold_loss_h": sum(t["exit_ts"] - t["entry_ts"] for t in losses) / len(losses) / HOUR if losses else 0.0,
        "avg_sl_pct": sum(t["sl_pct"] for t in trades) / n if n else 0.0,
        "avg_risk_pct": sum(t["risk_pct"] for t in trades) / n if n else 0.0,
        "longs": sum(1 for t in trades if t["dir"] == 1), "shorts": sum(1 for t in trades if t["dir"] == -1),
        "long_r": sum(t["r"] for t in trades if t["dir"] == 1), "short_r": sum(t["r"] for t in trades if t["dir"] == -1),
        "kelly_full": k["full"] * 100, "kelly_half": k["half"] * 100, "kelly_p": k["p"] * 100, "kelly_b": k["b"],
        "per_sym": per_sym, "open_count": len(open_pos or {}),
        "curve": curve, "r_curve": _r_curve(trades, start_ms),
    }


def _r_curve(trades, start_ms):
    out, cum = [[int(start_ms), 0.0]], 0.0
    for t in trades:
        cum += t["r"]
        out.append([t["exit_ts"], cum])
    return out


def _period_keys(ms, z):
    d = datetime.fromtimestamp(ms / 1000, z)
    iso = d.isocalendar()
    return (f"{d.year}", f"{d.year}-{d.month:02d}", f"{iso[0]}-W{iso[1]:02d}", f"{d.year}-{d.month:02d}-{d.day:02d}")


def month_returns(trades, cfg, start_balance, start_ms, end_ms):
    """calendar months (config timezone) between start and end, incl. months without trades"""
    z = tz(cfg)
    by = {}
    for t in trades:
        by.setdefault(_period_keys(t["exit_ts"], z)[1], []).append(t)
    out, bal = [], start_balance
    d = datetime.fromtimestamp(start_ms / 1000, z).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = datetime.fromtimestamp(max(end_ms - 1, start_ms) / 1000, z)
    while d <= end:
        key = f"{d.year}-{d.month:02d}"
        pnl = sum(t["pnl"] for t in by.get(key, []))
        out.append({"key": key, "start": bal, "pnl": pnl, "pct": pnl / bal * 100 if bal > 0 else 0.0,
                    "trades": len(by.get(key, []))})
        bal += pnl
        d = (d.replace(day=28) + timedelta(days=5)).replace(day=1)
    return out


def journal(trades, cfg, start_balance, start_ms, end_ms):
    """year -> month -> week -> day balances with parent links for drill-down."""
    z = tz(cfg)
    end_ms = max(end_ms, start_ms + 1)
    days = {}
    d = datetime.fromtimestamp(start_ms / 1000, z).replace(hour=0, minute=0, second=0, microsecond=0)
    end = datetime.fromtimestamp((end_ms - 1) / 1000, z)
    order = []
    while d <= end:
        iso = d.isocalendar()
        key = f"{d.year}-{d.month:02d}-{d.day:02d}"
        days[key] = {"key": key, "y": f"{d.year}", "m": f"{d.year}-{d.month:02d}", "w": f"{iso[0]}-W{iso[1]:02d}",
                     "pnl": 0.0, "r": 0.0, "fees": 0.0, "n": 0, "wins": 0}
        order.append(key)
        d = (d + timedelta(days=1)).replace(hour=0)
    for t in trades:
        k = _period_keys(t["exit_ts"], z)[3]
        if k not in days:
            continue
        x = days[k]
        x["pnl"] += t["pnl"]
        x["r"] += t["r"]
        x["fees"] += t["fees"]
        x["n"] += 1
        x["wins"] += 1 if t["pnl"] > 0 else 0
    out = {"day": [], "week": [], "month": [], "year": []}
    bal = start_balance
    agg = {"week": {}, "month": {}, "year": {}}
    for key in order:
        x = days[key]
        row = {"key": key, "start": bal, "end": bal + x["pnl"], "pnl": x["pnl"], "pct": x["pnl"] / bal * 100 if bal > 0 else 0.0,
               "r": x["r"], "n": x["n"], "wins": x["wins"], "fees": x["fees"], "week": x["w"], "month": x["m"], "year": x["y"]}
        out["day"].append(row)
        for lvl, pk in (("week", x["w"]), ("month", x["m"]), ("year", x["y"])):
            a = agg[lvl].get(pk)
            if a is None:
                a = agg[lvl][pk] = {"key": pk, "start": bal, "end": bal, "pnl": 0.0, "r": 0.0, "n": 0, "wins": 0,
                                    "fees": 0.0, "from": key, "to": key, "year": x["y"]}
                if lvl == "week":
                    a["month"] = x["m"]
                out[lvl].append(a)
            a["end"] = bal + x["pnl"]
            a["pnl"] += x["pnl"]
            a["r"] += x["r"]
            a["n"] += x["n"]
            a["wins"] += x["wins"]
            a["fees"] += x["fees"]
            a["to"] = key
        bal += x["pnl"]
    for lvl in ("week", "month", "year"):
        for a in out[lvl]:
            a["pct"] = a["pnl"] / a["start"] * 100 if a["start"] > 0 else 0.0
    return out


def bundle(eng, cfg, start_ms, end_ms, prices=None, risk_pct=None, with_trades=True):
    """everything the website needs to draw an account (live paper account or a backtest)"""
    prices = prices or eng.last_px
    st = compute_stats(eng.trades, cfg, eng.start_balance, start_ms, end_ms, days_eq=eng.days, open_pos=eng.pos,
                       prices=prices, risk_pct=risk_pct if risk_pct is not None else eng.risk_pct, balance=eng.balance)
    out = {"stats": st, "journal": journal(eng.trades, cfg, eng.start_balance, start_ms, end_ms),
           "months": month_returns(eng.trades, cfg, eng.start_balance, start_ms, end_ms),
           "open": eng.pos, "pending": eng.pending, "start_ms": int(start_ms), "end_ms": int(end_ms),
           "params": eng.p, "prices": prices, "symbols": eng.cfg_symbols}
    if with_trades:
        out["trades"] = eng.trades
    return out


# ---------------------------------------------------------------- discord trade messages
COIN = {"BTCUSDT": "₿", "ETHUSDT": "Ξ"}


def fmt_px(x):
    if x is None:
        return "-"
    a = abs(x)
    n = 1 if a >= 1000 else 2 if a >= 100 else 3 if a >= 10 else 4 if a >= 1 else 5 if a >= 0.1 else 6
    return f"{x:,.{n}f}"


def fmt_discord(ev, cfg):
    sym = ev.get("sym", "?")
    if ev["type"] == "open":
        p = ev["pos"]
        side = "🟢 LONG" if p["dir"] == 1 else "🔴 SHORT"
        return {"embeds": [{
            "title": f"🧊 ANTARCTICA · {sym} · {side} opened",
            "color": 0x5CF2C8 if p["dir"] == 1 else 0xFF5F7E,
            "fields": [
                {"name": "Entry", "value": f"`{fmt_px(p['entry'])}`", "inline": True},
                {"name": "Stop", "value": f"`{fmt_px(p['stop'])}` ({abs(p['entry'] - p['stop']) / p['entry'] * 100:.2f}%)", "inline": True},
                {"name": "Size", "value": f"`{p['qty']:g}` (${p['qty'] * p['entry']:,.2f})", "inline": True},
                {"name": "Risk", "value": f"${p['risk_usd']:.2f} ({p['risk_pct']:.2f}%)", "inline": True},
                {"name": "Breakout level", "value": f"`{fmt_px(p['level'])}`", "inline": True},
                {"name": "Balance", "value": f"${p['bal_before']:,.2f}", "inline": True}],
            "footer": {"text": fmt_ts(ev["ts"], cfg) + " · paper · 6h Aurora Breakout"}}]}
    if ev["type"] == "trail":
        return (f"🧊 **{sym}** trail #{ev['id']}: stop `{fmt_px(ev['old'])}` → `{fmt_px(ev['new'])}` "
                f"(locks {ev['locked_r']:+.2f} R)")
    if ev["type"] == "close":
        t = ev["trade"]
        win = t["pnl"] > 0
        return {"embeds": [{
            "title": f"🧊 ANTARCTICA · {sym} · {'LONG' if t['dir'] == 1 else 'SHORT'} closed · {t['r']:+.2f} R",
            "color": 0x5CF2C8 if win else 0xFF5F7E,
            "fields": [
                {"name": "Entry → Exit", "value": f"`{fmt_px(t['entry'])}` → `{fmt_px(t['exit'])}`", "inline": True},
                {"name": "PnL", "value": f"**${t['pnl']:+.2f}** ({t['r']:+.2f} R)", "inline": True},
                {"name": "Reason", "value": t["reason"], "inline": True},
                {"name": "Fees / funding", "value": f"${t['fees']:.3f} / ${t['funding']:.3f}", "inline": True},
                {"name": "Held", "value": f"{(t['exit_ts'] - t['entry_ts']) / HOUR:.1f} h", "inline": True},
                {"name": "Balance", "value": f"${t['bal_after']:,.2f}", "inline": True}],
            "footer": {"text": fmt_ts(ev["ts"], cfg) + " · paper"}}]}
    if ev["type"] == "skip":
        return f"🧊 **{sym}** signal skipped: {ev.get('reason', '')}"
    return None


def signal_status(sym, eng, bar, ind, forming=None, next_levels=None):
    """what the strategy waits for on `sym` (used by the website's strategy status panel).
    bar/ind = last CLOSED candle + its indicators; forming = the candle that is still open;
    next_levels = (highest high, lowest low) of the last entry_n closed candles = the levels the forming
    candle will be compared with when it closes."""
    p = eng.p
    out = {"sym": sym, "in_pos": sym in eng.pos, "pending": eng.pending.get(sym)}
    if not bar or not ind:
        return out
    ts, o, h, l, c, v = bar
    atr, hh, ll, ema = ind
    out.update({"close": c, "atr": atr, "hh": hh, "ll": ll, "ema": ema, "bar_ts": ts})
    cur = forming or bar
    fh, fl, fc = cur[2], cur[3], cur[4]
    lvl_hi, lvl_lo = next_levels if next_levels else (hh, ll)
    out["next_hh"], out["next_ll"] = lvl_hi, lvl_lo
    if atr == atr and atr:
        rng = (fh - fl) / atr
        out["long"] = {"breakout": fc > lvl_hi, "impulse": rng >= p["impulse"],
                       "trend": (not p["trend_filter"]) or (ema == ema and fc > ema), "allowed": p["allow_long"],
                       "dist_pct": (lvl_hi - fc) / fc * 100, "level": lvl_hi}
        out["short"] = {"breakout": fc < lvl_lo, "impulse": rng >= p["impulse"],
                        "trend": (not p["trend_filter"]) or (ema == ema and fc < ema), "allowed": p["allow_short"],
                        "dist_pct": (fc - lvl_lo) / fc * 100, "level": lvl_lo}
        out["range_atr"] = rng
    return out
