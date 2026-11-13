#!/usr/bin/env python3
"""
ANTARCTICA core - shared by the paper trader, the backtester, the optimizer and the website.

Strategy "Aurora Ensemble" (see antarctica_strategy.py): a long/short book of 12 Bybit USDT perpetuals on 4h
candles, built from three sleeves (trend, funding carry, cross-sectional momentum), each an ensemble of many
parameter sets, volatility-targeted and multiplied by `scale` (default 4).

Execution model (identical in the paper trader and the backtester):
  * at the CLOSE of candle t: funding of candle t is charged, equity is marked, the target weights of candle t
    are computed from data up to t only
  * at the OPEN of candle t+1: the book is re-balanced to those targets with market orders (taker fee +
    slippage), Bybit lot sizes, minimum order value, leverage cap and a no-trade band
NO LOOK-AHEAD: a decision uses candles <= t and is filled on candle t+1.
"""
import bisect
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

import antarctica_strategy as strat

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
    "timeframe": 240,
    "poll_seconds": 15,
    "account": {
        "start_balance": 1000.0,
        "max_leverage": 8.0,
        "band_pct": 2.0,
        "min_notional": 5.0,
        "lot_rounding": True,
        "kelly_sizing": False,
        "kelly_fraction": 0.5,
        "kelly_max_scale": 6.0,
    },
    "fees": {
        "maker": 0.0002,
        "taker": 0.00055,
        "slippage": 0.0001,
        "include_funding": True,
    },
    "strategy": dict(strat.DEFAULT_PARAMS),
    "optimizer": {
        "days": 1460,
        "combos": 40,
        "train_frac": 0.6,
        "workers": 0,
        "space": {
            "lt": [[14, 21, 30, 45, 60], [21, 30, 45], [30, 45, 60, 90], [7, 14, 21, 30]],
            "lc": [[3, 7, 14], [7, 14], [3, 7]],
            "lx": [[7, 14, 21, 30], [14, 21, 30], [7, 14]],
            "hs": [[18, 42], [12, 30], [42, 84], [6, 18, 42]],
            "w_ts": [1.0, 1.0, 1.5, 0.5],
            "w_carry": [1.0, 1.0, 1.5, 0.5],
            "w_xs": [1.0, 1.0, 0.5, 0.0],
            "scale": [4.0],
        },
    },
    "discord": {"errors": "", "notify_start": True, "rebalance_summary": True, **{s: "" for s in SYMBOLS}},
    "web": {"port": 8760, "passcode": "", "autolock_minutes": 10},
    "demo": {"scale_mult": 1.0, "max_leverage": 8},
    "live": {"scale_mult": 0.5, "max_leverage": 5},
}



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
    if raw is not None and "lt" not in (raw.get("strategy") or {}):
        # config of the old breakout version: keep passcode, webhooks, timezone, reset the rest
        raw = {"symbols": raw.get("symbols", SYMBOLS), "timezone": raw.get("timezone", "Europe/Berlin"),
               "account": DEFAULT_CONFIG["account"], "fees": DEFAULT_CONFIG["fees"], "strategy": DEFAULT_CONFIG["strategy"],
               "discord": deep_merge(DEFAULT_CONFIG["discord"], raw.get("discord") or {}),
               "web": deep_merge(DEFAULT_CONFIG["web"], raw.get("web") or {})}
        _write_config(raw)
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
    return strat.clean_params(raw, base)


def now_ms():
    return int(time.time() * 1000)


def tz(cfg):
    try:
        return ZoneInfo(cfg.get("timezone", "Europe/Berlin")) if ZoneInfo else timezone.utc
    except Exception:
        return timezone.utc


def tf_ms(cfg):
    return int(cfg.get("timeframe", 240)) * MIN


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


# ---------------------------------------------------------------- the engine
class Engine:
    """One account, a long/short book over many instruments. Fed candle by candle (same order live and backtest):

      on_open(ts, opens)                     re-balance to the targets decided at the previous close
      on_close(ts, opens, closes, targets)   funding of this candle, mark-to-market, store new targets

    A "trade" (for the statistics) is one position episode of one coin: it starts when the coin's position
    leaves 0 (or flips side) and ends when it returns to 0 (or flips). Size changes in between are
    adjustments of the same trade. R = PnL in % of the equity when the trade started (1 R = 1 % of equity).
    """

    def __init__(self, params, cfg, start_balance=None, funding=None, trade_from=0, instr=None, scale=None):
        a, f = cfg["account"], cfg["fees"]
        self.p = clean_params(params)
        if scale is not None:
            self.p["scale"] = float(scale)
        self.cfg_symbols = list(cfg["symbols"])
        self.tf = tf_ms(cfg)
        self.taker, self.maker, self.slip = f["taker"], f["maker"], f.get("slippage", 0.0)
        self.use_funding = bool(f.get("include_funding", True))
        self.start_balance = float(a["start_balance"] if start_balance is None else start_balance)
        self.max_lev = float(a.get("max_leverage", 8))
        self.band = float(a.get("band_pct", 1.0)) / 100.0
        self.min_notional = float(a.get("min_notional", 5.0))
        self.lot = bool(a.get("lot_rounding", True))
        self.instr = instr or instruments(cfg)
        self.kelly = bool(a.get("kelly_sizing", False))
        self.kelly_frac = float(a.get("kelly_fraction", 0.5))
        self.kelly_max = float(a.get("kelly_max_scale", 6.0))
        self.funding = funding  # {sym: {ts: rate}} or callable(sym, start, end) -> {ts: rate}
        self.trade_from = int(trade_from)
        self.cash = self.start_balance
        self.pos = {}
        self.trades = []
        self.seq = 0
        self.days = []        # [day_start_ms, equity at the first close of that day]
        self.curve = []       # [close_ts, equity, gross exposure] (every candle in backtests, thinned live)
        self.targets = {}     # weights decided at the last close
        self.target_ts = 0
        self.sleeves = {}     # per-sleeve weights of the last close (status display)
        self.last_close_ts = 0
        self.last_open_ts = 0
        self.last_px = {}
        self.fees_total = 0.0
        self.funding_total = 0.0
        self.turnover = 0.0
        self.rebalances = 0
        self.unit_rets = []   # [day_ms, daily return / scale used] for Kelly sizing
        self.cur_scale = self.p["scale"]
        self.events = []

    KEYS = ("cash", "pos", "trades", "seq", "days", "curve", "targets", "target_ts", "sleeves", "last_close_ts",
            "last_open_ts", "last_px", "fees_total", "funding_total", "turnover", "rebalances", "unit_rets",
            "cur_scale", "start_balance", "trade_from")

    def export(self):
        return {k: getattr(self, k) for k in self.KEYS}

    def load(self, st):
        for k in self.KEYS:
            if k in st:
                setattr(self, k, st[k])

    @property
    def balance(self):
        return self.cash

    def _ev(self, typ, ts, **kw):
        self.events.append({"type": typ, "ts": int(ts), **kw})

    def upnl(self, sym, price):
        ps = self.pos.get(sym)
        return (price - ps["avg"]) * ps["qty"] if ps and price else 0.0

    def equity(self, prices=None):
        prices = prices or self.last_px
        return self.cash + sum(self.upnl(s, prices.get(s) or p["avg"]) for s, p in self.pos.items())

    def gross(self, prices=None):
        prices = prices or self.last_px
        return sum(abs(p["qty"]) * (prices.get(s) or p["avg"]) for s, p in self.pos.items())

    def kelly_scale(self):
        """fraction x full-Kelly scale estimated from PAST daily returns per unit of scale (last 365 days)"""
        rs = [r for _, r in self.unit_rets[-365:]]
        if len(rs) < 120:
            return self.p["scale"]
        m = sum(rs) / len(rs)
        v = sum((x - m) ** 2 for x in rs) / (len(rs) - 1)
        if v <= 0 or m <= 0:
            return 0.5
        return max(0.5, min(self.kelly_max, self.kelly_frac * m / v))

    # ---- candle protocol
    def on_open(self, ts, opens):
        """re-balance at the open of candle ts to the targets decided at the close of ts - 1 candle"""
        if ts <= self.last_open_ts:
            return
        self.last_open_ts = int(ts)
        for s, o in opens.items():
            if o == o and o:
                self.last_px[s] = o
        if not self.targets or ts != self.target_ts + self.tf:
            return
        self._rebalance(ts, opens)

    def on_close(self, ts, opens, closes, targets, sleeves=None):
        if ts <= self.last_close_ts:
            return
        self.last_close_ts = int(ts)
        if self.use_funding and self.pos:
            self._funding(ts, opens)
        for s, c in closes.items():
            if c == c and c:
                self.last_px[s] = c
        eq = self.equity()
        day = ts // DAY * DAY
        if not self.days or day > self.days[-1][0]:
            if self.days:
                prev = self.days[-1][1]
                if prev > 0 and self.cur_scale > 0:
                    self.unit_rets.append([int(self.days[-1][0]), (eq / prev - 1) / self.cur_scale])
                    self.unit_rets = self.unit_rets[-400:]
            self.days.append([int(day), eq])
        self.curve.append([int(ts + self.tf), eq, self.gross() / eq if eq > 0 else 0.0])
        if ts >= self.trade_from - self.tf:
            sc = self.kelly_scale() if self.kelly else self.p["scale"]
            base = self.p["scale"] if self.p["scale"] > 0 else 1.0
            self.cur_scale = sc
            self.targets = {s: float(w) * sc / base for s, w in targets.items() if w == w}
            self.sleeves = sleeves or {}
            self.target_ts = int(ts)

    # ---- internals
    def _funding(self, ts, opens):
        src = self.funding
        if src is None:
            return
        for s, ps in list(self.pos.items()):
            rates = src(s, ts, ts + self.tf - 1) if callable(src) else self._fund_window(s, ts)
            px = opens.get(s) or self.last_px.get(s) or ps["avg"]
            for fts in sorted(rates):
                if fts <= ps["open_ts"]:
                    continue
                cost = ps["qty"] * px * rates[fts]  # longs pay positive funding, shorts receive it
                ps["funding"] += cost
                self.cash -= cost
                self.funding_total += cost

    def _fund_window(self, s, ts):
        idx = getattr(self, "_fidx", None)
        if idx is None:
            idx = self._fidx = {}
        if s not in idx:
            d = self.funding.get(s) or {}
            ks = sorted(d)
            idx[s] = (ks, [d[k] for k in ks])
        ks, rs = idx[s]
        a = bisect.bisect_left(ks, ts)
        b = bisect.bisect_left(ks, ts + self.tf)
        return {ks[i]: rs[i] for i in range(a, b)}

    def _round(self, sym, qty, price):
        if not self.lot:
            return qty
        ins = self.instr.get(sym, {"qty_step": 0.001, "min_qty": 0.001})
        step, mn = float(ins["qty_step"]), float(ins["min_qty"])
        q = math.floor(abs(qty) / step + 1e-9) * step
        if q < mn - 1e-12 or q * price < self.min_notional:
            return 0.0
        return round(math.copysign(q, qty), 10)

    def _rebalance(self, ts, opens):
        eq = self.equity(opens)
        if eq <= 0:
            return
        tg = {s: w for s, w in self.targets.items() if opens.get(s) == opens.get(s) and opens.get(s)}
        g = sum(abs(w) for w in tg.values())
        if g > self.max_lev:
            tg = {s: w * self.max_lev / g for s, w in tg.items()}
        syms = [s for s in self.cfg_symbols if s in tg or s in self.pos]
        changed = []
        for s in syms:
            o = opens.get(s)
            if not o or o != o:
                continue
            cur = self.pos[s]["qty"] if s in self.pos else 0.0
            want = self._round(s, tg.get(s, 0.0) * eq / o, o)
            diff = want - cur
            if abs(diff) < 1e-12:
                continue
            same_side = cur != 0 and want != 0 and (cur > 0) == (want > 0)
            if same_side and abs(diff) * o < self.band * eq:
                continue
            self._fill(s, ts, diff, o)
            changed.append(s)
        if changed:
            self.rebalances += 1
            self._ev("rebalance", ts, syms=changed, equity=self.equity(opens),
                     weights={s: (self.pos[s]["qty"] * opens.get(s, self.pos[s]["avg"]) / eq if s in self.pos else 0.0) for s in syms})

    def _fill(self, s, ts, diff, o):
        px = o * (1 + self.slip * (1 if diff > 0 else -1))
        ps = self.pos.get(s)
        cur = ps["qty"] if ps else 0.0
        # part that reduces / closes the current position
        if cur and (cur > 0) != (diff > 0):
            red = min(abs(diff), abs(cur))
            self._trade_part(s, ts, -math.copysign(red, cur), px)
            diff += math.copysign(red, cur)
            ps = self.pos.get(s)
        if abs(diff) > 1e-12:
            self._trade_part(s, ts, diff, px)

    def _trade_part(self, s, ts, q, px):
        fee = abs(q) * px * self.taker
        self.cash -= fee
        self.fees_total += fee
        self.turnover += abs(q) * px
        ps = self.pos.get(s)
        if ps is None:  # new trade
            self.seq += 1
            eq = self.equity()
            ps = self.pos[s] = {"id": self.seq, "sym": s, "dir": 1 if q > 0 else -1, "qty": 0.0, "avg": px,
                                "entry": px, "open_ts": int(ts), "fees": 0.0, "funding": 0.0, "realized": 0.0,
                                "eq_open": eq, "max_notional": 0.0, "adj": 0, "fills": []}
            self._ev("open", ts, sym=s, dir=ps["dir"], qty=q, price=px, notional=abs(q) * px, equity=eq)
        else:
            ps["adj"] += 1
        ps["fees"] += fee
        ps["fills"].append([int(ts), q, px])
        ps["fills"] = ps["fills"][-60:]
        if (ps["qty"] >= 0) == (q >= 0) or ps["qty"] == 0:  # increase
            nq = ps["qty"] + q
            ps["avg"] = (ps["avg"] * abs(ps["qty"]) + px * abs(q)) / abs(nq) if nq else px
            ps["qty"] = nq
            if ps["adj"]:
                self._ev("adjust", ts, sym=s, qty=q, price=px, new_qty=nq)
        else:  # reduce
            gain = (px - ps["avg"]) * (-q)
            ps["realized"] += gain
            self.cash += gain
            ps["qty"] += q
            if abs(ps["qty"]) < 1e-12:
                self._close(s, ts, px)
            else:
                self._ev("adjust", ts, sym=s, qty=q, price=px, new_qty=ps["qty"])
        if s in self.pos:
            ps["max_notional"] = max(ps["max_notional"], abs(ps["qty"]) * px)

    def _close(self, s, ts, px, reason="rebalance"):
        ps = self.pos.pop(s)
        pnl = ps["realized"] - ps["fees"] - ps["funding"]
        r = pnl / (ps["eq_open"] * 0.01) if ps["eq_open"] > 0 else 0.0
        t = {"id": ps["id"], "sym": s, "dir": ps["dir"], "qty": max(abs(f[1]) for f in ps["fills"]) if ps["fills"] else 0,
             "entry": ps["entry"], "exit": px, "entry_ts": ps["open_ts"], "exit_ts": int(ts), "fees": ps["fees"],
             "funding": ps["funding"], "gross": ps["realized"], "pnl": pnl, "r": r, "reason": reason,
             "bal_before": ps["eq_open"], "bal_after": self.cash, "notional": ps["max_notional"],
             "lev": ps["max_notional"] / ps["eq_open"] if ps["eq_open"] > 0 else 0.0, "adj": ps["adj"],
             "risk_usd": ps["eq_open"] * 0.01}
        self.trades.append(t)
        self._ev("close", ts, sym=s, trade=t)

    def force_close(self, ts, prices, reason="end of test"):
        for s in list(self.pos):
            ps = self.pos[s]
            px = prices.get(s) or ps["avg"]
            q = -ps["qty"]
            fee = abs(q) * px * self.taker
            self.cash -= fee
            self.fees_total += fee
            ps["fees"] += fee
            gain = (px - ps["avg"]) * ps["qty"]
            ps["realized"] += gain
            self.cash += gain
            self._close(s, ts, px, reason)
            self.trades[-1]["open_at_end"] = True


def load_panel(datas, funding, cfg, start=None, end=None):
    return strat.panel(datas, funding or {}, cfg["symbols"], tf_ms(cfg), start, end)


def run_backtest(datas, params, cfg, start_ms=0, funding=None, balance=None, end_ms=None, instr=None, scale=None,
                 close_end=True, weights=None, progress=None):
    """datas = {sym: candles incl. warm-up before start_ms}. Same candle protocol as the live trader."""
    p = clean_params(params)
    if scale is not None:
        p["scale"] = float(scale)
    tf = tf_ms(cfg)
    fund = funding if (funding is not None and cfg["fees"].get("include_funding", True)) else {}
    T, O, H, L, C, FR = load_panel(datas, funding or {}, cfg, end=end_ms)
    syms = cfg["symbols"]
    if weights is None:
        W, SL = strat.compute(T, C, FR, p, tf, detail=True)
    else:
        W, SL = weights
    eng = Engine(p, cfg, start_balance=balance, funding=fund, trade_from=start_ms, instr=instr)
    i0 = max(0, int(np.searchsorted(T, start_ms)) - 1)
    nan = float("nan")
    for i in range(i0, len(T)):
        t = int(T[i])
        opens = {s: O[i, j] for j, s in enumerate(syms) if O[i, j] == O[i, j]}
        closes = {s: C[i, j] for j, s in enumerate(syms) if C[i, j] == C[i, j]}
        eng.on_open(t, opens)
        tg = {s: W[i, j] for j, s in enumerate(syms) if C[i, j] == C[i, j]}
        sl = {k: {s: float(v[i, j]) for j, s in enumerate(syms)} for k, v in SL.items()} if i == len(T) - 1 else None
        eng.on_close(t, opens, closes, tg if t >= start_ms - tf else {}, sl)
        if progress and i % 2000 == 0:
            progress(i / len(T))
    eng.events = []
    eng.end_ts = int(T[-1]) + tf if len(T) else start_ms
    if close_end:
        last = {s: C[-1, j] for j, s in enumerate(syms) if C[-1, j] == C[-1, j]}
        eng.force_close(eng.end_ts, last)
    return eng


# ---------------------------------------------------------------- statistics
def kelly(rs):
    """Kelly fraction from a list of per-period returns (or R multiples): mean / variance"""
    rs = [r for r in rs if r == r]
    if len(rs) < 2:
        return {"full": 0.0, "half": 0.0}
    m = sum(rs) / len(rs)
    v = sum((x - m) ** 2 for x in rs) / (len(rs) - 1)
    f = max(0.0, m / v) if v > 0 else 0.0
    return {"full": f, "half": f / 2}


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


def equity_series(eng, start_ms, end_ms, prices=None):
    pts = [[int(start_ms), eng.start_balance, 0.0]] + [c for c in eng.curve if start_ms < c[0] <= end_ms]
    eq_now = eng.equity(prices)
    if not pts or pts[-1][0] < end_ms:
        pts.append([int(end_ms), eq_now, eng.gross(prices) / eq_now if eq_now > 0 else 0.0])
    return pts


def month_returns(curve, cfg, start_ms, end_ms):
    z = tz(cfg)
    out, cur, first, last = [], None, None, None
    for t, e, *_ in curve:
        k = datetime.fromtimestamp(max(start_ms, t - 1) / 1000, z).strftime("%Y-%m")
        if k != cur:
            if cur is not None:
                out.append({"key": cur, "start": first, "end": last, "pnl": last - first,
                            "pct": (last / first - 1) * 100 if first > 0 else 0.0})
            cur, first = k, (last if last is not None else e)
        last = e
    if cur is not None:
        out.append({"key": cur, "start": first, "end": last, "pnl": last - first,
                    "pct": (last / first - 1) * 100 if first > 0 else 0.0})
    return out


def compute_stats(eng, cfg, start_ms, end_ms, prices=None, trades=None):
    trades = eng.trades if trades is None else trades
    curve = equity_series(eng, start_ms, end_ms, prices)
    sb = curve[0][1]
    equity = curve[-1][1]
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    rs = [t["r"] for t in trades]
    gp, gl = sum(t["pnl"] for t in wins), -sum(t["pnl"] for t in losses)
    cum, peak, dd_r = 0.0, 0.0, 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        dd_r = max(dd_r, peak - cum)
    pk, dd_pct, dd_usd = sb, 0.0, 0.0
    for _, e, *_ in curve:
        pk = max(pk, e)
        if pk > 0 and (pk - e) / pk * 100 > dd_pct:
            dd_pct, dd_usd = (pk - e) / pk * 100, pk - e
    # daily closes for daily DD and Kelly
    z = tz(cfg)
    daily = {}
    for t, e, *_ in curve:
        daily[datetime.fromtimestamp((t - 1) / 1000, z).strftime("%Y-%m-%d")] = e
    dv = list(daily.values())
    dret = [dv[i] / dv[i - 1] - 1 for i in range(1, len(dv)) if dv[i - 1] > 0]
    daily_dd = max([-r * 100 for r in dret] + [0.0])
    days = max((end_ms - start_ms) / DAY, 1e-9)
    growth = equity / sb if sb > 0 else 1.0
    cagr = (growth ** (365.0 / days) - 1) * 100 if days >= 1 and growth > 0 else 0.0
    geo_month = (growth ** (30.4375 / days) - 1) * 100 if days >= 1 and growth > 0 else 0.0
    months = month_returns(curve, cfg, start_ms, end_ms)
    avg_month = sum(m["pct"] for m in months) / len(months) if months else 0.0
    bw, bl, cw, cl = _streaks([t["pnl"] > 0 for t in trades])
    aw_r = sum(t["r"] for t in wins) / len(wins) if wins else 0.0
    al_r = -sum(t["r"] for t in losses) / len(losses) if losses else 0.0
    scale = eng.p.get("scale", 1.0) or 1.0
    k = kelly(dret)
    sharpe = 0.0
    if len(dret) > 2:
        m = sum(dret) / len(dret)
        sd = (sum((x - m) ** 2 for x in dret) / (len(dret) - 1)) ** 0.5
        sharpe = m / sd * (365 ** 0.5) if sd > 0 else 0.0
    per_sym = {}
    for t in trades:
        x = per_sym.setdefault(t["sym"], {"n": 0, "wins": 0, "r": 0.0, "pnl": 0.0})
        x["n"] += 1
        x["wins"] += 1 if t["pnl"] > 0 else 0
        x["r"] += t["r"]
        x["pnl"] += t["pnl"]
    for s, ps in eng.pos.items():
        x = per_sym.setdefault(s, {"n": 0, "wins": 0, "r": 0.0, "pnl": 0.0})
        x["open"] = eng.upnl(s, (prices or eng.last_px).get(s)) - ps["fees"] - ps["funding"] + ps["realized"]
    gross_avg = sum(c[2] for c in curve[1:]) / max(1, len(curve) - 1)
    f = cfg["fees"]
    thin = curve if len(curve) <= 3000 else curve[:: max(1, len(curve) // 3000)] + [curve[-1]]
    return {
        "trades": n, "wins": len(wins), "losses": len(losses), "winrate": len(wins) / n * 100 if n else 0.0,
        "cum_r": sum(rs), "max_dd_r": dd_r, "avg_r": sum(rs) / n if n else 0.0,
        "start_balance": sb, "balance": eng.cash, "equity": equity, "upnl": equity - eng.cash,
        "scale": scale, "cur_scale": eng.cur_scale, "band_pct": eng.band * 100, "max_lev": eng.max_lev,
        "fee_maker": f["maker"] * 100, "fee_taker": f["taker"] * 100, "slippage": f.get("slippage", 0) * 100,
        "days": days, "net_pnl": equity - sb, "net_pct": (growth - 1) * 100,
        "profit_factor": gp / gl if gl > 0 else (None if gp == 0 else 999.0),
        "annual_return": cagr, "avg_month": avg_month, "geo_month": geo_month, "months": len(months),
        "pos_months": sum(1 for m in months if m["pct"] > 0),
        "best_month": max((m["pct"] for m in months), default=0.0),
        "worst_month": min((m["pct"] for m in months), default=0.0), "sharpe": sharpe,
        "avg_win": gp / len(wins) if wins else 0.0, "avg_loss": -gl / len(losses) if losses else 0.0,
        "avg_win_r": aw_r, "avg_loss_r": -al_r, "avg_rtr": aw_r / al_r if al_r > 0 else 0.0,
        "best_r": max(rs, default=0.0), "worst_r": min(rs, default=0.0),
        "max_dd_pct": dd_pct, "max_dd_usd": dd_usd, "max_daily_dd": daily_dd,
        "max_consec_wins": bw, "max_consec_losses": bl, "cur_wins": cw, "cur_losses": cl,
        "fees": eng.fees_total, "funding": eng.funding_total,
        "avg_fee": sum(t["fees"] for t in trades) / n if n else 0.0,
        "avg_size": sum(t["notional"] for t in trades) / n if n else 0.0,
        "avg_lev": gross_avg, "rebalances": eng.rebalances, "turnover": eng.turnover,
        "avg_hold_h": sum(t["exit_ts"] - t["entry_ts"] for t in trades) / n / HOUR if n else 0.0,
        "avg_hold_win_h": sum(t["exit_ts"] - t["entry_ts"] for t in wins) / len(wins) / HOUR if wins else 0.0,
        "avg_hold_loss_h": sum(t["exit_ts"] - t["entry_ts"] for t in losses) / len(losses) / HOUR if losses else 0.0,
        "longs": sum(1 for t in trades if t["dir"] == 1), "shorts": sum(1 for t in trades if t["dir"] == -1),
        "long_r": sum(t["r"] for t in trades if t["dir"] == 1), "short_r": sum(t["r"] for t in trades if t["dir"] == -1),
        "kelly_full": k["full"] / scale, "kelly_half": k["half"] / scale, "kelly_scale_full": k["full"],
        "per_sym": per_sym, "open_count": len(eng.pos),
        "curve": [[c[0], c[1]] for c in thin], "gross_curve": [[c[0], c[2]] for c in thin],
        "r_curve": _r_curve(trades, start_ms),
    }


def _r_curve(trades, start_ms):
    out, cum = [[int(start_ms), 0.0]], 0.0
    for t in sorted(trades, key=lambda x: x["exit_ts"]):
        cum += t["r"]
        out.append([t["exit_ts"], cum])
    return out


def journal(eng, cfg, start_ms, end_ms, prices=None):
    """year -> month -> week -> day equity (mark-to-market) with drill-down links"""
    z = tz(cfg)
    curve = equity_series(eng, start_ms, end_ms, prices)
    eod = {}
    for t, e, *_ in curve:
        eod[datetime.fromtimestamp((max(t, start_ms + 1) - 1) / 1000, z).strftime("%Y-%m-%d")] = e
    tcount = {}
    for t in eng.trades:
        k = datetime.fromtimestamp(t["exit_ts"] / 1000, z).strftime("%Y-%m-%d")
        x = tcount.setdefault(k, [0, 0, 0.0])
        x[0] += 1
        x[1] += 1 if t["pnl"] > 0 else 0
        x[2] += t["r"]
    out = {"day": [], "week": [], "month": [], "year": []}
    d = datetime.fromtimestamp(start_ms / 1000, z).replace(hour=0, minute=0, second=0, microsecond=0)
    end = datetime.fromtimestamp((max(end_ms, start_ms + 1) - 1) / 1000, z)
    bal = curve[0][1]
    agg = {"week": {}, "month": {}, "year": {}}
    while d <= end:
        key = d.strftime("%Y-%m-%d")
        iso = d.isocalendar()
        w, m, y = f"{iso[0]}-W{iso[1]:02d}", d.strftime("%Y-%m"), d.strftime("%Y")
        e = eod.get(key, bal)
        tc = tcount.get(key, [0, 0, 0.0])
        out["day"].append({"key": key, "start": bal, "end": e, "pnl": e - bal, "pct": (e / bal - 1) * 100 if bal > 0 else 0.0,
                           "n": tc[0], "wins": tc[1], "r": tc[2], "week": w, "month": m, "year": y})
        for lvl, pk in (("week", w), ("month", m), ("year", y)):
            a = agg[lvl].get(pk)
            if a is None:
                a = agg[lvl][pk] = {"key": pk, "start": bal, "end": bal, "pnl": 0.0, "n": 0, "wins": 0, "r": 0.0,
                                    "from": key, "to": key, "year": y}
                if lvl == "week":
                    a["month"] = m
                out[lvl].append(a)
            a["end"] = e
            a["pnl"] = e - a["start"]
            a["n"] += tc[0]
            a["wins"] += tc[1]
            a["r"] += tc[2]
            a["to"] = key
        bal = e
        d = (d + timedelta(days=1)).replace(hour=0)
    for lvl in ("week", "month", "year"):
        for a in out[lvl]:
            a["pct"] = (a["end"] / a["start"] - 1) * 100 if a["start"] > 0 else 0.0
    return out


def open_positions(eng, prices=None):
    prices = prices or eng.last_px
    eq = eng.equity(prices)
    out = {}
    for s, ps in eng.pos.items():
        px = prices.get(s) or ps["avg"]
        out[s] = {"sym": s, "dir": 1 if ps["qty"] > 0 else -1, "qty": abs(ps["qty"]), "entry": ps["avg"],
                  "first_entry": ps["entry"], "open_ts": ps["open_ts"], "mark": px, "notional": abs(ps["qty"]) * px,
                  "weight": ps["qty"] * px / eq if eq > 0 else 0.0, "upnl": eng.upnl(s, px),
                  "pnl": eng.upnl(s, px) + ps["realized"] - ps["fees"] - ps["funding"], "fees": ps["fees"],
                  "funding": ps["funding"], "adj": ps["adj"], "target": eng.targets.get(s, 0.0),
                  "fills": ps["fills"][-20:]}
    return out


def bundle(eng, cfg, start_ms, end_ms, prices=None, with_trades=True):
    prices = prices or eng.last_px
    st = compute_stats(eng, cfg, start_ms, end_ms, prices)
    out = {"stats": st, "journal": journal(eng, cfg, start_ms, end_ms, prices),
           "months": month_returns(equity_series(eng, start_ms, end_ms, prices), cfg, start_ms, end_ms),
           "open": open_positions(eng, prices), "targets": eng.targets, "target_ts": eng.target_ts,
           "sleeves": eng.sleeves, "start_ms": int(start_ms), "end_ms": int(end_ms), "params": eng.p,
           "prices": prices, "symbols": eng.cfg_symbols}
    if with_trades:
        out["trades"] = eng.trades
    return out


# ---------------------------------------------------------------- discord messages
def fmt_px(x):
    if x is None:
        return "-"
    a = abs(x)
    n = 1 if a >= 1000 else 2 if a >= 100 else 3 if a >= 10 else 4 if a >= 1 else 5 if a >= 0.1 else 6
    return f"{x:,.{n}f}"


def fmt_discord(ev, cfg):
    sym = ev.get("sym", "?")
    if ev["type"] == "open":
        side = "🟢 LONG" if ev["dir"] == 1 else "🔴 SHORT"
        return {"embeds": [{
            "title": f"🧊 ANTARCTICA · {sym} · {side} opened",
            "color": 0x5CF2C8 if ev["dir"] == 1 else 0xFF5F7E,
            "fields": [
                {"name": "Price", "value": f"`{fmt_px(ev['price'])}`", "inline": True},
                {"name": "Size", "value": f"`{abs(ev['qty']):g}` (${ev['notional']:,.2f})", "inline": True},
                {"name": "Weight", "value": f"{ev['notional'] / ev['equity'] * 100:.1f} % of equity", "inline": True},
                {"name": "Equity", "value": f"${ev['equity']:,.2f}", "inline": True}],
            "footer": {"text": fmt_ts(ev["ts"], cfg) + " · paper · Aurora Ensemble"}}]}
    if ev["type"] == "adjust":
        return (f"🧊 **{sym}** {'added' if (ev['qty'] > 0) == (ev['new_qty'] > 0) else 'reduced'} "
                f"`{abs(ev['qty']):g}` @ `{fmt_px(ev['price'])}` → position `{ev['new_qty']:+g}`")
    if ev["type"] == "close":
        t = ev["trade"]
        return {"embeds": [{
            "title": f"🧊 ANTARCTICA · {sym} · {'LONG' if t['dir'] == 1 else 'SHORT'} closed · {t['r']:+.2f} R",
            "color": 0x5CF2C8 if t["pnl"] > 0 else 0xFF5F7E,
            "fields": [
                {"name": "Entry → Exit", "value": f"`{fmt_px(t['entry'])}` → `{fmt_px(t['exit'])}`", "inline": True},
                {"name": "PnL", "value": f"**${t['pnl']:+.2f}** ({t['r']:+.2f} R · 1 R = 1 % equity)", "inline": True},
                {"name": "Fees / funding", "value": f"${t['fees']:.2f} / ${t['funding']:.2f}", "inline": True},
                {"name": "Held", "value": f"{(t['exit_ts'] - t['entry_ts']) / DAY:.1f} d · {t['adj']} adjustments", "inline": True},
                {"name": "Balance", "value": f"${t['bal_after']:,.2f}", "inline": True}],
            "footer": {"text": fmt_ts(ev["ts"], cfg) + " · paper"}}]}
    return None


def fmt_rebalance(ev, cfg):
    w = ev.get("weights", {})
    parts = [f"{s.replace('USDT', '')} {v * 100:+.0f}%" for s, v in sorted(w.items(), key=lambda x: -abs(x[1])) if abs(v) > 0.005]
    return (f"🧊 **ANTARCTICA re-balance** {fmt_ts(ev['ts'], cfg)} · equity ${ev['equity']:,.2f} · "
            f"gross {sum(abs(v) for v in w.values()):.2f}x\n" + " · ".join(parts))
