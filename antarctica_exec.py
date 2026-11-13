#!/usr/bin/env python3
"""
ANTARCTICA exchange executors.

  DEMO  mirrors the paper book on Bybit Demo Trading (api-demo.bybit.com) once demo API keys are saved on the
        website (CONTROL tab) and the executor is ARMED: after every paper re-balance each coin's position is
        set to paper weight x demo equity x scale_mult with market orders (reduce-only when shrinking).
  LIVE  real money (api.bybit.com). Keys can be stored and the connection tested, but order execution is
        deliberately NOT enabled (LIVE_ENABLED = False). Nothing in this file places a real-money order while
        that flag is False - an armed live executor only logs what it would do.

API keys live in data/secrets.json (chmod 600, never sent back to the browser, never in git).
"""
import hashlib
import logging
import hmac
import json
import math
import os
import threading
import time
import urllib.parse

import requests

import antarctica_core as core

HOSTS = {"demo": "https://api-demo.bybit.com", "live": "https://api.bybit.com"}
LIVE_ENABLED = False
RECV = "10000"
log = logging.getLogger("antarctica")  # handlers are set up by the trader / web process


class ExecError(RuntimeError):
    pass


# ---------------------------------------------------------------- key storage
def load_keys(kind):
    s = (core.read_json(core.SECRETS_PATH, {}) or {}).get(kind, {})
    return s.get("api_key", ""), s.get("api_secret", "")


def save_keys(kind, key, secret):
    if kind not in HOSTS:
        raise ExecError("unknown account type")
    s = core.read_json(core.SECRETS_PATH, {}) or {}
    if key or secret:
        s[kind] = {"api_key": key.strip(), "api_secret": secret.strip(), "saved": core.now_ms()}
    else:
        s.pop(kind, None)
    core.save_json(core.SECRETS_PATH, s)
    try:
        os.chmod(core.SECRETS_PATH, 0o600)
    except Exception:
        pass


def keys_info(kind):
    s = (core.read_json(core.SECRETS_PATH, {}) or {}).get(kind, {})
    k = s.get("api_key", "")
    return {"present": bool(k and s.get("api_secret")),
            "masked": (k[:4] + "•" * 8 + k[-3:]) if len(k) > 8 else ("•" * len(k)), "saved": s.get("saved")}


def control_path(kind):
    return os.path.join(core.DATA_DIR, f"exec_{kind}_control.json")


def state_path(kind):
    return os.path.join(core.DATA_DIR, f"exec_{kind}_state.json")


def set_control(kind, **kw):
    c = core.read_json(control_path(kind), {}) or {}
    c.update(kw)
    core.save_json(control_path(kind), c)
    return c


# ---------------------------------------------------------------- signed REST client
class BybitPrivate:
    """Minimal signed Bybit v5 REST client."""

    def __init__(self, key, secret, host):
        if host not in HOSTS.values() and not host.startswith("http://127.0.0.1"):
            raise ExecError(f"host {host} not allowed")
        self.host, self.key, self.secret = host, key or "", secret or ""
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"antarctica/{core.VERSION}"
        self.lock = threading.Lock()
        self.time_offset = 0

    def ready(self):
        return bool(self.key and self.secret)

    def _sign(self, ts, payload):
        return hmac.new(self.secret.encode(), f"{ts}{self.key}{RECV}{payload}".encode(), hashlib.sha256).hexdigest()

    def sync_time(self):
        try:
            j = self.s.get(self.host + "/v5/market/time", timeout=10).json()
            srv = int(j.get("time") or int(j["result"]["timeNano"]) // 1_000_000)
            self.time_offset = srv - int(time.time() * 1000)
        except Exception:
            pass

    def req(self, method, path, params=None, signed=True, retries=2):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        err = None
        for attempt in range(retries + 1):
            try:
                ts = str(int(time.time() * 1000) + self.time_offset)
                headers = {}
                if method == "GET":
                    qs = urllib.parse.urlencode(params)
                    url, payload, body = self.host + path + ("?" + qs if qs else ""), qs, None
                else:
                    url = self.host + path
                    body = json.dumps(params, separators=(",", ":"))
                    payload = body
                    headers["Content-Type"] = "application/json"
                if signed:
                    if not self.ready():
                        raise ExecError("API key / secret missing")
                    headers.update({"X-BAPI-API-KEY": self.key, "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": RECV,
                                    "X-BAPI-SIGN": self._sign(ts, payload)})
                with self.lock:
                    r = self.s.request(method, url, data=body, headers=headers, timeout=15)
                j = r.json()
                rc = j.get("retCode")
                if rc == 10002 and attempt < retries:
                    self.sync_time()
                    continue
                if rc not in (0, None):
                    raise ExecError(f"{path} retCode={rc} {j.get('retMsg')}")
                return j.get("result", {})
            except ExecError:
                raise
            except Exception as e:
                err = e
                time.sleep(1.0 * (attempt + 1))
        raise ExecError(f"{path} request failed: {err}")

    # --- helpers
    def equity(self):
        r = self.req("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        lst = r.get("list") or [{}]
        return float(lst[0].get("totalEquity") or 0), float(lst[0].get("totalAvailableBalance") or 0)

    def positions(self):
        r = self.req("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
        return {p["symbol"]: p for p in r.get("list", []) if float(p.get("size") or 0) > 0}

    def set_leverage(self, sym, lev):
        try:
            self.req("POST", "/v5/position/set-leverage", {"category": "linear", "symbol": sym,
                                                          "buyLeverage": str(lev), "sellLeverage": str(lev)})
        except ExecError as e:
            if "110043" not in str(e):  # leverage not modified
                raise

    def market(self, sym, side, qty, stop=None, reduce=False):
        p = {"category": "linear", "symbol": sym, "side": side, "orderType": "Market", "qty": qty,
             "positionIdx": 0, "reduceOnly": reduce, "orderLinkId": f"ant-{int(time.time() * 1000)}-{sym[:4]}"}
        if stop is not None:
            p.update({"stopLoss": stop, "slTriggerBy": "LastPrice", "tpslMode": "Full", "slOrderType": "Market"})
        return self.req("POST", "/v5/order/create", p)

    def set_stop(self, sym, stop):
        return self.req("POST", "/v5/position/trading-stop", {"category": "linear", "symbol": sym, "stopLoss": stop,
                                                              "slTriggerBy": "LastPrice", "tpslMode": "Full",
                                                              "positionIdx": 0})


def test_connection(kind):
    k, s = load_keys(kind)
    cli = BybitPrivate(k, s, HOSTS[kind])
    cli.sync_time()
    eq, av = cli.equity()
    pos = cli.positions()
    return {"ok": True, "equity": eq, "available": av, "positions": len(pos)}


def _fmt_qty(q, step):
    dec = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return f"{q:.{dec}f}"


def _fmt_px(px, tick):
    dec = max(0, -int(math.floor(math.log10(tick)))) if tick < 1 else 0
    return f"{round(px / tick) * tick:.{dec}f}"


# ---------------------------------------------------------------- executor (runs inside the trader process)
class Executor:
    """Mirrors the paper book: after every paper re-balance each coin's exchange position is set to
    paper weight x exchange equity x scale_mult (market orders, reduce-only when shrinking)."""

    def __init__(self, kind, trader):
        if kind not in HOSTS:
            raise ExecError("unknown executor")
        self.kind, self.trader = kind, trader
        self.cli = None
        self.keys_saved = None
        self.state = core.read_json(state_path(kind), {}) or {"log": []}
        self.eq = None
        self.last_sync = 0
        self.last_error = None
        self.connected = False
        self.ex_pos = {}

    def _client(self):
        info = keys_info(self.kind)
        if not info["present"]:
            self.cli = None
            return None
        if self.cli is None or self.keys_saved != info["saved"]:
            k, s = load_keys(self.kind)
            self.cli = BybitPrivate(k, s, HOSTS[self.kind])
            self.cli.sync_time()
            self.keys_saved = info["saved"]
        return self.cli

    def armed(self):
        return bool((core.read_json(control_path(self.kind), {}) or {}).get("armed"))

    def _note(self, msg, level="info"):
        self.state.setdefault("log", []).append([core.now_ms(), msg])
        self.state["log"] = self.state["log"][-200:]
        core.save_json(state_path(self.kind), self.state)
        getattr(log, level)(f"[{self.kind}] {msg}")

    def _cfg(self):
        return self.trader.cfg.get(self.kind, {})

    def on_event(self, ev):
        if ev["type"] != "rebalance" or not self.armed():
            return
        if self.kind == "live" and not LIVE_ENABLED:
            self._note("LIVE disabled in code - would re-balance " + ", ".join(ev.get("syms", [])))
            return
        self.sync(ev.get("weights", {}))

    def sync(self, weights):
        cli = self._client()
        if not cli:
            return
        try:
            self.eq = cli.equity()[0]
            self.ex_pos = cli.positions()
            mult = float(self._cfg().get("scale_mult", 1.0))
            lev = float(self._cfg().get("max_leverage", 8))
            g = sum(abs(w) for w in weights.values()) * mult
            k = min(1.0, lev / g) if g > 0 else 1.0
            prices = self.trader.eng.last_px
            for sym, w in weights.items():
                ins = core.instruments(self.trader.cfg).get(sym, {"qty_step": 0.001, "min_qty": 0.001})
                px = prices.get(sym)
                if not px:
                    continue
                want = w * mult * k * self.eq / px
                want = math.copysign(math.floor(abs(want) / ins["qty_step"] + 1e-9) * ins["qty_step"], want)
                if abs(want) < ins["min_qty"] or abs(want) * px < 5:
                    want = 0.0
                p = self.ex_pos.get(sym)
                cur = (float(p["size"]) if p["side"] == "Buy" else -float(p["size"])) if p else 0.0
                diff = want - cur
                if abs(diff) * px < max(5.0, 0.01 * self.eq) and want != 0 and (want > 0) == (cur > 0):
                    continue
                if abs(diff) < ins["qty_step"] / 2:
                    continue
                if cur and (cur > 0) != (diff > 0):  # reduce / close first (reduce-only)
                    red = min(abs(diff), abs(cur))
                    cli.market(sym, "Sell" if cur > 0 else "Buy", _fmt_qty(red, ins["qty_step"]), reduce=True)
                    diff += math.copysign(red, cur)
                if abs(diff) >= ins["qty_step"] / 2:
                    cli.set_leverage(sym, int(min(lev, 10)))
                    cli.market(sym, "Buy" if diff > 0 else "Sell", _fmt_qty(abs(diff), ins["qty_step"]))
                self._note(f"{sym}: position {cur:+g} -> {want:+g}")
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)
            self._note(f"re-balance failed: {e}", "error")

    def tick(self):
        if time.time() - self.last_sync < 60:
            return
        self.last_sync = time.time()
        cli = self._client()
        if not cli:
            self.connected = False
            return
        try:
            self.eq = cli.equity()[0]
            self.ex_pos = cli.positions()
            self.connected = True
            ctl = core.read_json(control_path(self.kind), {}) or {}
            if ctl.get("flatten") and (self.kind == "demo" or LIVE_ENABLED):
                for sym, pos in self.ex_pos.items():
                    cli.market(sym, "Sell" if pos["side"] == "Buy" else "Buy", pos["size"], reduce=True)
                    self._note(f"{sym}: flattened by user")
                set_control(self.kind, flatten=False, armed=False)
            elif ctl.get("flatten"):
                set_control(self.kind, flatten=False)
            elif ctl.get("sync_now") and self.armed() and (self.kind == "demo" or LIVE_ENABLED):
                eng = self.trader.eng
                eqp = eng.equity()
                self.sync({s: (p["qty"] * eng.last_px.get(s, p["avg"]) / eqp) for s, p in eng.pos.items()} |
                          {s: 0.0 for s in self.ex_pos if s not in eng.pos})
                set_control(self.kind, sync_now=False)
            self.last_error = None
        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            log.warning(f"[{self.kind}] sync failed: {e}")

    def status(self):
        return {"kind": self.kind, "armed": self.armed(), "keys": keys_info(self.kind)["present"],
                "connected": self.connected, "equity": self.eq, "positions": {
                    s: {"side": p.get("side"), "size": p.get("size"), "entry": p.get("avgPrice"),
                        "upnl": p.get("unrealisedPnl")} for s, p in self.ex_pos.items()},
                "last_error": self.last_error, "log": (self.state.get("log") or [])[-15:],
                "live_enabled": LIVE_ENABLED if self.kind == "live" else True}
