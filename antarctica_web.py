#!/usr/bin/env python3
"""
ANTARCTICA dashboard + API (tmux session "AntarcticaWeb", port 8760).

The page itself holds no data: every /api call needs a token that only a correct passcode hands out
(checked here on the Pi). Wrong codes escalate the waiting time. Backtests and optimizer runs are started
as separate processes (antarctica_backtest.py --job ...) so the website never blocks.
"""
import glob
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, request, send_from_directory, abort

import antarctica_core as core
import antarctica_exec as execmod

log = core.setup_logging("web")
app = Flask(__name__, static_folder=None)
STATIC = os.path.join(core.BASE_DIR, "static")
JOBS_DIR = os.path.join(core.DATA_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

TOKENS = {}
FAILS = {"n": 0, "until": 0}
LOCK = threading.Lock()
WAITS = [0, 0, 0, 0, 0, 60, 300, 900, 3600]


# ---------------------------------------------------------------- auth
def _cfg():
    return core.load_config()


def _check_token():
    tok = request.headers.get("X-Antarctica-Token", "")
    cfg = _cfg()
    idle = float(cfg["web"].get("autolock_minutes", 10)) * 60
    with LOCK:
        t = TOKENS.get(tok)
        if not t or time.time() - t > max(idle, 60) + 30:
            TOKENS.pop(tok, None)
            return False
        TOKENS[tok] = time.time()
    return True


@app.before_request
def guard():
    p = request.path
    if p.startswith("/api/") and p not in ("/api/unlock", "/api/vault"):
        if not _check_token():
            return jsonify({"error": "locked"}), 401


@app.after_request
def headers(r):
    r.headers["Cache-Control"] = "no-store"
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["Referrer-Policy"] = "no-referrer"
    r.headers["X-Frame-Options"] = "DENY"
    return r


@app.get("/api/vault")
def vault():
    cfg = _cfg()
    pc = str(cfg["web"].get("passcode", ""))
    return jsonify({"digits": len(pc) if pc else 6, "configured": bool(pc), "wait": max(0, FAILS["until"] - time.time()),
                    "autolock": cfg["web"].get("autolock_minutes", 10), "name": core.NAME, "version": core.VERSION})


@app.post("/api/unlock")
def unlock():
    cfg = _cfg()
    pc = str(cfg["web"].get("passcode", ""))
    if not pc:
        return jsonify({"ok": False, "error": "no passcode configured - run ./setup.sh on the Pi"}), 403
    with LOCK:
        if time.time() < FAILS["until"]:
            return jsonify({"ok": False, "wait": FAILS["until"] - time.time()}), 429
    code = str((request.get_json(silent=True) or {}).get("code", ""))
    if secrets.compare_digest(code, pc):
        tok = secrets.token_urlsafe(32)
        with LOCK:
            FAILS.update(n=0, until=0)
            TOKENS[tok] = time.time()
            for k in [k for k, v in TOKENS.items() if time.time() - v > 86400]:
                TOKENS.pop(k, None)
        log.info(f"vault unlocked from {request.headers.get('CF-Connecting-IP', request.remote_addr)}")
        return jsonify({"ok": True, "token": tok, "autolock": cfg["web"].get("autolock_minutes", 10)})
    with LOCK:
        FAILS["n"] += 1
        w = WAITS[min(FAILS["n"], len(WAITS) - 1)]
        FAILS["until"] = time.time() + w
    log.warning(f"wrong passcode ({FAILS['n']}x) from {request.headers.get('CF-Connecting-IP', request.remote_addr)}")
    time.sleep(0.6)
    return jsonify({"ok": False, "fails": FAILS["n"], "wait": w}), 403


@app.post("/api/lock")
def lock():
    with LOCK:
        TOKENS.pop(request.headers.get("X-Antarctica-Token", ""), None)
    return jsonify({"ok": True})


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:p>")
def static_files(p):
    return send_from_directory(STATIC, p)


# ---------------------------------------------------------------- live data
_live_cache = {"t": 0, "v": None}


def public_url():
    try:
        with open(os.path.join(core.LOG_DIR, "cloudflared.log"), "r", errors="ignore") as f:
            urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", f.read()[-200000:])
        return urls[-1] if urls else None
    except Exception:
        return None


def tail_log(n=200):
    try:
        with open(core.LOG_PATH, "r", errors="ignore") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 120_000))
            lines = f.read().splitlines()[-n:]
        return lines
    except Exception:
        return []


def live_bundle():
    if time.time() - _live_cache["t"] < 3 and _live_cache["v"]:
        return _live_cache["v"]
    cfg = _cfg()
    st = core.read_json(core.STATE_PATH, None)
    status = core.read_json(core.STATUS_PATH, {}) or {}
    eng = core.Engine(cfg["strategy"], cfg)
    started = core.now_ms()
    if st and st.get("engine"):
        eng.load(st["engine"])
        started = st.get("started") or started
    prices = {s: v.get("price") for s, v in (status.get("symbols") or {}).items() if v.get("price")}
    b = core.bundle(eng, cfg, started, core.now_ms(), prices=prices or None)
    b["kind"] = "live"
    b["status"] = status
    b["events"] = (core.read_json(core.EVENTS_PATH, []) or [])[-150:]
    b["config"] = {"risk_pct": cfg["account"]["risk_pct"], "heat_pct": cfg["account"]["heat_pct"],
                   "start_balance": cfg["account"]["start_balance"], "max_leverage": cfg["account"]["max_leverage"],
                   "fees": cfg["fees"], "symbols": cfg["symbols"], "timeframe": cfg["timeframe"],
                   "strategy": cfg["strategy"], "kelly": cfg["account"]["kelly_sizing"], "timezone": cfg["timezone"]}
    b["public_url"] = public_url()
    b["alive"] = bool(status.get("heartbeat") and core.now_ms() - status["heartbeat"] < 120_000)
    _live_cache.update(t=time.time(), v=b)
    return b


@app.get("/api/live")
def api_live():
    return app.response_class(json.dumps(core.clean_json(live_bundle())), mimetype="application/json")


@app.get("/api/log")
def api_log():
    return jsonify({"lines": tail_log(int(request.args.get("n", 250)))})


@app.get("/api/chart")
def api_chart():
    cfg = _cfg()
    sym = request.args.get("sym", cfg["symbols"][0])
    if sym not in cfg["symbols"]:
        abort(404)
    n = min(1500, max(50, int(request.args.get("n", 400))))
    p = core.clean_params(cfg["strategy"])
    tfm = int(cfg["timeframe"])
    store = core.Store()
    d = store.tail(sym, tfm, n + core.warmup_bars(p))
    if not len(d["ts"]):
        return jsonify({"sym": sym, "candles": []})
    ind = core.indicators(d, p)
    k = max(0, len(d["ts"]) - n)
    rows = []
    for i in range(k, len(d["ts"])):
        rows.append([int(d["ts"][i]), d["o"][i], d["h"][i], d["l"][i], d["c"][i], d["v"][i],
                     ind["ema"][i], ind["hh"][i], ind["ll"][i], ind["atr"][i]])
    status = core.read_json(core.STATUS_PATH, {}) or {}
    forming = ((status.get("symbols") or {}).get(sym) or {}).get("forming")
    st = core.read_json(core.STATE_PATH, {}) or {}
    e = st.get("engine") or {}
    trades = [t for t in e.get("trades", []) if t.get("sym") == sym][-200:]
    pos = (e.get("pos") or {}).get(sym)
    out = {"sym": sym, "tf": tfm, "candles": rows, "forming": forming, "trades": trades, "pos": pos,
           "pending": (e.get("pending") or {}).get(sym), "params": p}
    return app.response_class(json.dumps(core.clean_json(out)), mimetype="application/json")


# ---------------------------------------------------------------- jobs (backtest / optimizer) in subprocesses
def _job_paths(jid):
    return os.path.join(JOBS_DIR, f"{jid}.req.json"), os.path.join(JOBS_DIR, f"{jid}.out.json")


def start_job(kind, params):
    for f in sorted(glob.glob(os.path.join(JOBS_DIR, "*.json")), key=os.path.getmtime)[:-40]:
        try:
            os.remove(f)
        except Exception:
            pass
    jid = uuid.uuid4().hex[:12]
    req, out = _job_paths(jid)
    core.save_json(req, {"kind": kind, "params": params, "created": core.now_ms()})
    core.save_json(out, {"state": "queued", "progress": 0, "msg": "starting"})
    subprocess.Popen([sys.executable, os.path.join(core.BASE_DIR, "antarctica_backtest.py"), "--job", jid],
                     cwd=core.BASE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return jid


@app.post("/api/backtest")
def api_backtest():
    j = request.get_json(silent=True) or {}
    q = {"days": min(3650, max(7, float(j.get("days") or 365))), "risk": min(20, max(0.05, float(j.get("risk") or _cfg()["account"]["risk_pct"]))),
         "balance": min(1e9, max(1, float(j.get("balance") or 100))), "heat": min(200, max(0.1, float(j.get("heat") or _cfg()["account"]["heat_pct"]))),
         "kelly": bool(j.get("kelly", False)), "params": j.get("params")}
    return jsonify({"job": start_job("backtest", q)})


@app.post("/api/optimize")
def api_optimize():
    j = request.get_json(silent=True) or {}
    o = _cfg()["optimizer"]
    q = {"days": min(3650, max(60, float(j.get("days") or o["days"]))), "combos": min(2000, max(5, int(j.get("combos") or o["combos"]))),
         "train_frac": min(0.9, max(0.3, float(j.get("train_frac") or o["train_frac"]))),
         "risk": float(j.get("risk") or _cfg()["account"]["risk_pct"]), "heat": float(j.get("heat") or _cfg()["account"]["heat_pct"])}
    return jsonify({"job": start_job("optimize", q)})


@app.get("/api/job/<jid>")
def api_job(jid):
    if not re.fullmatch(r"[0-9a-f]{12}", jid):
        abort(404)
    out = core.read_json(_job_paths(jid)[1], None)
    if out is None:
        abort(404)
    return app.response_class(json.dumps(out), mimetype="application/json")


@app.post("/api/apply")
def api_apply():
    p = core.clean_params((request.get_json(silent=True) or {}).get("params") or {}, core.load_config()["strategy"])
    core.update_config({"strategy": p})
    log.info(f"strategy parameters applied from the website: {p}")
    return jsonify({"ok": True, "strategy": p})


# ---------------------------------------------------------------- settings / discord / keys / executors
def _mask(u):
    return (u[:34] + "…" + u[-4:]) if u and len(u) > 40 else u


@app.get("/api/settings")
def api_settings_get():
    cfg = _cfg()
    return jsonify({"account": cfg["account"], "fees": cfg["fees"], "strategy": cfg["strategy"], "symbols": cfg["symbols"],
                    "discord": {k: {"set": bool(v), "masked": _mask(v)} if isinstance(v, str) else v
                                for k, v in cfg["discord"].items()},
                    "demo": cfg["demo"], "live": cfg["live"], "web": {"autolock_minutes": cfg["web"]["autolock_minutes"]},
                    "keys": {k: execmod.keys_info(k) for k in ("demo", "live")}, "live_enabled": execmod.LIVE_ENABLED,
                    "timezone": cfg["timezone"]})


@app.post("/api/settings")
def api_settings_post():
    j = request.get_json(silent=True) or {}
    patch = {}
    if "account" in j:
        a = j["account"]
        acc = {}
        for k, lo, hi in (("risk_pct", 0.05, 20), ("heat_pct", 0.1, 200), ("max_leverage", 1, 100),
                          ("kelly_fraction", 0.05, 1), ("kelly_max_risk_pct", 0.1, 20), ("minqty_risk_mult", 1, 10)):
            if k in a:
                acc[k] = min(hi, max(lo, float(a[k])))
        if "kelly_sizing" in a:
            acc["kelly_sizing"] = bool(a["kelly_sizing"])
        if "start_balance" in a:
            acc["start_balance"] = min(1e9, max(1, float(a["start_balance"])))
        patch["account"] = acc
    for kind in ("demo", "live"):
        if kind in j:
            patch[kind] = {k: min(100 if k == "max_leverage" else 20, max(0.05, float(v))) for k, v in j[kind].items()
                           if k in ("risk_pct", "max_leverage")}
    if "discord" in j:
        d = {}
        for k, v in j["discord"].items():
            if k in _cfg()["symbols"] + ["errors"] and isinstance(v, str):
                v = v.strip()
                if v and not v.startswith("https://discord"):
                    return jsonify({"ok": False, "error": f"{k}: not a Discord webhook URL"}), 400
                d[k] = v
        patch["discord"] = d
    if "autolock_minutes" in j:
        patch["web"] = {"autolock_minutes": min(240, max(1, int(j["autolock_minutes"])))}
    core.update_config(patch)
    log.info(f"settings changed from the website: {sorted(patch)}")
    return jsonify({"ok": True})


@app.post("/api/discord/test")
def api_discord_test():
    cfg = _cfg()
    n = 0
    for k, url in cfg["discord"].items():
        if isinstance(url, str) and url.startswith("http"):
            core.discord_send(url, f"🧊 **ANTARCTICA** test message for `{k}` · {core.fmt_ts(core.now_ms(), cfg)}")
            n += 1
    return jsonify({"ok": True, "sent": n})


@app.post("/api/keys")
def api_keys():
    j = request.get_json(silent=True) or {}
    kind = j.get("kind")
    if kind not in ("demo", "live"):
        return jsonify({"ok": False, "error": "kind must be demo or live"}), 400
    execmod.save_keys(kind, str(j.get("key", "")), str(j.get("secret", "")))
    log.info(f"{kind} API keys {'saved' if j.get('key') else 'removed'} from the website")
    return jsonify({"ok": True, "info": execmod.keys_info(kind)})


@app.post("/api/exec")
def api_exec():
    j = request.get_json(silent=True) or {}
    kind, action = j.get("kind"), j.get("action")
    if kind not in ("demo", "live"):
        return jsonify({"ok": False, "error": "kind must be demo or live"}), 400
    try:
        if action == "test":
            return jsonify(execmod.test_connection(kind))
        if action == "arm":
            if kind == "live" and not execmod.LIVE_ENABLED:
                return jsonify({"ok": False, "error": "live execution is switched off in code (LIVE_ENABLED = False)"}), 403
            if not execmod.keys_info(kind)["present"]:
                return jsonify({"ok": False, "error": "save API keys first"}), 400
            execmod.set_control(kind, armed=True)
        elif action == "disarm":
            execmod.set_control(kind, armed=False)
        elif action == "flatten":
            execmod.set_control(kind, flatten=True)
        else:
            return jsonify({"ok": False, "error": "unknown action"}), 400
        log.info(f"{kind} executor: {action}")
        return jsonify({"ok": True})
    except Exception as e:
        log.warning(f"{kind} {action} failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 400


def main():
    cfg = _cfg()
    port = int(cfg["web"].get("port", 8760))
    if not cfg["web"].get("passcode"):
        log.warning("no website passcode configured - run ./setup.sh (the vault stays closed until then)")
    log.info(f"ANTARCTICA dashboard on http://0.0.0.0:{port}")
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=8, ident="antarctica")
    except ImportError:
        app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
