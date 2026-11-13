# 🧊 ANTARCTICA — Aurora Breakout, a 12-coin trend portfolio for Bybit perpetuals (6h)

Paper trader + dashboard + backtester + optimizer for a Raspberry Pi, with a Bybit demo mirror and a (locked) live
executor. The paper trader and the backtester run the **same** `Engine` class with the same two-phase candle protocol,
so a paper trade and a backtest trade come from the same code.

## The strategy in five lines
Runs on 12 liquid USDT perpetuals at once: BTC ETH SOL XRP DOGE BNB ADA LINK LTC AVAX DOT BCH. Every rule applies to every coin.
1. **Entry** – a **6h candle closes** above the highest high (below the lowest low) of the previous **15 candles**. That candle must be an
   impulse (range ≥ **1.5 × ATR(20)**) and agree with the trend (close above / below **EMA 200**). Market order at the **next candle's open**.
2. **Stop** – entry ∓ **2 × ATR**. Size = risk % of the balance ÷ stop distance (Bybit lot sizes, max 10× leverage).
3. **Exit** – a chandelier trail at the best price since entry ∓ **8 × ATR** (moved only at candle closes) and a **time stop after 30 candles** (7.5 days).
4. **Portfolio heat** – all coins share one balance. A new position opens only while total open risk stays ≤ **12 %**,
   because the coins are strongly correlated and tend to break out together.
5. Fees on every fill: Bybit **taker 0.055 %** (market entries, stops, time exits), **maker 0.020 %** in the fee model
   (this strategy sends no limit orders), + 0.01 % slippage, + real funding payments.

## What the backtest says (honest version)
Research data: Binance USDT-perpetual 1-minute candles (Jan 2020 – 12 Nov 2026) aggregated to 6h, funding from Binance.
This development machine cannot reach Bybit's API. Your Pi backtests on Bybit's own candles.
$100 start, all fees, slippage, funding, lot sizes. **Default = 2 % risk per trade, 12 % heat cap.**

| Period | Trades | Win rate | **Avg month** | Compounded / month | Green months | Max DD | $100 → |
|---|---|---|---|---|---|---|---|
| last 365 days | 277 | 36.5 % | **+5.4 %** | +1.5 % | 7 / 13 | 58 % | $120 |
| last 1095 days | 843 | 38.2 % | **+10.0 %** | +6.5 % | 20 / 37 | 61 % | $959 |
| Apr 2020 – Nov 2026 | 1,688 | 38.5 % | **+11.1 %** | +7.3 % | 40 / 80 | 68 % | $26,462 |

**Your targets (≥ 15 %/month on average, win rate > 35 %).** The win rate target is met: 36–39 % in every period.
The 15 %/month average needs more risk. This is the price, from the same backtester:

| Risk / heat | Avg month 3 y | Avg month 6.6 y | Max DD 6.6 y | Last 365 days (compounded) |
|---|---|---|---|---|
| 1 % / 6 % | +5.2 % | +5.4 % | 42 % | +1.7 %/m |
| **2 % / 12 % (default)** | +10.0 % | +11.1 % | 68 % | +1.5 %/m |
| 3 % / 18 % | **+15.5 %** | **+15.7 %** | 84 % | −1.3 %/m |
| 4 % / 24 % | +19.2 % | +19.2 % | 94 % | −5.7 %/m |

**Read this before trusting any number above**
* "Avg month" is the plain average of monthly % returns (what you asked for). Real account growth is the
  *compounded* column. It peaks at about 8 %/month around 3 % risk. Above that, bigger risk only means bigger
  swings, not more money.
* The edge is real but modest. Across all 12 coins the monthly Sharpe ratio was about 0.4 in 2020–23 and
  about 0.2–0.3 in 2024–26. Half of all months are red.
* **The last 12 months were the weakest period** for crypto trend-following in the whole sample (+5.4 % avg month,
  but only +1.5 %/month compounded at the default).
* Drawdowns of 40–70 % at the default are normal for this system. Losing streaks are long, because most of the profit
  comes from the 38 % of trades that run.
* Past results guarantee nothing. Paper first, then demo, then (maybe) small real money.

### How the strategy was found (and why it is not curve-fitted)
* ~25,000 backtests of breakout, pullback, mean-reversion and partial-take-profit variants on 15m – 12h candles.
* **Mean reversion with maker limit orders (high win rate) failed completely.** With realistic fills (price must trade
  *through* the limit), all 288 variants lost money in both halves of the history.
* **Out-of-sample across coins.** Parameters chosen only on BTC/ETH/SOL were then run unchanged on 9 coins the search had
  never seen: 6.2 R/month, 36.7 % win rate, all 9 coins profitable.
* **Out-of-sample in time.** Choosing on 2020–23 and testing on 2024–26: 99 % of 3,000 random parameter sets of this
  family were profitable in the unseen years.
* **Plateau, not a peak.** All 324 neighbouring parameter combinations (timeframe 4/6/8h, channel 8/10/15, impulse
  1/1.25/1.5, stop 2/2.5, trail 6/8/10, time stop 20/30) were profitable. They averaged 8.5 R/month in 2020–23 and
  5.1 R/month in 2024–26. The default sits in the middle of that plateau. Picking "the best" line by in-sample Sharpe
  **lost money in walk-forward**. The optimizer therefore shows train and test side by side, and the README recommends the middle of a stable region.
* Filters that looked good in 2020–23 but did not help in 2024–26 were dropped: BTC regime, volatility regime, ADX, EMA slope.

### Verified
* **No look-ahead:** the data was cut off at 25 random points. All 7,018 trades already closed at each cut were
  identical to the full run.
* **Paper trader = backtester:** the real `Trader` class ran against a mock Bybit API for 120 days of simulated time:
  7,301 polls, forming 6h candles built from 1-minute data, 24 downtimes of up to 14 h, 27 crashes/restarts.
  It produced the same 100 trades (prices, sizes, fees, funding, exit reasons) and the same final balance
  (to 6 decimals) as the backtester.
* **Executor:** tested against a signature-checking mock of the Bybit v5 API (wallet, leverage, market entry with stop,
  trail, reduce-only close, flatten). The live executor sends **nothing** while `LIVE_ENABLED = False`.
* The scripts are in `research/` (they point at the development machine's data folders).

## Files
| File | Purpose |
|---|---|
| `antarctica_core.py` | config, Bybit data + sqlite cache, indicators, the portfolio Engine, statistics, journal, Kelly, Discord |
| `antarctica_trader.py` | live paper trader — tmux session **Antarctica** |
| `antarctica_web.py` | dashboard + API on port **8760** — tmux session **AntarcticaWeb** |
| `antarctica_backtest.py` | backtester, optimizer, walk-forward (website and command line) |
| `antarctica_exec.py` | Bybit demo executor, the locked live executor, API-key storage |
| `static/index.html` | the dashboard (Glacier Vault lock screen, live, chart, backtest, optimizer, control) |
| `setup.sh` / `start.sh` / `stop.sh` | install / start the 3 tmux sessions (AntarcticaCloud = tunnel) / stop |
| `research/` | the research and verification scripts behind this README |

## Setup on the Pi (Raspberry Pi OS Bookworm, 64-bit recommended)
```bash
cd ~
git clone https://github.com/investorgative/Antarctica.git antarctica
cd antarctica
git checkout claude/confident-ritchie-4nule3     # until this branch is merged into main
chmod +x setup.sh && ./setup.sh                   # venv, packages, tmux, cloudflared, asks for the website passcode
./start.sh                                        # starts Antarctica, AntarcticaWeb, AntarcticaCloud
```
* Dashboard on your network: `http://<pi-ip>:8760` (find the IP with `hostname -I`).
* Public address: `start.sh` prints the `https://….trycloudflare.com` URL. It is also behind the **☁ PUBLIC** chip in
  the dashboard header (click to copy). This is Cloudflare's free quick tunnel: no account and no domain needed, but the
  address changes whenever `AntarcticaCloud` restarts. Everything (vault, live, backtest, optimizer, control) works through it.
* The first start downloads ~1,100 candles per coin from Bybit (a few seconds). The first website backtest downloads the
  requested period (a 3-year backtest of 12 coins takes about 1–2 minutes on a Pi 4).

### Discord
Dashboard → **CONTROL** → Discord webhooks → paste → **SAVE** → **SEND TEST** (or `nano config.json` → `"discord"`).
* One field per coin (`BTCUSDT` … `BCHUSDT`): every open, trail, close and skipped signal. You can paste the same URL into several fields.
* `errors`: every ERROR from trader, web server, backtester and optimizer (the same message at most once per 5 min).

To create a webhook in Discord: Server settings → Integrations → Webhooks → New webhook → Copy webhook URL.

### Daily use
```bash
tmux attach -t Antarctica        # paper trader      (leave with Ctrl+B, then D)
tmux attach -t AntarcticaWeb     # web server
tmux attach -t AntarcticaCloud   # cloudflare tunnel (shows the public URL)
./stop.sh && ./start.sh          # restart everything
git pull && ./stop.sh && ./start.sh                                          # update
./stop.sh && ./venv/bin/python antarctica_trader.py --reset && ./start.sh    # wipe the paper account
```
Autostart after a reboot: `crontab -e`, then add `@reboot sleep 30 && /home/<user>/antarctica/start.sh`.

### Command line
```bash
./venv/bin/python antarctica_backtest.py --days 365 --risk 2 --balance 100
./venv/bin/python antarctica_backtest.py --days 1095 --risk 3 --heat 18
./venv/bin/python antarctica_backtest.py --days 1095 --kelly
./venv/bin/python antarctica_backtest.py --optimize --days 1095 --combos 150
./venv/bin/python antarctica_backtest.py --walkforward --days 1825 --combos 40
```

## The website
* **Glacier Vault (lock mode)** – every visit and reload starts at the lock screen: animated aurora, snowfall, a rotating
  ice crystal and hexagonal ice keys (the number keys on your keyboard work too). The correct code shatters the vault.
  The passcode (4–8 digits) is checked on the Pi. The page itself holds no data, and every API call needs a token that
  only a correct code hands out. From the 5th wrong code the wait escalates (1, 5, 15, 60 min). Auto-lock after 10 idle
  minutes (changeable in CONTROL); the 🔒 button locks it at once. To change the code: `nano config.json` →
  `"web": {"passcode": "…"}` → restart.
* **Header** – ticker selector with live price and 24h change, timeframe with a countdown to the next 6h close, clock and
  date, PAPER / DEMO / LIVE switch, trader heartbeat, public-URL chip, and a ticker tape of all 12 coins.
  * **DEMO** shows the Bybit demo mirror bar.
  * **LIVE** shows the real-money header (built, execution locked).
* **LIVE tab**, top to bottom:
  * account overview with every statistic you listed (trades, W/L, win rate, cumulative R, max DD in R and %, account,
    risk, fees, days, net PnL, profit factor, annual return, avg win/loss, max daily DD, streaks, RtR, fees total/avg,
    order size, leverage, time in position, SL size, Kelly…) plus a $/R equity curve with monthly bars and a per-coin breakdown
  * strategy status per coin: which conditions are met — breakout level and distance, impulse, trend — plus portfolio heat
  * open positions
  * event feed
  * live chart with EMA, channel high/low, entries/exits and the stop path
  * journal (year → month → week → day, click to drill down; default: weekly, last 4 weeks)
  * trade log
  * log
* **CHART** – full-screen chart for any coin.
* **BACKTEST** – period, balance, risk, heat, Kelly. The result is shown in the same interface as the paper account.
* **OPTIMIZER** – random search, ranked on the train part, with the test part next to it. Every row can be backtested or
  applied to the paper trader.
* **CONTROL** – risk / heat / leverage / Kelly, Discord webhooks, Bybit **demo** API (save, test, arm, disarm, flatten)
  and Bybit **live** API (save, test; arming is locked).

### Kelly criterion
Every result shows full and half Kelly computed from its trade list (about 11 % / 6 % for the full history, but that
assumes independent trades, which correlated coins are not). With **Kelly sizing on**, each trade risks
`kelly_fraction` × Kelly computed **only from trades already closed** (starts after 40 trades, capped at `kelly_max_risk_pct` = 4 %).
It is an option, not a recommendation: the 12 coins are correlated, so full Kelly would be far too aggressive.

### Demo and live trading
* **Demo** – create an API key inside Bybit *Demo Trading* (Contract: Orders + Positions, no withdrawals), paste it under
  CONTROL, press TEST CONNECTION, then ARM. Every paper trade is mirrored:
  * market entry with the stop-loss on the position, the stop moved on every trail, a reduce-only close on exit;
  * sized with the demo account's own equity (`demo.risk_pct`);
  * if Bybit rejects the stop, the position is closed immediately.
* **Live** – header, key storage, connection test and the execution path are built. Orders are blocked in code
  (`LIVE_ENABLED = False` in `antarctica_exec.py`). Use a key **without withdrawal permission**, IP-restricted to your Pi.
* Keys are stored in `data/secrets.json` (chmod 600). They are never sent back to the browser and never put in git.

### Small accounts ($100) and Bybit minimum order sizes
At $100 and 2 % risk ($2), some coins' minimum order would risk more than intended (e.g. 0.001 BTC). The rule:
* round **up** to the exchange minimum if that risks at most `minqty_risk_mult` (2×) the intended amount;
* otherwise skip the signal and log why.

The backtester applies the same rule, so its $100 results include this effect.

## Settings (`config.json`, created on first start)
| Section | Settings (default) |
|---|---|
| `account` | `risk_pct` (2), `heat_pct` (12), `start_balance` (100), `max_leverage` (10), `minqty_risk_mult` (2), `kelly_sizing` (false), `kelly_fraction` (0.5), `kelly_max_risk_pct` (4) |
| `fees` | `maker` (0.0002), `taker` (0.00055), `slippage` (0.0001), `include_funding` (true) |
| `strategy` | `entry_n` (15), `impulse` (1.5), `ema_n` (200), `atr_n` (20), `stop_atr` (2), `trail_atr` (8), `max_bars` (30), `be_r` (0), `trend_filter`, `allow_long`, `allow_short` |
| other | `optimizer.space`, `symbols`, `timezone` (Europe/Berlin), `web.port` (8760) |

Chart library: TradingView Lightweight Charts 4.1.3 (Apache-2.0), bundled in `static/`.
