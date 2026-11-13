# 🧊 ANTARCTICA — Aurora Ensemble, a market-neutral-leaning 12-coin long/short book for Bybit perpetuals (4h)

Paper trader + dashboard + backtester + optimizer for a Raspberry Pi, with a Bybit demo mirror and a (locked) live
executor. The paper trader and the backtester run the **same** `Engine` with the same candle protocol, so the paper
account is exactly what a backtest of the same period computes (verified, see below).

## The strategy
A book of 12 USDT perpetuals (BTC ETH SOL XRP DOGE BNB ADA LINK LTC AVAX DOT BCH), long some coins and short others.
Three **sleeves** make money for different reasons, so they rarely lose at the same time (daily correlations −0.2 … 0.25):

| Sleeve | Idea | Look-backs (all run at once) |
|---|---|---|
| **Trend** (time-series momentum) | long coins above their price L days ago, short the others, sized by 1 / volatility | 14, 21, 30, 45, 60 days |
| **Carry** (funding) | long the 3 coins with the lowest average funding rate, short the 3 with the highest — shorts collect funding | 3, 7, 14 days |
| **XS-momentum** (cross-sectional) | long the 3 strongest coins, short the 3 weakest | 7, 14, 21, 30 days |

* **Ensemble, not optimisation.** Every look-back × re-balance clock (every 3 or 7 days) × 3 staggered start times
  runs at the same time. The book is the average. This removes "lucky parameter" and "lucky timing" effects.
  In research, choosing the single best set looked like Sharpe 2.0, but its neighbours averaged 1.2. Re-balance timing
  alone moved the Sharpe between 1.0 and 1.5. The ensemble shows what is left without any lucky choice.
* **Volatility targeting.** Each set and each sleeve is scaled to 20 % annual volatility using **past** returns only.
  The three sleeves are averaged, then the whole book is multiplied by **`scale` (default 4×, your choice)**.
* **Execution.** Every 4h close: compute the target weights. At the next 4h open: re-balance with market orders (taker
  fee + slippage) — but only coins that drifted more than **2 % of equity** from their target (no-trade band). Bybit
  lot sizes and $5 minimum order value apply. Max gross leverage is 8×. There is no stop-loss per position: risk is
  controlled by volatility targeting and diversification across sleeves and coins.
* **Fees on every fill:** taker 0.055 % (maker 0.020 % is in the fee model, but this strategy only sends market orders),
  + 0.01 % slippage. **Funding** is paid and received at the real rates. The carry sleeve earns it: about +$440 net over
  3 years in the 4× backtest.

## What the backtest says (honest version)
Data: Binance USDT-perpetual 1-minute candles + funding (Jan 2020 – 12 Nov 2026) aggregated to 4h (this development
machine cannot reach Bybit; your Pi backtests on Bybit's own candles and funding). Start $1,000, all fees, funding, lot sizes.

| Scale | Last 365 days avg month / compounded / max DD | Last 3 years avg / comp. / DD | Oct 2020 – Nov 2026 avg / comp. / DD | $1,000 → (6.1 y) |
|---|---|---|---|---|
| 2.5× | +1.6 % / +1.4 % / 23 % | +3.4 % / +3.2 % / 23 % | +4.0 % / +3.5 % / 35 % | $12,750 |
| 3× | +1.8 % / +1.5 % / 27 % | +4.1 % / +3.6 % / 28 % | +4.6 % / +3.9 % / 40 % | $16,809 |
| **4× (default)** | **+2.4 % / +1.7 % / 36 %** | **+5.4 % / +4.5 % / 37 %** | **+6.3 % / +5.0 % / 50 %** | **$36,175** |
| 5× | +2.7 % / +1.6 % / 44 % | +6.5 % / +5.1 % / 44 % | +7.8 % / +5.8 % / 59 % | $64,281 |

Win rate (one position episode per coin = one trade): 40–44 % in every period. About 60 % of months are green.
Sharpe (daily, annualised): 1.3–1.4 over 3 and 6 years, but **0.6 over the last 12 months**.

**Read this before trusting any number**
* **Your targets (≥ 10 %/month, max DD < 30 %) are not reached together.** At max DD < 30 % (scale ≈ 2.5–3) the long-run
  average is about 4 %/month. At 4× it is about 6 %/month, with drawdowns of about 50 %. 10 %/month would need about 7×, with
  drawdowns of 70 %+. I did not find any strategy family that does better without over-fitting.
* **4× is above the Kelly optimum.** Kelly, estimated from the strategy's own daily returns, puts the growth-optimal scale
  at about 2.5×. Beyond that, extra leverage adds risk faster than growth (compare the compounded columns: 5× barely beats 4×).
  The dashboard shows the live Kelly estimate.
* **Weak recent stretch.** The last 12 months were the weakest (Sharpe 0.6). 2022 and 2026 were flat years for this book.
  Profit came in 2020–21 and 2023–25.
* **Not dependent on recent luck.** Leaving out the last 2 months changes the long-run average month by less than 0.1
  points. Every calendar year except 2022 and 2026 had a positive Sharpe, and those two were roughly flat.
* Past results guarantee nothing. Paper first, then demo, then (maybe) small real money.

### How it was found
1. The first version (6h breakout, kept in `research/v1_breakout/`) showed 15 %/month only with 80 %+ drawdowns and a
   spiky equity curve. You were right that it was not sustainable.
2. Strategy families were tested as whole portfolios with costs and funding, judged year by year
   (`research/v2_ensemble/fam1.py`):
   * trend, carry and cross-sectional momentum worked;
   * short-term reversal and limit-order mean reversion lost money.
3. Combining the three sleeves raised the Sharpe from about 1.1 (best single sleeve) to about 1.5 (`combo.py`, `ensemble.py`).
4. Robustness checks:
   * 180 parameter neighbours (`robust.py`) showed that the best single set was luck, so the book runs all of them.
   * The re-balance-timing check showed that timing was luck too, so every clock runs at 3 staggered phases.
5. The product engine reproduces the research numbers (Sharpe 1.36 vs 1.31, avg month 6.3 % vs 6.0 % at 4×).

### Verified
* **No look-ahead:** the data was cut off at 20 random points. 7,203 already-closed trades and 69,441 equity points were
  identical to the full run.
* **Paper trader = backtester:** the real `Trader` class ran against a mock Bybit API for 120 days of simulated time:
  7,232 polls, 1-minute data, 27 downtimes of up to 14 h, 32 crashes/restarts. It produced the same 131 trades
  (prices, fees, funding, PnL), the same cash balance (to 6 decimals) and the same open positions as the backtester.
  The test found and fixed one real bug: long downtimes left gaps in the candle history.
* **Executor:** tested against a signature-checking mock of the Bybit v5 API (sizing from demo equity, flips via
  reduce-only, flatten). The live executor sends **nothing** while `LIVE_ENABLED = False`.
* **Weights:** start-independent (identical whatever the history window starts with), so live and backtest agree exactly.

## Files
| File | Purpose |
|---|---|
| `antarctica_strategy.py` | the Aurora Ensemble: target weights from candles + funding (vectorised, no look-ahead) |
| `antarctica_core.py` | config, Bybit data + sqlite cache, the portfolio Engine, statistics, journal, Kelly, Discord |
| `antarctica_trader.py` | live paper trader — tmux session **Antarctica** |
| `antarctica_web.py` | dashboard + API on port **8760** — tmux session **AntarcticaWeb** |
| `antarctica_backtest.py` | backtester, optimizer (train/test), walk-forward — website and command line |
| `antarctica_exec.py` | Bybit demo mirror, the locked live executor, API-key storage |
| `static/index.html` | the dashboard (Glacier Vault lock screen, live, chart, backtest, optimizer, control) |
| `setup.sh` / `start.sh` / `stop.sh` | install / start the 3 tmux sessions (AntarcticaCloud = tunnel) / stop |
| `research/` | research and verification scripts behind this README |

## Setup on the Pi (Raspberry Pi OS Bookworm, 64-bit recommended)
```bash
cd ~
git clone https://github.com/investorgative/Antarctica.git antarctica
cd antarctica
git checkout claude/confident-ritchie-4nule3     # until this branch is merged into main
chmod +x setup.sh && ./setup.sh                   # venv, packages, tmux, cloudflared, asks for the website passcode
./start.sh                                        # starts Antarctica, AntarcticaWeb, AntarcticaCloud
```
Already installed the first (breakout) version? Run `cd ~/antarctica && ./stop.sh && git pull && ./venv/bin/pip install -r requirements.txt && ./start.sh`.
The old config is migrated automatically (passcode and webhooks are kept), and the old paper account is replaced by a new $1,000 one.

* Dashboard on your network: `http://<pi-ip>:8760` (`hostname -I` shows the IP).
* Public address: `start.sh` prints the `https://….trycloudflare.com` URL. It is also behind the **☁ PUBLIC** chip in
  the dashboard header (click to copy). This is Cloudflare's free quick tunnel: no account or domain needed, but the
  address changes whenever `AntarcticaCloud` restarts. Everything works through it.
* The first start downloads ~1,700 candles + funding per coin from Bybit (about a minute). The strategy needs ~250 days
  of history before its first weights.
* Python 3.9+ is needed (Raspberry Pi OS Bookworm ships 3.11). numpy and pandas are installed by `setup.sh`.

### Discord
Dashboard → **CONTROL** → Discord webhooks → paste → **SAVE** → **SEND TEST** (or `nano config.json` → `"discord"`).
* One field per coin (`BTCUSDT` … `BCHUSDT`): position opened, size changed, position closed (with PnL).
* `errors`: every ERROR of trader, web server, backtester and optimizer (the same message at most once per 5 min), plus a
  short re-balance summary (set `"rebalance_summary": false` in config.json to turn that off).

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
./venv/bin/python antarctica_backtest.py --days 365 --balance 1000 --scale 4
./venv/bin/python antarctica_backtest.py --days 1095 --scale 2.5
./venv/bin/python antarctica_backtest.py --days 1460 --kelly
./venv/bin/python antarctica_backtest.py --optimize --days 1460 --combos 40
./venv/bin/python antarctica_backtest.py --walkforward --days 2190 --combos 20
```

## The website
* **Glacier Vault (lock mode)** – every visit and reload starts at the lock screen: animated aurora, snowfall, a rotating
  ice crystal and hexagonal ice keys (the number keys on your keyboard work too). The correct code shatters the vault.
  The passcode (4–8 digits) is checked on the Pi. The page holds no data, and every API call needs a token that only a
  correct code hands out. From the 5th wrong code the wait escalates (1, 5, 15, 60 min). Auto-lock after 10 idle minutes;
  🔒 locks at once.
* **Header** – ticker selector with live price and 24h change, timeframe with a countdown to the next 4h re-balance,
  clock and date, PAPER / DEMO / LIVE switch, trader heartbeat, public-URL chip, and a ticker tape of all 12 coins.
  * **DEMO** shows the Bybit demo mirror bar.
  * **LIVE** shows the real-money header (built, execution locked).
* **LIVE tab**, top to bottom:
  * account overview with every statistic (trades, W/L, win rate, cumulative R, max DD in R and %, account, sizing,
    fees, days, net PnL, profit factor, annual return, Sharpe, avg win/loss, max daily DD, streaks, RtR, fees, funding,
    position size, gross leverage, time in position, Kelly…) plus a $/R equity curve with monthly bars and per-coin results
  * strategy status: the book, i.e. per coin the target weight split into trend / carry / xs-momentum, current holding,
    drift and funding rate, plus gross leverage
  * open positions
  * event feed
  * live chart with trades and the position weight over time
  * journal (year → month → week → day, click to drill down; default: weekly, last 4 weeks)
  * trade log
  * log
* **CHART** – full-screen chart for any coin.
* **BACKTEST** – period, balance, scale, band, Kelly. The result is shown in the same interface as the paper account.
* **OPTIMIZER** – random search over look-back sets, re-balance clocks and sleeve weights. It is ranked on the train part,
  with the test part next to it. Every row can be backtested or applied.
* **CONTROL** – scale / band / leverage / Kelly, Discord webhooks, Bybit **demo** API (save, test, arm, disarm, sync now,
  flatten) and Bybit **live** API (save, test; arming is locked).

Definitions: a **trade** is one position episode of one coin (from leaving 0 until back to 0 or a side flip). Size
changes in between count as adjustments. **1 R = 1 % of equity** when the trade started. "Avg SL size" shows "none",
because positions have no stop-loss.

### Kelly criterion
Every result shows the Kelly-optimal scale estimated from the account's daily returns (about 2.5× historically). With
**Kelly sizing on**, the book uses `kelly_fraction` × the Kelly scale. The estimate uses only the past 365 days of
returns, starts after 120 days, and is capped at `kelly_max_scale`.

### Demo and live trading
* **Demo** – create an API key inside Bybit *Demo Trading* (Contract: Orders + Positions, no withdrawals), paste it under
  CONTROL, press TEST CONNECTION, then ARM. After every paper re-balance each coin's demo position is set to
  paper weight × demo equity × `scale_mult` (reduce-only orders when shrinking). SYNC NOW aligns the demo account at once.
* **Live** – header, key storage, connection test and the execution path are built. Orders are blocked in code
  (`LIVE_ENABLED = False` in `antarctica_exec.py`). The live default is `scale_mult` 0.5 (= 2× instead of 4×). Use a key
  **without withdrawal permission**, IP-restricted to your Pi.
* Keys are stored in `data/secrets.json` (chmod 600). They are never sent back to the browser and never put in git.

### Account size
The book holds ~12 positions long and short. With $1,000 at 4× (~2.3× gross on average), positions are about $50–$600.
Bybit's minimums (e.g. 0.001 BTC, $5 order value) are fine at that size. Below ~$500, small target weights round to 0
and the book becomes lumpy.

## Settings (`config.json`)
| Section | Settings (default) |
|---|---|
| `strategy` | `lt`, `lc`, `lx` (look-back sets), `hs` ([18, 42] re-balance clocks), `phases` (3), `n_xs` (3), `w_ts` / `w_carry` / `w_xs` (1/1/1), `vol_days` (30), `sleeve_vol` (0.20), `sleeve_vol_days` (60), `lev_cap` (8), **`scale` (4)** |
| `account` | `start_balance` (1000), `max_leverage` (8), `band_pct` (2), `min_notional` (5), `kelly_sizing` (false), `kelly_fraction` (0.5), `kelly_max_scale` (6) |
| `fees` | `maker` (0.0002), `taker` (0.00055), `slippage` (0.0001), `include_funding` (true) |
| other | `demo` / `live` (`scale_mult`, `max_leverage`), `timezone` (Europe/Berlin), `web.port` (8760) |

Chart library: TradingView Lightweight Charts 4.1.3 (Apache-2.0), bundled in `static/`.
