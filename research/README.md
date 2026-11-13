# Research & verification scripts

These are the scripts behind the numbers in the main README. They were run on a development machine with
Binance USDT-perpetual 1-minute archives (data.binance.vision) in `/home/user/data` - adjust the paths to rerun them.

* `build_data.py` - zip archives -> numpy files; `rlib.py` / `fam.py` - fast numba research simulator + strategy family
* `explore1-3.py`, `search_wf.py`, `psearch.py`, `ysearch.py` - random searches (R, Sharpe, per-year objectives)
* `oos_coins.py` - parameters chosen on BTC/ETH/SOL, tested on 9 unseen coins
* `holdout.py` - choose on 2020-23, test on 2024-26; `wf_analyze.py`, `ys_an.py` - walk-forward of the selection methods
* `neigh.py` - 324-combination neighbourhood (plateau) check; `filters.py` - regime filters
* `mr.py`, `mr_search.py` - limit-order mean reversion (failed: 0 of 288 variants profitable)
* `portfolio.py`, `final_port.py` - portfolio compounding, heat caps, risk levels
* `tests/lookahead.py` - truncation test (no look-ahead); `tests/live_vs_bt.py` - real Trader vs backtester with a mock
  Bybit API, downtime and restarts; `tests/exec_mock.py` - executor vs a signature-checking Bybit mock; `tests/ui.js` - Playwright UI run
