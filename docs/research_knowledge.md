# Research Knowledge Base — Ключевые знания из литературы

Извлечено из 12 статей + 1 книга from academic papers and books.
Прочитано ПОЛНОСТЬЮ: Ernest Chan (175 стр, все 8 глав), Triple Barrier, Order Flow, PredictionMarketBench, + 8 papers (abstracts + key sections).
Организовано по фазам проекта для прямого применения.

---

## Каталог материалов

### TIER 1 — Прямое применение (прочитано целиком)

| # | Файл | Название | Применение |
|---|-------|----------|------------|
| 1 | `papers/Triple_Barrier_Info_Bars_Crypto_DL_2025.pdf` | Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning (Gradzki et al., Financial Innovation, 2025) | Phase 1, 3, 4 |
| 2 | `papers/PredictionMarketBench_Backtesting_2026.pdf` | PredictionMarketBench: SWE-bench-Style Framework for Backtesting Trading Agents on Prediction Markets (Arora & Malpani, 2026) | Phase 5 |
| 3 | `books/Quantitative Trading...Wiley (2008).pdf` | Quantitative Trading: How to Build Your Own Algorithmic Trading Business (Ernest Chan) | Phase 2-7 |
| 4 | `papers/Order_Flow_Signal_Extraction_Matched_Filter_2026.pdf` | Optimal Signal Extraction from Order Flow: A Matched Filter Perspective (Kang, 2026) | Phase 3 (smart money) |

### TIER 2 — Полезные техники (abstract + key sections)

| # | Файл | Название | Применение |
|---|-------|----------|------------|
| 5 | `papers/Cross-Market_Alpha_Alpha191_LASSO_2026.pdf` | Cross-Market Alpha: Alpha191 Factors via Double-Selection LASSO | Phase 3 (factors) |
| 6 | `papers/GA_Strategy_Optimization_Walk-Forward_2026.pdf` | Adaptive Multi-Asset Trading Strategy via Genetic Algorithms + Walk-Forward | Phase 5 |
| 7 | `papers/Bayesian_Robust_Trading_Adversarial_Synthetic_2026.pdf` | Bayesian Robust Financial Trading with Adversarial Synthetic Data | Phase 4-5 |

### TIER 3 — Фон / архитектурные идеи

| # | Файл | Ценность |
|---|-------|----------|
| 8 | TiMi (ICLR 2026) | Мульти-агентная система для трейдинга: policy→optimization→deployment chain |
| 9 | FinAgent (NeurIPS 2025) | Маппинг AT компонентов на агентов |
| 10 | AI-Trader (2025) | LLM-агенты плохо торгуют: extreme variance, action flipping |
| 11 | AlphaForgeBench (KDD 2026) | LLM для генерации alpha factors |
| 12 | AI Algo Trading Simulation (2025) | Пример backtesting платформы |
| 13 | Quant Trading Python (2025) | Базовый пример |

---

## DEEP DIVE 1: Triple Barrier + Information-Driven Bars

### Information-Driven Bars — Полные определения

**Зачем:** Стандартные time bars (1min, 1h, 1d) не отражают реальную рыночную активность. Объём торгов НЕ распределён равномерно по времени. Info-driven bars адаптируются к рынку.

#### CUSUM Filter (лучший результат в статье)

```
S_t+ = max(0, S_{t-1}+ + r_t)    # кумулятивная сумма положительных returns
S_t- = min(0, S_{t-1}- + r_t)    # кумулятивная сумма отрицательных returns
S_t = max(|S_t+|, |S_t-|)        # абсолютный максимум

Новый bar создаётся когда S_t >= h (порог)
После создания bar: S_t+ = S_t- = 0 (сброс)
```

где r_t = return между периодами t и t-1, h = порог (типично 1-3%).

**Ключевое свойство:** CUSUM фильтрует шум и сэмплирует только при значимых движениях. Адаптируется к волатильности автоматически — больше bars в активные периоды.

**Оптимальные параметры (из sensitivity analysis):**
- CUSUM threshold: 2-2.5% — оптимум
- Ниже 1%: слишком частый сэмплинг, costs убивают прибыль
- Выше 3%: слишком редко, теряем сигналы
- Для ETH: 2% CUSUM → ~40 bars/день в среднем

#### Volume Bars

Новый bar создаётся после накопления фиксированного объёма торгов.

**Параметры из статьи:**
- ETH: 30K, 50K, 100K ETH → 40K, 24K, 12K bars за 2007 дней (20-5 bars/день)
- BTC: 5K, 10K, 20K BTC → 33K, 16K, 8K bars

**Результат:** Проиграли CUSUM. Уязвимы к wash trading (искусственно завышенный объём).

#### Dollar Bars

Новый bar после фиксированной суммы в долларах.

**Параметры:**
- ETH: $50M, $100M, $150M
- BTC: $100M, $200M, $300M

**Результат:** Худший метод. Все конфигурации убыточны. Dollar bars страдают от тех же проблем что volume bars + дополнительная нестабильность из-за изменения цены актива.

#### Range Bars

Новый bar когда |P_t - P_{t-1}| / P_{t-1} >= R (фиксированный % порог).

**Параметры:** 1%, 2%, 3%

**Результат:** Второй после CUSUM. 5 из 12 конфигураций прибыльны с Triple Barrier.

#### Сравнение (ranked by profitability):
1. **CUSUM** — consistently profitable (2% optimal)
2. **Range bars** — mixed results, some profitable
3. **Time bars** — unprofitable with next-bar, mixed with Triple Barrier
4. **Volume bars** — mostly unprofitable
5. **Dollar bars** — always unprofitable

### Triple Barrier Labeling — Полное описание

**Проблема next-bar prediction:**
- Accuracy ~50% для всех моделей (BTC 1h: max 58.8%, но unprofitable)
- Частое переключение long/short → transaction costs убивают прибыль
- 29 из 210 экспериментов прибыльны с next-bar (14%)

**Triple Barrier Method (López de Prado, 2018):**

Три барьера определяют label каждого наблюдения:

```
Upper barrier: entry_price * (1 + take_profit%)   → label = 1 (long)
Lower barrier: entry_price * (1 - stop_loss%)     → label = -1 (short)
Vertical barrier: entry_time + max_holding_period  → label = first barrier breached
```

Label определяется первым достигнутым барьером:
- Цена достигает upper → 1 (profitable long)
- Цена достигает lower → -1 (profitable short)
- Время истекло → label по направлению движения к моменту vertical barrier

**Оптимальные параметры (из sensitivity analysis):**
- Barriers: 5-6% symmetric (take-profit = stop-loss)
- Vertical barrier: 24 периода
- Для наиболее частых bars: 2.5% barriers
- Для менее частых bars: 5% barriers
- **Лучшие Sharpe ratios: (CUSUM 2-2.5%, Triple Barrier 5-6%) → Sharpe 1.9-2.0**

**Динамические barriers (volatility-adjusted):**
- Протестировано: barriers = 1σ и 2σ от exponentially weighted moving average of std
- Результат: НЕ улучшили. Для CUSUM даже ухудшили.
- Вывод: лучше включить volatility как фичу в модель, а не в labeling

**Фильтрация предсказаний:**
- Long только если P(up) > 60%
- Short только если P(up) < 40%
- Predictions в зоне 40-60% → no action
- Один открытый trade в каждый момент времени

### Результаты по моделям

**Протестировано 2700 моделей** (5 bars × 3 values × 2 labels × 6 models × 3 seeds × 5 periods)

| Модель | Triple Barrier | Next-bar | Комментарий |
|--------|---------------|----------|-------------|
| **ResNet-LSTM** | **Лучшая** | Средняя | CNN (spatial) + LSTM (temporal) — consistently good |
| XGBoost | Плохая | Лучшая accuracy (54.9%) | Высокая accuracy, но unprofitable! |
| Autoformer | Плохая | 2nd best profit | Проблемы с адаптацией encoder-decoder к Triple Barrier |
| FEDformer | Плохая | Плохая | Теряет temporal info из-за parallel processing |
| Transformer Encoder | Плохая | Плохая | Stagnating error rate, infrequent signals |
| Attention-LSTM | Плохая | Плохая | Theoretical potential unrealized |
| TSMixer | Средняя | Плохая | MLP-based, mixed results |

**Главный вывод:** ResNet-LSTM > Transformers для financial time series classification.
XGBoost: лучшая accuracy но это не коррелирует с прибыльностью!

### Лучший результат (ETH, CUSUM 2%, Triple Barrier 5%, ResNet-LSTM)

```
Annual Net Profit:    +91.6%
Profitable txns:      58.1%
Accuracy:             53.3%
Annualized Sharpe:    1.42
Max Drawdown:         25.1%
Time active:          64.1%
```

Buy-and-hold за тот же период: ETH -44%, BTC -34%.

### Extensibility to other cryptos

| Crypto | CUSUM | Triple Barrier | Annual Profit | Sharpe |
|--------|-------|---------------|---------------|--------|
| MATIC  | 2.5%  | 5.0%          | +129.0%       | 1.43   |
| LINK   | 5.0%  | 8.0%          | +103.1%       | 1.55   |

**Важно:** Optimal parameters РАЗНЫЕ для разных активов. Для более волатильных нужны бОльшие пороги.

### Feature Engineering (33 фичи)

1. EMA close: periods 5, 10, 15, 20, 50 (5 фичей)
2. Std dev close: periods 5, 10, 15, 20, 50 (5 фичей)
3. MACD (12, 26) (1 фича)
4. RSI: periods 6, 10, 14 (3 фичи)
5. Stochastic Oscillator %K, %D (period 14) (2 фичи)
6. Williams %R (period 14) (1 фича)
7. Bollinger Bands (period 5, 2 std) (3 фичи: upper, lower, width)
8. Historical returns (1 фича)
9. Chaikin Money Flow (period 21) (1 фича)
10. Money Flow Index (period 14) (1 фича)
11. Sine/cosine hour + weekday (4 фичи)
12. OHLCV (5 фичей)

**Ablation test:** Feature engineering критична для BTC (без неё: +20.4% → -24.2%). Для ETH минимальный эффект.

### Ключевые выводы для стратегии

- Стратегия работает ЛУЧШЕ в volatile markets. Equity curve растёт при high volatility (LUNA crash, Aug 2022 recovery), стагнирует при low volatility (2023).
- Нет значимой разницы между торговыми сессиями (Asian/European/US).
- ETH более профитабелен чем BTC (BTC более efficient).
- Transaction costs: 0.1% per trade (Binance). Без учёта costs многие стратегии кажутся прибыльными но реально убыточны.

---

## DEEP DIVE 2: Ernest Chan — Quantitative Trading

### Ch.2: Fishing for Ideas — Критерии выбора стратегии

**Checklist для оценки стратегии:**
1. Sharpe ratio > 1 (ideally > 2)
2. Drawdown: глубина < 2× годовой return, длительность < 1 год
3. Transaction costs included? (commission + half bid-ask spread + slippage)
4. Survivorship bias free? (включены делистированные/закрытые рынки)
5. Data-snooping bias? (чем больше параметров, тем больше риск)
6. Performance stable over time? (не только в одном периоде)
7. "Fly under radar"? (стратегия работает на масштабе, доступном retail)

**Transaction cost estimation:**
- Cost ≈ half(bid-ask spread) + commission
- Round-trip = 2× single transaction
- Для Polymarket: maker fee + taker fee + spread

**Data-snooping mitigation:**
- Минимизировать число параметров
- Варьировать параметры на ±20% — стратегия стабильна?
- Averaging over parameter sets — аллокация капитала на несколько конфигураций
- Out-of-sample testing обязателен

### Ch.3: Backtesting — Полные правила

**Platforms:** Excel, MATLAB, TradeStation, high-end

**Historical data checks:**
1. **Split/dividend adjusted?** — для акций обязательно. Для Polymarket не актуально (нет splits).
2. **Survivorship bias free?** — КРИТИЧНО. Включать resolved/closed markets, не только active.
3. **High/Low data reliable?** — intraday high/low могут быть ненадёжны. Для Polymarket: tick data надёжнее.

**Lookahead bias:**
- НЕ использовать данные из будущего в features
- Типичная ошибка: использование close price для entry, когда close известен только после закрытия
- Для Polymarket: не использовать resolution outcome в features

**Common pitfall:**
- Short sell при неограниченном downside — стратегия с огромными returns но потенциально безграничными потерями
- Prediction markets: max loss = cost of contract (bounded!), это преимущество

**Example 3.7: Mean reversion with transaction costs:**
- Khandani-Lo strategy (buy yesterday's losers, sell winners)
- Before costs: profitable
- After costs (5 bp per transaction): heavily unprofitable
- Lesson: high-frequency mean reversion may not survive transaction costs

### Ch.4: Setting Up Your Business

**Key decisions:**
- Home-based vs institutional
- Capital requirements: $50K minimum recommended (2008), now lower
- Brokerage selection: API access, low commissions, good execution
- Technology: автоматизация > manual execution

### Ch.5: Execution Systems

**Types:**
- Semi-automatic: signals generated, manual execution
- Fully automatic: signals + execution automated
- Start semi-auto, then automate when confident

**Key insight:** Execution quality matters. Slippage can destroy marginal strategies. For prediction markets: limit orders > market orders.

### Ch.6: Money and Risk Management — Kelly Formula (ПОЛНАЯ)

**Kelly Formula (single strategy):**
```
f* = μ / σ²
```
где f* = optimal fraction of capital (leverage), μ = excess mean return, σ² = variance of returns.

**Kelly Formula (portfolio of strategies):**
```
F* = C⁻¹ × M
```
где C = covariance matrix, M = vector of mean excess returns.

**Maximum compounded growth rate:**
```
g* = r + S²/2
```
где r = risk-free rate, S = Sharpe ratio. Growth rate зависит ТОЛЬКО от Sharpe ratio!

**Practical rules from Chan:**

1. **ВСЕГДА half-Kelly:** f*/2. Full Kelly оптимален теоретически, но:
   - Оценки μ и σ неточны
   - Returns не Gaussian (fat tails)
   - Одна ошибка → catastrophic loss
   - Half-Kelly: 75% growth rate при dramatically lower risk

2. **Kelly is independent of time scale:**
   - Считай на дневных или часовых returns — f* одинаковый
   - Но rebalancing frequency должна соответствовать holding period

3. **Kelly requires continuous rebalancing:**
   - Equity выросла → увеличивай позицию
   - Equity упала → УМЕНЬШАЙ позицию (built-in stop-loss!)
   - Это психологически сложно (sell low)

4. **Maximum drawdown под Kelly:**
   - Theoretical max DD under full Kelly can be catastrophic
   - Under half-Kelly: значительно лучше
   - Chan's example: SPY Kelly leverage = 2.528, half-Kelly = 1.264

5. **Multi-strategy Kelly:**
   - MATLAB code: F = inv(C) * M
   - Отрицательные f означают short/avoid strategy
   - Correlation между стратегиями УМЕНЬШАЕТ оптимальный leverage

**Application to Polymarket:**
```python
# Для каждой стратегии:
excess_return = mean(daily_returns) - risk_free_rate / 252
variance = var(daily_returns)
kelly_f = excess_return / variance
position_size = kelly_f / 2  # half-Kelly

# Max position per market = position_size * total_equity
```

### Ch.6: Risk Management (beyond Kelly)

**Stop Loss — CRITICAL INSIGHT:**
- **Momentum regime → stop loss beneficial** (exit before further decline)
- **Mean-reverting regime → stop loss HARMFUL** (exits at worst time, price will revert!)
- News-driven moves → momentum, use stop loss
- Unexplained moves → likely liquidity event, mean-reverting, DON'T stop loss
- **For Polymarket: 77% of our markets are MR → stop loss should be loose or absent for HTR strategy**

**Fat tails protection:**
- Max historical 1-period loss determines max safe leverage
- SPY example: max 1-day loss = 20.47% (Black Monday 1987)
- Half-Kelly leverage = 1.26 → NOT safe enough for Black Monday!
- **Rule: max_leverage = min(half_Kelly, max_tolerable_DD / max_historical_loss)**

**Model risk:**
- Continuously update Kelly leverage with trailing m,s (lookback ~6 months)
- As mean return → 0, Kelly leverage → 0 (gradual shutdown, NOT abrupt)
- Preferable to panicking and shutting down during drawdown

**Maximum drawdown protection:**
- Set max acceptable drawdown (e.g., 20%)
- If reached → stop trading, reassess
- Chan: "A 50% loss requires a 100% gain to recover"

**Contagion risk:**
- August 2007: quant funds cascade sell-off
- Lesson: correlation spikes in crisis. Strategies thought uncorrelated become correlated.
- For Polymarket: diversify across market categories (sports ≠ politics ≠ crypto)

**Psychological biases (Ch.6):**
- **Loss aversion**: hold losers too long, exit winners too early
- **Representativeness bias**: overweight recent, underweight long-term average → don't tweak model after one big loss
- **Despair**: shut down model during drawdown (or double down — equally bad)
- **Greed**: increase leverage after winning streak → overleveraging
- **Golden rule**: keep portfolio size under control. Start small, scale gradually.

**Practical risk limits:**
- Max position per market: Kelly-based
- Max total exposure: sum of positions < equity × max_leverage
- Daily loss limit: stop after X% loss in a day
- Sector concentration: no more than Y% in one category

### Ch.7: Special Topics

**Mean Reversion vs Momentum:**

| Property | Mean Reversion | Momentum |
|----------|---------------|----------|
| Test | ADF test (p < 0.05 → MR) | Hurst exponent (H > 0.5 → Momentum) |
| Time horizon | Short (hours-days) | Medium (days-weeks) |
| Market condition | Range-bound | Trending |
| Prediction markets | Price oscillates around "true" probability | News-driven directional moves |
| Risk | Gap/jump through mean | Trend reversal |

**ADF Test (Augmented Dickey-Fuller):**
- H0: time series has unit root (random walk, not mean-reverting)
- p < 0.05 → reject H0 → series is stationary/mean-reverting
- Apply to: spread between two related markets, deviation from "fair value"

**Hurst Exponent:**
- H < 0.5: mean reverting
- H = 0.5: random walk
- H > 0.5: trending
- Calculate using: Variance Ratio test or R/S analysis

**Regime Switching:**
- Markets alternate between MR and momentum regimes
- Detection: rolling volatility, volume patterns, macro state
- Hidden Markov Model for regime identification
- **For Polymarket: pre-event (MR) → event happening (Momentum) → post-event (MR)**

**Cointegration (for pairs/arb):**
- Two time series are cointegrated if their linear combination is stationary
- Test: Johansen test, Engle-Granger two-step (CADF)
- **Cointegration ≠ correlation!** KO/PEP: corr=0.4849, but NOT cointegrated
- GLD/GDX: >95% cointegrated (CADF t=-3.36, 5% critical=-3.34)
- **For Polymarket:** related markets (e.g., "Trump wins" vs "Republican wins") should be cointegrated
- Trade: when spread deviates from equilibrium, bet on convergence

**Spread Half-Life (Ornstein-Uhlenbeck):**
```
dz(t) = -θ(z(t) - μ)dt + dW
half-life = ln(2) / θ
```
- Fit: regress dz on z → θ = regression coefficient
- GLD/GDX half-life ≈ 10 days → optimal holding period
- **For Polymarket: compute half-life of spread between related markets → determines exit timing**

**Factor Models:**
- R = Xb + u (APT: excess returns = factor exposures × factor returns + idiosyncratic)
- PCA to identify common factors driving returns (factor exposures = eigenvectors)
- Good model R² ≈ 30-40% for 1000 stocks, 50 factors
- Factor returns have momentum → can predict next-period returns
- Residuals = idiosyncratic returns = alpha opportunity
- For Polymarket: factor = market category (sports, politics, crypto)

**Exit Strategy:**
- **Mean reversion: target price (μ) or holding period (half-life). NEVER stop-loss for MR!**
- **Momentum: latest entry signal reversal (more justified than arbitrary SL price)**
- Prediction markets: hold to resolution (binary payoff!) or exit when edge disappears
- Fixed holding period: default exit for any strategy

**Seasonal Strategies:**
- Equity: "Sell in May" mostly dead (competition eroded edge)
- Commodities: gasoline seasonal still works (real economic needs, not speculative)
- **Prediction markets: election cycles, sports seasons, crypto events — explore seasonality**

**HFT Insight:**
- High SR due to law of large numbers (more bets → lower deviation from mean)
- Backtesting inadequate for HFT (need historical order book, not just prices)
- **For Polymarket: multiple small bets across many markets → pseudo-HFT without speed requirement**

**Leverage vs Beta:**
- High leverage + low beta > low leverage + high beta (same expected return)
- Low-beta portfolio has higher Sharpe → higher compounded growth rate g = r + S²/2
- **For Polymarket: prefer higher leverage on low-volatility edge (stable markets) over large bets on volatile markets**

---

## DEEP DIVE 3: Order Flow Matched Filter

### Теоретический Framework

**Matched Filter Principle:**
> Optimal normalization must match the scaling behaviour of the signal-generating process.

Два типа трейдеров генерируют сигналы по-разному:

**1. Capacity-Constrained Traders (институционалы):**
- Торгуют фиксированный % от market cap (allocate % portfolio)
- Signal: Net_Flow_i / MarketCap_i (S_MC)
- Пример: пенсионный фонд аллоцирует 2% в актив i → flow пропорционален market cap

**2. Volume-Targeting Traders (алго-исполнители, VWAP/TWAP):**
- Торгуют фиксированную долю дневного объёма (participation rate)
- Signal: Net_Flow_i / TradingValue_i (S_TV)
- Пример: VWAP алгоритм исполняет 5% дневного объёма → flow пропорционален volume

### Формулы

```
S_MC(i,t) = NetFlow(i,t) / MarketCap(i,t)     # для capacity-constrained

S_TV(i,t) = NetFlow(i,t) / TradingValue(i,t)   # для volume-targeting
```

### Participation Rate Fallacy

**КЛЮЧЕВОЙ ИНСАЙТ:** Participation rate (flow/volume) — популярная метрика, но она ИСКАЖАЕТ сигнал от capacity-constrained трейдеров.

Математически: E[flow_i/volume_i] ≠ E[flow_i]/E[volume_i] (Jensen's inequality).

**Consequence:** Нормализация по volume создаёт ЛОЖНУЮ cross-sectional variation для институционалов. Маленькие stocks с низким volume получают завышенный participation rate, даже если flow пропорционален market cap.

### Monte Carlo результаты

- 1000 simulations, 500 stocks, 500 periods
- Matched filter: correlation 0.57-0.77 с true signal
- Mismatched: correlation 0.25-0.44 (до 1.99× хуже)
- Turnover heterogeneity усиливает разницу

### Эмпирические результаты (Korea, 2020-2024, 2.7M observations)

**Domestic Institutions (capacity-constrained):**
```
S_MC: t-stat = 9.65 (next-day return prediction)
S_TV: t-stat = 5.57
→ S_MC is matched filter, 1.73× stronger signal
```

**Foreign Investors (volume-targeting executors):**
```
S_TV: t-stat = 16.35
S_MC: t-stat = 7.33
→ S_TV is matched filter, 2.23× stronger signal
```

**Horizon Analysis (no sign reversal):**
- 1-day, 5-day, 20-day: signal persists, no reversal
- Means: genuine private information, NOT temporary price impact
- If it was just price impact, we'd see reversal at longer horizons

### Informed Executor Hypothesis

> Sophisticated foreign investors possess genuine private information but employ volume-targeting algorithms (VWAP/TWAP) for stealth execution.

Volume-scaling reflects **execution methodology**, not absence of information.

### Применение к Polymarket (Smart Money Strategy)

**Для whale tracking (strategy #6):**

```python
# 1. Определить тип трейдера:
# - Крупные трейдеры с фиксированными позициями → capacity-constrained
# - Боты с алгоритмическим исполнением → volume-targeting

# 2. Правильная нормализация:
# Для capacity-constrained (большинство whales на PM):
signal_mc = net_flow / market_liquidity  # NOT / daily_volume!

# Для volume-targeting:
signal_tv = net_flow / daily_trading_value

# 3. НЕ ДЕЛАТЬ:
# participation_rate = trade_size / daily_volume  # FALLACY!
# Это создаёт ложный сигнал для рынков с низким volume
```

**Как классифицировать трейдеров на Polymarket:**
- Polymarket Data API: /holders → top holders с размерами позиций
- Если позиция пропорциональна market liquidity → capacity-constrained
- Если trades появляются равномерно по времени (TWAP pattern) → volume-targeting
- Аномальный volume spike от одного адреса → informed trader

---

## DEEP DIVE 4: PredictionMarketBench

### Episode-Based Data Structure

Для backtesting prediction markets — структура данных:

```
episode_id/
  metadata.json     — tickers, time bounds, bankroll, fee model
  orderbook.parquet  — time-series orderbook snapshots
  trades.parquet     — historical trades
  settlement.json    — final YES/NO outcome per ticker
```

### Execution Realism

**Два режима:**
1. **Taker-only:** ордера должны match immediately (cross spread)
2. **Maker-taker:** resting limit orders join queue behind existing volume

**Fee model:**
- Per-fill fees (maker vs taker different rates)
- Settlement fees (at resolution)
- Fees can dominate marginal strategy improvements in binary contracts!

### Agent Interface

```
AgentContext API:
- query market summaries (best bid/ask)
- query full orderbook (depth-limited)
- query positions and cash/equity
- query open resting orders
- place orders (side, direction, type, size, time-in-force)
- cancel orders
```

**Decision cadence:** Fixed interval (e.g., every 5 seconds)
**Episode termination:** Settlement processed, all positions marked-to-settlement

### Baseline Results (Kalshi data)

| Agent | Strategy | Result |
|-------|----------|--------|
| RandomAgent | Random trades | Heavily negative (fees dominate) |
| LLM (gpt-4.1-nano) | Tool-calling | Inconsistent, high variance |
| Bollinger Bands | Mean reversion | Competitive in volatile episodes |

**Key insight:** Naive activity UNDERPERFORMS due to transaction costs. Fee-aware strategies required.

### Metrics

- P&L (absolute и relative)
- Max drawdown
- Sharpe ratio
- Fees paid (as % of P&L)
- Fill ratio (% orders filled)
- Slippage (execution vs signal price)

---

## СВОДНАЯ ТАБЛИЦА: Что применять по фазам

### Phase 1: Data Collection
- [ ] Собирать **tick-level trades** (каждая сделка), не только OHLC
- [ ] Построить pipeline: tick data → CUSUM bars (h=2%), volume bars, time bars
- [ ] Episode format: metadata.json + orderbook.parquet + trades.parquet + settlement.json
- [ ] Включать resolved/closed markets (survivorship bias!)

### Phase 3: Feature Engineering
- [ ] Triple Barrier labeling вместо next-bar prediction
  - Параметры старта: CUSUM 2%, barriers 5% symmetric, vertical 24 periods
  - Фильтр: trade only when P(up) > 60% or P(up) < 40%
- [ ] 33 technical features (EMAs, RSI, MACD, Bollinger, etc.)
- [ ] Matched filter normalization для whale/flow features
  - S_MC для capacity-constrained, S_TV для volume-targeting
  - НЕ participation rate (Jensen's inequality fallacy)
- [ ] ADF test для mean reversion detection
- [ ] Hurst exponent для momentum detection

### Phase 4: Modeling
- [ ] Baseline: XGBoost/LightGBM (competitive с DL на табличных данных)
- [ ] DL: ResNet-LSTM (лучшая архитектура из Triple Barrier paper)
- [ ] НЕ Transformer для первой итерации (underperforms ResNet-LSTM)
- [ ] Ensemble: voting из 3 best configs per model type
- [ ] Hyperband tuning (Keras Tuner)
- [ ] 3 seeds per config, average results

### Phase 5: Backtesting & Risk Management
- [ ] Walk-Forward validation (expanding window, quarterly test periods)
- [ ] Fee-aware evaluation (transaction costs MUST be included)
- [ ] Kelly formula: f* = μ/σ², use half-Kelly
- [ ] Sensitivity analysis: vary params ±20%, check stability
- [ ] Metrics: Annual P&L, Sharpe, Max Drawdown, % profitable, fill ratio
- [ ] Regime analysis: performance in high-vol vs low-vol periods

### Phase 7: Live Trading
- [ ] Start semi-automatic → fully automatic
- [ ] Limit orders preferred over market orders
- [ ] Max position per market = half-Kelly × equity
- [ ] Daily loss limit
- [ ] Diversify across categories (sports ≠ politics ≠ crypto)
- [ ] Continuous rebalancing per Kelly

---

## DEEP DIVE 5: Код AFML (finance_ml_afml)

Реализация алгоритмов из Lopez de Prado "Advances in Financial Machine Learning".
Путь: code/finance_ml_afml/finance_ml/`

### Triple Barrier (`labeling/barriers.py`)

- `get_barrier_labels(close, timestamps, trgt, sltp, seconds, side)` — главная функция
- `get_events()` — DataFrame событий (t1, trgt, side)
- `get_touch_idx()` — timestamps касания барьеров (multiprocessing)
- `get_labels()` — labels (-1, 0, 1) по первому barrier
- Metalabeling: `side` param для bet sizing

### Sample Weighting (`sampling/`)

- `get_num_co_events()` — concurrent events, вес = 1/co_events
- `get_sample_tw()` — средний вес за lifetime label
- `get_time_decay(uniq_weight, last)` — линейное затухание
- `seq_bootstrap()` — bootstrap с uniqueness-aware sampling

### Purged K-Fold (`model_selection/kfold.py`)

- **PurgedKFold**: embargo + purging overlapping labels
- **CPKFold**: Combinatorial, C(N,k) backtest paths

### Fractional Differentiation (`features/fraction.py`)

- `frac_diff_FFD(series, d)` — FFD fractional diff
- `get_opt_d(series)` — min d for stationarity (ADF test)

### Рецепт (Phase 3-4):

```python
from finance_ml.labeling.barriers import get_barrier_labels
from finance_ml.sampling.co_events import get_num_co_events
from finance_ml.model_selection.kfold import PurgedKFold
from finance_ml.features.fraction import frac_diff_FFD, get_opt_d
from finance_ml.stats.vol import get_vol

vol = get_vol(close, span=100, seconds=3600)
labels = get_barrier_labels(close, timestamps=cusum_samples,
    trgt=vol, sltp=[1, 1], seconds=86400, num_threads=8)
cv = PurgedKFold(n_splits=5, t1=labels['t1'], pct_embargo=0.01)
```

---

## DEEP DIVE 6: prediction-market-analysis (Jon-Becker)

Путь: code/prediction-market-analysis/`

### Indexers

- **Kalshi**: KalshiClient (httpx, cursor pagination), concurrent trades fetching (10 workers)
- **Polymarket API**: PolymarketClient (Gamma + Data API, offset pagination)
- **Polymarket Blockchain**: PolygonClient (web3.py, OrderFilled events)
  - CTF: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`, start block 33605403

### Kalshi Data Model

```
Trade: trade_id, ticker, count, yes_price (cents), no_price, taker_side
Market: ticker, event_ticker, status, result, volume, open_interest
```

### Анализы (DuckDB + SQL)

1. **Calibration**: win_rate(price) vs price — Taker/Maker/Combined
2. **Mispricing**: (actual_win - price) / price x 100
3. **Maker vs Taker**: excess return, z-stats per price level
4. **Statistical Tests**: trade size by role, YES/NO asymmetry, categories, regression, direction

### Ключевые инсайты

- **Favourite-longshot bias** подтвержден: longshots переоценены, favourites недооценены
- **Maker advantage**: makers outperform takers, larger trade sizes
- **Blockchain data**: OrderFilled дает maker/taker адреса для whale tracking

### Применение: Phase 1 (indexers), Phase 2 (calibration), Phase 3 (bias features), Phase 5 (DuckDB+parquet)
