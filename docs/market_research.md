# Polymarket-Specific Research — Академические статьи, закономерности, стратегии

Собрано из 6+ академических статей, on-chain аналитики, и индустриальных отчётов (2024-2026).

---

## 1. Точность и калибровка Polymarket

### Официальная статистика (polymarket.com/accuracy)

| Горизонт | Accuracy |
|----------|----------|
| 4 часа до resolution | 97.0% |
| 12 часов | 96.6% |
| 1 день | 96.1% |
| 1 неделя | 94.4% |
| 1 месяц | 91.2% |

- **Brier Score**: 0.0834 (overall), **0.0247** для рынков с ликвидностью >$1M
- 75% рынков резолвятся как NO, 25% как YES
- Калибровка хорошая: предсказанные вероятности близки к реализованным

### Reichenbach & Walther (SSRN, Dec 2025) — 124M trades, $48B volume

- Цены Polymarket closely track realized probabilities
- Слегка outperform букмекерские коэффициенты
- **Нет general longshot bias** на уровне рынка (в отличие от Kalshi!)
- Есть тенденция overtrade "Yes" и default option
- Только **30% трейдеров** зарабатывают (доля снижается со временем)
- Но среди прибыльных: profits persistent over time (skill, not luck)
- **Вывод**: skilled traders могут exploiting biases менее опытных участников

**Для стратегии**: Polymarket более efficient чем Kalshi. Favourite-longshot bias слабее. Нужны более тонкие подходы.

---

## 2. Анатомия Polymarket (arxiv 2603.03136, Mar 2026)

Транзакционный анализ on-chain данных за Jan-Nov 2024 (президентские выборы).

### Volume Decomposition

Три метрики вместо наивного on-chain подсчёта:
- **Exchange-Equivalent Volume (V^E)**: вторичный рынок без double-counting
- **Net Inflow (F)**: свежий капитал (mint - burn)
- **Gross Market Activity (V^G)**: V^E + |F|

Пиковые объёмы (Октябрь 2024):
- Trump: $391M exchange volume, $176M net inflow
- Harris: $192M exchange volume, $106M net inflow

### Поведение трейдеров

- **40%** трейдеров торговали ТОЛЬКО Trump market
- **30.7%** торговали ТОЛЬКО Trump YES (directional bettors)
- Только **0.7%** торговали >2 кандидатских рынка
- **19.4%** торговали все 4 token markets (market-makers/hedgers)
- **Вывод**: Высокая специализация. Большинство — directional bettors, не арбитражёры.

### Kyle's Lambda (Price Impact)

```
Ранние месяцы: λ ≈ 0.518  ($1M сдвигает цену на ~13 п.п.)
Сентябрь:      λ ≈ 0.04
Октябрь:       λ ≈ 0.01   ($1M сдвигает цену на ~0.25 п.п.)
```

Каждые $1M daily volume снижают λ на 0.073 (p<0.01).

**Для стратегии**: В новых/молодых рынках price impact огромный — можно двигать цену маленькими ордерами. В зрелых рынках нужны бОльшие позиции.

### Arbitrage Deviations (YES + NO != $1.00)

- Ранний период: отклонения до ±$0.05
- Зрелый рынок: сужение до ±$0.01-0.02
- Trump market: persistent positive deviation (оба контракта overbought)

### Три ключевых эпизода

1. **Biden Withdrawal (Jul 21)**: Мгновенный surge объёмов Harris market
2. **September Debate**: Trump 53% → 50% за часы, корреляция рынков упала с 0.6 до 0.17
3. **October Whale ($30M)**: Французский трейдер, daily inflow $0.1M → $5M. Рынок абсорбировал через объём ($391M/мес)

**Для стратегии**: Мониторить net inflow как leading indicator. Whale trades создают кратковременные dislocations.

---

## 3. Арбитраж на Polymarket (arxiv 2508.03474, AFT 2025)

Dataset: 17,218 conditions, 10,237 markets, 86M on-chain bids. Период: Apr 2024 - Apr 2025.

### Два типа арбитража

**1. Market Rebalancing Arbitrage (внутри одного рынка)**:
- Когда сумма YES prices всех outcomes != $1.00
- Long: сумма < $1 (купить все outcomes)
- Short: сумма > $1 (продать все outcomes)
- 7,051 conditions с хотя бы 1 возможностью
- 662 из 1,578 NegRisk markets содержали арбитраж (~100 opportunities per market)

**2. Combinatorial Arbitrage (между рынками)**:
- Эксплуатация ценовых несоответствий между зависимыми рынками
- 13 зависимых пар найдено во время выборов 2024
- Pair 4: 6,630 opportunities

### Прибыли ($39.6M total)

| Стратегия | Прибыль |
|-----------|---------|
| Single condition buying (YES < $1) | $5.9M |
| Single condition selling (YES > $1) | $4.7M |
| Multi-market buying YES | $11.1M |
| Multi-market buying NO | **$17.3M** |
| Cross-market arbitrage | $95K |

**Top арбитражёр**: $2.01M через 4,049 транзакций.
**Крупнейшая single trade**: $58,983 прибыли (купил YES + NO за <$0.02 каждый).

### Median profit margin: ~$0.60 per dollar invested

**Для стратегии #2 (Stat Arb)**:
- Мониторить суммы prices в NegRisk markets
- Multi-market NO buying — самый прибыльный тип ($17.3M)
- Политические рынки: больше всего opportunities
- Спорт: consistent opportunities, но underexploited

---

## 4. Fees и микроструктура

### Fee Formula

```python
fee(p) = p * (1 - p) * fee_rate_bps / 10000
```

Для 15-min crypto markets: `fee_rate_bps = 1000` (10%).

| Price (p) | Effective Fee | Breakeven Edge Required |
|-----------|--------------|------------------------|
| 0.05 | 0.30% | 0.31% |
| 0.10 | 0.56% | 0.63% |
| 0.20 | 1.00% | 1.25% |
| **0.50** | **1.56%** | **3.13%** |
| 0.80 | 1.00% | 5.00% |
| 0.90 | 0.56% | 5.63% |
| 0.95 | 0.30% | 5.94% |

**Критический инсайт**: При p=0.50 нужен edge >3.1%. При p=0.90 — >5.6% (хотя абсолютная fee ниже, upside сжат до $0.10/share).

### Total Cost

```
c_total = fee(p) + half_spread + slippage(size, depth)
```

- **Half-spread**: 1-3 центов (liquid), 5-10 центов (thin)
- **Slippage**: ~0 для $10, 2-5% для $500 на thin markets
- **Maker rebates**: post-only orders с Jan 2026, ex-post pool-based (не guaranteed)

### Влияние fees на стратегии

- Fees параболические: максимум при p=0.50, минимум у краёв
- **Contrarian стратегия (#8) выгодна по fees**: торгуем у краёв (low p или high p) где fees минимальны
- **Market making (#5)**: maker rebates компенсируют часть fees
- Wash trading снизился с 25% до 5% после введения taker fees

---

## 5. Шесть моделей заработка на Polymarket

Из анализа 95M on-chain транзакций (ChainCatcher/MEXC, 2025).

### Model 1: Information Arbitrage
- Theo (Fredi9999): $85M profit, commissioned custom polling
- Barrier: очень высокий (оригинальный research + большой капитал)

### Model 2: Cross-Platform Arbitrage
- $40M extracted за год
- Top 3 wallets: $4.2M combined
- Пример: BTC $95K prediction at $0.45 (PM) vs $0.48 (Kalshi) = 7.5% return/hour
- **Риск**: разные определения events на разных платформах!

### Model 3: High-Probability Bonds
- 90%+ ордеров >$10K на ценах >$0.95
- Пример: Fed rate-cut at $0.95 → 5.2% return за 72 часа
- **Риск**: Black swan events уничтожают десятки успешных trades

### Model 4: Liquidity Provision (Market Making)
- @defiance_cr: $10K start → $200-800/day
- Returns 80-200% annualized в новых рынках
- После выборов: rewards снизились, конкуренция выросла

### Model 5: Domain Specialization
- HyperLiquid0xb (sports): $1.4M+ profits, single trade $755K
- Axios (mention markets): 96% win rate, 10-30 trades/year
- **Лучший подход для retail**: глубокая экспертиза в узкой нише

### Model 6: Speed Trading
- 10,200+ speed trades за 2024-2025, $4.2M profits
- Окна сужаются с минут до секунд
- Не для retail

### Статистика успеха

- Только **0.51%** кошельков достигают $1,000+ прибыли
- Только **1.74%** кошельков достигают $50,000+ volume
- Оптимально: 6-10 позиций, 5-10% risk per trade, 20-40% cash reserve

---

## 6. Dual-Loop Architecture для бота

Паттерн для интеграции WebSocket (async) с synchronous trading pipeline.

### BookCache (thread-safe bridge)

```python
class BookCache:
    def __init__(self):
        self._books: dict[str, BookSnapshot] = {}
        self._lock = threading.Lock()

    def update(self, token_id: str, snapshot: BookSnapshot):
        with self._lock:
            self._books[token_id] = snapshot

    def is_fresh(self, token_id: str, max_age_sec: float = 30) -> bool:
        snap = self.get(token_id)
        if not snap:
            return False
        return (time.time() - snap.last_update) < max_age_sec
```

### Graceful Degradation

| WSS State | Cost Model | Quality |
|-----------|-----------|---------|
| Fresh (<30s) | Real spread + depth + slippage | Best |
| Stale (>30s) | Last known book + warning | Good |
| Down | Estimated costs (2% flat) | Baseline |

### Signal Scoring с учётом fees

```python
def score_signal(signal, book_cache):
    snap = book_cache.get(token_id)
    if snap and book_cache.is_fresh(token_id):
        p = snap.best_ask
        c_total = taker_fee(p) + snap.spread/2 + slippage
    else:
        c_total = 0.02  # conservative fallback

    q = compute_posterior(p, signal)
    ev_net = q - p - c_total
    return ev_net
```

---

## 7. Закономерности и аномалии — Сводка для стратегий

### Подтверждённые закономерности

1. **YES bias**: трейдеры overtrade YES option (Reichenbach 2025)
2. **Directional concentration**: 40% трейдеров в одном рынке, 30% в одном direction
3. **Liquidity = accuracy**: Brier 0.087 (low liq) vs 0.025 (high liq) — 3.5x разница
4. **Price impact falls with volume**: λ drops from 0.518 to 0.01 по мере роста рынка
5. **NegRisk markets have persistent arbitrage**: 662/1578 markets, ~100 opportunities each
6. **NO buying dominates arb profits**: $17.3M из $39.6M total (43%)
7. **Maker advantage**: makers outperform takers (подтверждено на Kalshi)
8. **Net inflow predicts direction**: surge в net flow = leading indicator

### Аномалии для эксплуатации

| Аномалия | Стратегия | Fees Impact |
|----------|-----------|-------------|
| YES + NO != $1.00 в NegRisk | Stat Arb (#2) | Low (торгуем у краёв) |
| YES bias (overtrade YES) | Contrarian (#8) — sell YES, buy NO | Low |
| New market high λ | Early entry + limit orders (#5) | Varies |
| Cross-platform price diff | Cross-platform Arb (#3) | Medium |
| High-prob underpricing ($0.95+) | Bond strategy | Very low |
| Domain-specific edge | Specialization (#5, #1) | Varies |

### Что НЕ работает

- **General longshot bias отсутствует** на Polymarket (есть на Kalshi)
- **Speed trading**: окна <секунды, не для retail
- **Naive copy trading**: malicious GitHub bots (Dec 2025), слишком много followers = slippage
- **Wash trading**: снизился с 25% до 5% после taker fees — уже не exploitable

---

## 8. Применение к фазам проекта

### Phase 1: Data Collection
- [ ] On-chain данные: OrderFilled events с Polygon (CTF + NegRisk exchanges)
- [ ] Volume decomposition: отделять exchange volume от mint/burn
- [ ] Net inflow как отдельный feature
- [ ] NegRisk market detection: sum(YES prices) для arbitrage monitoring

### Phase 2: EDA
- [ ] Calibration analysis (win_rate vs price) — повторить Reichenbach
- [ ] Brier score по ликвидности
- [ ] YES bias quantification
- [ ] Kyle's lambda estimation по рынкам

### Phase 3: Features
- [ ] Net inflow (leading indicator)
- [ ] Price impact (lambda) как feature
- [ ] NegRisk sum deviation
- [ ] Cross-market correlation
- [ ] Time-to-resolution features

### Phase 5: Backtesting
- [ ] Fee model: `fee(p) = p*(1-p)*rate` с учётом rebates
- [ ] Breakeven edge calculation per price level
- [ ] Slippage model зависящая от depth

### Phase 7: Architecture
- [ ] Dual-loop: async WSS + sync strategy pipeline
- [ ] BookCache с graceful degradation
- [ ] Position sizing: 5-10% per trade, 6-10 positions, 20-40% reserve

---

## Источники

1. [The Anatomy of Polymarket (arxiv 2603.03136)](https://arxiv.org/html/2603.03136) — Mar 2026
2. [Unravelling the Probabilistic Forest: Arbitrage (arxiv 2508.03474)](https://arxiv.org/html/2508.03474v1) — AFT 2025
3. [Exploring Decentralized PM: Accuracy, Skill, Bias (SSRN 5910522)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5910522) — Dec 2025
4. [Price Discovery in Modern Prediction Markets (SSRN 5331995)](https://papers.ssrn.com/sol3/Delivery.cfm/5331995.pdf?abstractid=5331995) — Jan 2026
5. [Polymarket Fee Curve & Dual-Loop Architecture](https://quantjourney.substack.com/p/understanding-the-polymarket-fee) — 2026
6. [Six Profit Models from 95M Transactions](https://www.chaincatcher.com/en/article/2233047) — 2025
7. [Systematic Edges in Prediction Markets (QuantPedia)](https://quantpedia.com/systematic-edges-in-prediction-markets/)
8. [Polymarket Official Accuracy](https://polymarket.com/accuracy)
