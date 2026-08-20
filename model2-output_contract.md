# Model 2 Output Contract

## How to call it

```
POST http://127.0.0.1:8001/forecast-cashflow
Content-Type: application/json
```

### Request body

| Field           | Type   | Required | Default | Meaning |
|-----------------|--------|----------|---------|---------|
| opening_cash    | float  | yes      | -       | Cash on hand today |
| daily_expense   | float  | yes      | -       | Flat expected daily outflow |
| horizon_days    | int    | no       | 90      | How many days ahead to forecast |
| n_sims          | int    | no       | 5000    | Number of Monte Carlo simulations |
| min_buffer      | float  | no       | 0       | Cash level counted as a "liquidity breach" |
| overdue_cap     | float  | no       | 0.4     | Confidence cap for invoices already past their own P90 (see README) |
| model1_api_url  | string | no       | http://127.0.0.1:8000/predict/open-invoices | Where Model 1's predictions are fetched from |

Example minimal request:
```json
{"opening_cash": 1200000, "daily_expense": 18000}
```

### Response body

```json
{
  "forecast": [
    {
      "day": 0,
      "cash_p10": 11212857.04,
      "cash_p50": 13787040.22,
      "cash_p90": 16384453.27,
      "prob_breach": 0.0
    },
    { "day": 1, "...": "..." }
  ],
  "summary": {
    "expected_min_cash": 13801434.15,
    "days_to_likely_breach": null,
    "overdue_invoice_count": 78,
    "overdue_invoice_value": 12287980.67
  }
}
```

### Field meanings

**`forecast`** - one entry per day, `0` to `horizon_days` inclusive.
- `cash_p10` / `cash_p50` / `cash_p90` - the pessimistic / typical / optimistic
  cash balance on that day, across all simulated scenarios
- `prob_breach` - fraction of simulations (0.0-1.0) where cash fell below
  `min_buffer` on that day

**`summary`**
- `expected_min_cash` - average, across all simulations, of the lowest cash
  balance reached at any point in the horizon
- `days_to_likely_breach` - first day where more than half of simulations show
  cash below `min_buffer`; `null` if that never happens
- `overdue_invoice_count` / `overdue_invoice_value` - how many open invoices
  (and their total value) were already past their own P90 prediction, and
  therefore had their payment-soon probability capped instead of trusted
  outright (see README "Why we cap overdue invoices")

### Error responses

- `502` - Model 1's API was unreachable, errored, or returned zero predictions.
  `detail` explains which.
- `422` - the request body itself was invalid (e.g. missing `opening_cash`) -
  standard FastAPI/Pydantic validation error.
