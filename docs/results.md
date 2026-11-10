# Results (28-day holdout, 2021-03-24 → 2021-04-20)

## New cases

Selected model: SARIMA(2, 1, 2)x(1, 1, 1, 7) (AIC 193.0), fitted on log(1 + y). 80 % interval coverage on the test period: 100 %.

| | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA | 10,135 | 12,011 | 15.9 % |
| Seasonal naive | 12,373 | 13,782 | 17.9 % |

## New deaths

Selected model: SARIMA(2, 1, 2)x(0, 1, 1, 7) (AIC 25.4), fitted on log(1 + y). 80 % interval coverage on the test period: 93 %.

| | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA | 339 | 481 | 36.0 % |
| Seasonal naive | 269 | 402 | 29.4 % |
