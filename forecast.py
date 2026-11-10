"""Forecast US daily COVID-19 cases and deaths with SARIMA, against a seasonal-naive baseline.

    python forecast.py            # writes docs/forecast_*.png and docs/results.md

Data: New York Times county-level snapshot (2020-01-21 → 2021-04-20), aggregated to the
national level in data/us_national_cumulative.csv. The NYT counts are *cumulative*, so the
daily series are first differences.
"""

from __future__ import annotations

import itertools
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

HORIZON = 28          # days held out for testing
SEASON = 7            # strong day-of-week reporting pattern
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")


def load_daily() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "us_national_cumulative.csv")
    cum = pd.read_csv(path, parse_dates=["date"], index_col="date").asfreq("D")
    daily = cum.diff().clip(lower=0).dropna()          # cumulative -> daily new counts
    daily.columns = ["new_cases", "new_deaths"]
    return daily


def select_order(y: pd.Series) -> tuple[tuple, tuple, float]:
    """Small AIC grid search over SARIMA orders (seasonal differencing fixed at D=1)."""
    best = (None, None, np.inf)
    for p, d, q, P, Q in itertools.product([0, 1, 2], [0, 1], [0, 1, 2], [0, 1], [0, 1]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(y, order=(p, d, q), seasonal_order=(P, 1, Q, SEASON)).fit(disp=False)
        except Exception:  # noqa: BLE001 - some orders fail to converge; skip them
            continue
        if res.aic < best[2]:
            best = ((p, d, q), (P, 1, Q, SEASON), res.aic)
    return best


def metrics(actual: pd.Series, pred: pd.Series) -> dict:
    err = actual - pred
    return {
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE %": float((err.abs() / actual.replace(0, np.nan)).mean() * 100),
    }


def evaluate(series: pd.Series, label: str) -> dict:
    train, test = series[:-HORIZON], series[-HORIZON:]
    # Model log(1 + y): counts are multiplicative and right-skewed.
    y = np.log1p(train)
    order, seasonal, aic = select_order(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(y, order=order, seasonal_order=seasonal).fit(disp=False)
    fc = fit.get_forecast(HORIZON)
    pred = np.expm1(fc.predicted_mean).clip(lower=0)
    ci = np.expm1(fc.conf_int(alpha=0.2)).clip(lower=0)      # 80 % interval
    pred.index = ci.index = test.index

    # Baseline: repeat the last observed week (seasonal naive).
    naive = pd.Series(np.tile(train[-SEASON:].values, HORIZON // SEASON + 1)[:HORIZON], index=test.index)

    m_model, m_naive = metrics(test, pred), metrics(test, naive)
    coverage = float(((test >= ci.iloc[:, 0]) & (test <= ci.iloc[:, 1])).mean() * 100)

    fig, ax = plt.subplots(figsize=(11, 4))
    recent = series[-120:]
    ax.plot(recent.index, recent, color="#334155", lw=1.2, label="observed")
    ax.plot(pred.index, pred, color="#2563eb", lw=2, label=f"SARIMA{order}x{seasonal}")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="#2563eb", alpha=0.15, label="80 % interval")
    ax.plot(naive.index, naive, color="#f59e0b", lw=1.5, ls="--", label="seasonal naive")
    ax.axvline(test.index[0], color="#94a3b8", ls=":", lw=1)
    ax.set(title=f"US daily {label}: {HORIZON}-day forecast", xlabel=None)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    os.makedirs(DOCS, exist_ok=True)
    fig.savefig(os.path.join(DOCS, f"forecast_{label.replace(' ', '_')}.png"), dpi=150)
    plt.close(fig)

    return {"label": label, "order": order, "seasonal": seasonal, "aic": aic,
            "model": m_model, "naive": m_naive, "coverage": coverage}


def main():
    daily = load_daily()
    results = [evaluate(daily["new_cases"], "new cases"), evaluate(daily["new_deaths"], "new deaths")]

    lines = [f"# Results ({HORIZON}-day holdout, {daily.index[-HORIZON].date()} → {daily.index[-1].date()})", ""]
    for r in results:
        lines += [f"## {r['label'].capitalize()}", "",
                  f"Selected model: SARIMA{r['order']}x{r['seasonal']} (AIC {r['aic']:.1f}), "
                  f"fitted on log(1 + y). 80 % interval coverage on the test period: {r['coverage']:.0f} %.", "",
                  "| | MAE | RMSE | MAPE |", "|---|---|---|---|"]
        for name, m in [("SARIMA", r["model"]), ("Seasonal naive", r["naive"])]:
            lines.append(f"| {name} | {m['MAE']:,.0f} | {m['RMSE']:,.0f} | {m['MAPE %']:.1f} % |")
        lines.append("")
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
