"""Every figure in the project, drawn in one house style.

Each function returns a matplotlib figure; ``finish_figure`` saves it at
200 dpi and either shows it (notebook) or closes it (pipeline).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from .artifacts import RunContext

PALETTE = {"ink": "#17252f", "green": "#0b6e4f", "blue": "#2b6cb0", "red": "#c0392b", "amber": "#d68910",
           "grey": "#8a96a3", "light": "#d9dee3"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TEST_MODEL_LABELS = {"baseline": "Hour-of-week\naverage", "baseline_holiday": "Average\n+ holidays", "ridge": "Ridge\n(linear)",
                     "univariate": "Hour only", "bivariate": "Hour +\nweekday", "alternative_inputs": "Other\ninput set",
                     "champion": "Champion"}


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.labelcolor": PALETTE["ink"], "axes.edgecolor": PALETTE["light"], "text.color": PALETTE["ink"],
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    })


def finish_figure(fig, name: str, ctx: RunContext) -> None:
    """Save the figure, then show it (notebook) or close it (pipeline). Returns nothing, so a notebook
    cell that ends with this call prints no file path."""
    path = ctx.fig_dir / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if ctx.show:
        plt.show()
    else:
        plt.close(fig)


def recorded_hours_per_month(raw: pd.DataFrame) -> pd.Series:
    stamps = pd.to_datetime(raw["date_time"]).drop_duplicates()
    per_month = stamps.dt.to_period("M").value_counts().sort_index()
    full = pd.period_range(per_month.index.min(), per_month.index.max(), freq="M")
    return per_month.reindex(full, fill_value=0)


def plot_timeline(raw: pd.DataFrame):
    """Recorded hours per month: shows the gaps in the series."""
    per_month = recorded_hours_per_month(raw)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(per_month.index.to_timestamp(), per_month.values, width=25,
           color=[PALETTE["red"] if v == 0 else PALETTE["green"] for v in per_month.values])
    ax.set(ylabel="Recorded hours", title="Recorded hours per month (gaps: months with few or no hours)")
    return fig


def plot_cleaning_log(log: list[dict]):
    frame = pd.DataFrame(log)
    frame["affected"] = frame["rows_removed"] + frame["values_changed"].fillna(0)
    frame = frame[frame["affected"] > 0]
    colors = [PALETTE["red"] if r > 0 else PALETTE["amber"] for r in frame["rows_removed"]]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(frame["rule"].str.replace("_", " "), frame["affected"], color=colors)
    ax.set_xscale("log")
    for i, v in enumerate(frame["affected"]):
        ax.text(v * 1.1, i, f"{int(v):,}", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="Rows removed (red) or values changed (amber), log scale", title="Cleaning rules")
    return fig


def plot_precipitation_audit(amounts: list[dict]):
    """Share of rainy and snowy records that carry a recorded amount, by year."""
    a = pd.DataFrame(amounts)
    rain = a["of_which_rain_1h_recorded"] / a["rain_weather_rows"].clip(lower=1)
    snow = a["of_which_snow_1h_recorded"] / a["snow_weather_rows"].clip(lower=1)
    x = np.arange(len(a))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(x - 0.2, rain * 100, 0.4, color=PALETTE["blue"], label="rain_1h > 0 when the weather says rain")
    ax.bar(x + 0.2, snow * 100, 0.4, color=PALETTE["grey"], label="snow_1h > 0 when the weather says snow")
    ax.set_xticks(x, a["year"])
    ax.set(ylabel="% of such records", ylim=(0, 105),
           title="The rain and snow amounts are not recorded consistently from year to year")
    ax.legend(fontsize=8)
    return fig


def plot_low_volume(clean_before: pd.DataFrame, target: str, threshold: int):
    """Hourly volumes on a log scale: the closure hours sit far below anything a normal hour produces."""
    v = clean_before[target]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].hist(np.log10(v.clip(lower=1)), bins=80, color=PALETTE["green"])
    axes[0].axvline(np.log10(threshold), color=PALETTE["red"], ls="--", label=f"{threshold} vehicles")
    axes[0].set(xlabel="Vehicles per hour (log10)", ylabel="Hours", title="A handful of hours far below the rest")
    axes[0].legend()
    low = clean_before[v < threshold]
    per_day = low.groupby(low["date_time"].dt.strftime("%Y-%m-%d")).size().sort_values(ascending=False).head(8)[::-1]
    axes[1].barh(per_day.index, per_day.values, color=PALETTE["red"])
    axes[1].set(xlabel=f"Hours below {threshold} vehicles", title="They cluster on a few days (road closures)")
    fig.tight_layout()
    return fig


def plot_holiday_evidence(evidence: pd.DataFrame, quiet: list[str]):
    e = evidence.sort_values("share_of_normal", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [PALETTE["red"] if n in quiet else PALETTE["grey"] for n in e["holiday"]]
    ax.barh(e["holiday"], e["share_of_normal"] * 100, color=colors)
    ax.errorbar(e["share_of_normal"] * 100, np.arange(len(e)),
                xerr=[(e["share_of_normal"] - e["lowest"]) * 100, (e["highest"] - e["share_of_normal"]) * 100],
                fmt="none", color=PALETTE["ink"], capsize=3, lw=1)
    ax.axvline(100, color=PALETTE["ink"], ls=":", lw=1)
    ax.set(xlabel="Traffic on the named day, % of an ordinary day of the same weekday (training years; bars: lowest and highest year)",
           title="Six named days are quiet (red); the other five are ordinary days (grey)")
    ax.xaxis.label.set_size(8.5)
    return fig


def plot_target_distribution(y: pd.Series, hours: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].hist(y, bins=60, color=PALETTE["green"])
    axes[0].set(xlabel="Vehicles per hour", ylabel="Hours", title="Traffic volume: two humps")
    night = hours.isin(range(0, 6))
    axes[1].hist(y[night], bins=40, alpha=0.6, color=PALETTE["blue"], label="00:00-05:59")
    axes[1].hist(y[~night], bins=40, alpha=0.6, color=PALETTE["amber"], label="06:00-23:59")
    axes[1].set(xlabel="Vehicles per hour", title="The humps are night and day")
    axes[1].legend()
    fig.tight_layout()
    return fig


def hour_profiles(clean: pd.DataFrame, target: str) -> pd.DataFrame:
    """Mean volume by hour for four kinds of day."""
    t = clean["date_time"]
    kind = np.where(clean["quiet_holiday"], "Quiet holiday",
                    np.where(clean["holiday"] != "None", "Other named day",
                             np.where(t.dt.dayofweek >= 5, "Weekend", "Weekday")))
    return clean.assign(hour=t.dt.hour, kind=kind).groupby(["kind", "hour"])[target].mean().unstack(0)


def plot_hour_profiles(clean: pd.DataFrame, target: str):
    prof = hour_profiles(clean, target)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for col, color, style in [("Weekday", PALETTE["green"], "-"), ("Other named day", PALETTE["amber"], "--"),
                              ("Weekend", PALETTE["blue"], "-"), ("Quiet holiday", PALETTE["red"], "-")]:
        if col in prof:
            ax.plot(prof.index, prof[col], marker="o", ms=3, color=color, ls=style, label=col)
    ax.set(xlabel="Hour of day", ylabel="Mean vehicles per hour", xticks=range(0, 24, 2),
           title="Daily profile by kind of day")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    return fig


def hour_weekday_table(hours, weekdays, values) -> pd.DataFrame:
    frame = pd.DataFrame({"hour": np.asarray(hours), "weekday": np.asarray(weekdays), "v": np.asarray(values, float)})
    return frame.groupby(["weekday", "hour"])["v"].mean().unstack("hour")


def plot_heatmap(table: pd.DataFrame, title: str, ax=None, vmax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 3.6))
    sns.heatmap(table, cmap="YlGn", ax=ax, vmin=0, vmax=vmax, cbar_kws={"label": "vehicles per hour"},
                yticklabels=[WEEKDAYS[int(i)] for i in table.index], xticklabels=[int(c) for c in table.columns])
    ax.tick_params(axis="y", rotation=0)
    ax.set(xlabel="Hour of day", ylabel="", title=title)
    return ax.figure if own else ax


def plot_weather_effects(clean: pd.DataFrame, target: str):
    day = clean[clean["date_time"].dt.hour.between(7, 18) & (clean["date_time"].dt.dayofweek < 5)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    order = day.groupby("weather_main")[target].median().sort_values().index
    sns.boxplot(data=day, x="weather_main", y=target, order=order, ax=axes[0], color=PALETTE["light"], fliersize=1)
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].set(xlabel="", ylabel="Vehicles per hour", title="Weekday daytime volume by weather")
    temp_c = day["temp"] - 273.15
    axes[1].scatter(temp_c, day[target], s=3, alpha=0.15, color=PALETTE["green"])
    axes[1].set(xlabel="Temperature (°C)", title="Weekday daytime volume against temperature")
    fig.tight_layout()
    return fig


def plot_monthly_trend(clean: pd.DataFrame, target: str):
    m = clean.set_index("date_time")[target].resample("MS").mean()
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(m.index, m.values, marker="o", ms=3, color=PALETTE["green"])
    ax.set(ylabel="Mean vehicles per hour", title="Monthly mean volume (gaps where no data)")
    return fig


def plot_correlation(features: pd.DataFrame, y: pd.Series):
    corr = features.select_dtypes("number").assign(traffic_volume=y.to_numpy()).corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.7})
    ax.tick_params(labelsize=8)
    ax.set_title("Correlation between numeric features and the target")
    return fig


def plot_candidates(table: pd.DataFrame):
    t = table.sort_values("val_rmse", ascending=False)
    names = {"baseline": "Hour-of-week average", "baseline_holiday": "Average + holidays", "linear": "Linear regression",
             "ridge": "Ridge", "lasso": "Lasso", "poly_ridge": "Polynomial ridge", "tree": "Decision tree",
             "rf": "Random forest", "hgb": "HistGradientBoosting", "lgbm": "LightGBM", "mlp": "Neural network"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh([names.get(m, m) for m in t["model"]], t["val_rmse"], color=PALETTE["green"],
                   label="Validation year (fit on all training years)")
    ax.bar_label(bars, fmt="%.0f", padding=-26, color="white", fontsize=8)
    ax.errorbar(t["cv_rmse_mean"], np.arange(len(t)), xerr=t["cv_rmse_std"], fmt="o", color=PALETTE["ink"], ms=4,
                label="Time-series CV, mean ± std")
    base = table.set_index("model").loc["baseline", "val_rmse"]
    ax.axvline(base, color=PALETTE["red"], ls="--", lw=1, label="Hour-of-week baseline")
    ax.set(xlabel="RMSE (vehicles per hour, lower is better)", title="Candidate models before tuning")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, frameon=False)
    return fig


def plot_search_comparison(tuning: dict):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for key, color, name in [("lgbm_grid", PALETTE["grey"], "Grid"), ("lgbm_random", PALETTE["blue"], "Random"),
                             ("lgbm_bayesian", PALETTE["red"], "Bayesian (Optuna TPE)")]:
        r = tuning[key]
        x = np.arange(1, len(r["best_so_far"]) + 1)
        ax.plot(x, r["best_so_far"], color=color, lw=2, label=name)
        ax.scatter(x, r["history"], color=color, s=10, alpha=0.4)
    lo = min(min(tuning[k]["history"]) for k in ("lgbm_grid", "lgbm_random", "lgbm_bayesian"))
    ax.set(ylim=(lo * 0.98, lo * 1.25), xlabel="Configurations tried (same budget each)",
           ylabel="CV RMSE (lower is better)", title="LightGBM: best error found vs. search effort")
    ax.legend()
    return fig


def plot_finalists(comparison: list[dict], champion_label: str, tolerance: float):
    t = pd.DataFrame(comparison).sort_values("val_rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colors = [PALETTE["green"] if m == champion_label else PALETTE["grey"] for m in t["model"]]
    bars = ax.barh(t["model"], t["val_rmse"], color=colors)
    ax.bar_label(bars, labels=[f"{v:.0f}   ({s / 1024:.1f} MB)" if s >= 1024 else f"{v:.0f}   ({s:.0f} KB)"
                               for v, s in zip(t["val_rmse"], t["size_kb"])], padding=3, fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    fig.set_size_inches(9, 5.6)
    best = t["val_rmse"].min()
    ax.axvspan(best, best + tolerance, color=PALETTE["amber"], alpha=0.25, label=f"tie zone ({tolerance:.0f} vehicles/h)")
    ax.set(xlim=(best * 0.8, t["val_rmse"].max() * 1.12), xlabel="Validation RMSE (model size in brackets)",
           title="Finalists on the validation year")
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_learning_curve(lc: dict):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sizes = np.array(lc["train_sizes"])
    ax.plot(sizes, lc["train_rmse"], marker="o", color=PALETTE["blue"], label="Training hours used")
    ax.plot(sizes, lc["val_rmse"], marker="o", color=PALETTE["red"], label="Validation year")
    ax.set(xlabel="Most recent training hours used", ylabel="RMSE", ylim=(0, None),
           title="Learning curve: error against the amount of history")
    ax.legend()
    return fig


def plot_complexity_curve(cc: dict):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(cc["values"], cc["train_rmse"], marker="o", color=PALETTE["blue"], label="Training years")
    ax.plot(cc["values"], cc["val_rmse"], marker="o", color=PALETTE["red"], label="Validation year")
    ax.axvline(cc["chosen"], color=PALETTE["ink"], ls=":", label=f"chosen: {cc['chosen']}")
    ax.set_xscale("log", base=2)
    ax.set(xlabel=f"{cc['param']} (model capacity)", ylabel="RMSE", ylim=(0, None),
           title="Complexity curve: underfitting on the left, overfitting on the right")
    ax.legend()
    return fig


def plot_residuals(report: dict):
    f = report["frame"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].scatter(f["actual"], f["predicted"], s=3, alpha=0.2, color=PALETTE["green"])
    lim = max(f["actual"].max(), f["predicted"].max())
    axes[0].plot([0, lim], [0, lim], color=PALETTE["ink"], ls=":")
    axes[0].set(xlabel="Actual", ylabel="Predicted", title="Predicted vs actual (validation)")
    bh = pd.DataFrame(report["by_hour"])
    axes[1].bar(bh["hour"], bh["rmse"], color=PALETTE["amber"])
    axes[1].set(xlabel="Hour of day", ylabel="RMSE", title="Error by hour of day")
    axes[2].hist(f["residual"], bins=80, color=PALETTE["blue"])
    axes[2].set(xlabel="Actual minus predicted", title="Residuals")
    fig.tight_layout()
    return fig


def plot_slices(slices: dict, where: str):
    """Baseline against champion in each kind of hour, and month by month."""
    seg, mon = pd.DataFrame(slices["by_segment"]), pd.DataFrame(slices["by_month"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(seg))
    axes[0].bar(x - 0.2, seg["baseline_rmse"], 0.4, color=PALETTE["red"], label="Hour-of-week average")
    bars = axes[0].bar(x + 0.2, seg["champion_rmse"], 0.4, color=PALETTE["green"], label="Champion")
    axes[0].bar_label(bars, labels=[f"{-r:+.0%}" for r in seg["reduction"]], fontsize=9, padding=2)
    axes[0].set_xticks(x, [g.replace(" ", "\n", 1) for g in seg["group"]], fontsize=9)
    axes[0].set(ylabel="RMSE (vehicles per hour)", title=f"{where}: error by kind of hour (label: change vs baseline)")
    axes[0].legend(fontsize=9)
    axes[1].plot(mon["group"], mon["baseline_rmse"], marker="o", color=PALETTE["red"], label="Hour-of-week average")
    axes[1].plot(mon["group"], mon["champion_rmse"], marker="o", color=PALETTE["green"], label="Champion")
    axes[1].set_xticks(mon["group"], [MONTHS[int(m) - 1] for m in mon["group"]])
    axes[1].set(ylabel="RMSE", ylim=(0, None), title=f"{where}: error month by month")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_importance(frame: pd.DataFrame, value: str, error: str | None, title: str, top: int = 15):
    t = frame.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(t["feature"], t[value], xerr=t[error] if error else None, color=PALETTE["green"])
    ax.set(title=title, xlabel=value.replace("_", " "))
    return fig


def plot_shap_summary(values):
    shap.plots.beeswarm(values, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5.5)
    plt.title("SHAP: how each feature moves the prediction (vehicles per hour)")
    return fig


def plot_variable_ranks(ranking: pd.DataFrame):
    ranks = ranking[[c for c in ranking.columns if c.endswith("_rank") and c != "mean_rank"]]
    ranks.columns = [c.replace("_rank", "").replace("_", " ") for c in ranks.columns]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.heatmap(ranks, annot=True, fmt=".0f", cmap="Greens_r", cbar=False, ax=ax)
    ax.set(xlabel="", ylabel="", title="Variable importance: rank by five methods (1 = most important)")
    return fig


def plot_univariate_fits(curves: dict, X: pd.DataFrame, y: pd.Series, variable: str, label: str):
    feats_col = {"temp": "temp_c"}.get(variable, variable)
    grid_raw = np.linspace(X[feats_col].min(), X[feats_col].max(), 200)
    fig, ax = plt.subplots(figsize=(8, 4))
    sample = X.sample(n=min(4000, len(X)), random_state=0).index
    ax.scatter(X.loc[sample, feats_col], y.loc[sample], s=3, alpha=0.15, color=PALETTE["grey"], label="Hours")
    colors = {"linear_raw": PALETTE["red"], "linear": PALETTE["amber"], "spline": PALETTE["blue"],
              "tree": PALETTE["green"]}
    for key, c in curves.items():
        lab, family = key.split("|")
        if lab != label:
            continue
        if variable == "hour":
            h = np.arange(24)
            grid = pd.DataFrame({"hour": h.astype(float), "hour_sin": np.sin(2 * np.pi * h / 24),
                                 "hour_cos": np.cos(2 * np.pi * h / 24)})
            xs = h
        else:
            grid = pd.DataFrame({feats_col: grid_raw})
            xs = grid_raw
        ax.plot(xs, c["model"].predict(grid[c["columns"]]), color=colors[family], lw=2.2, label=family.replace("_", " "))
    ax.set(xlabel=feats_col.replace("_", " "), ylabel="Vehicles per hour", title=f"Univariate models on {variable}")
    ax.legend(fontsize=8)
    return fig


def plot_bivariate_surface(actual: pd.DataFrame, predicted: pd.DataFrame):
    vmax = float(max(actual.max().max(), predicted.max().max()))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    plot_heatmap(actual, "Actual mean volume (validation year)", ax=axes[0], vmax=vmax)
    plot_heatmap(predicted, "Bivariate model (hour and weekday only)", ax=axes[1], vmax=vmax)
    fig.tight_layout()
    return fig


def plot_variable_models(table: pd.DataFrame):
    t = table[~table["model"].str.contains("raw weather")]
    order = ["univariate", "bivariate", "multivariate"]
    fams = ["linear_raw", "linear", "spline", "tree"]
    colors = {"linear_raw": PALETTE["red"], "linear": PALETTE["amber"], "spline": PALETTE["blue"],
              "tree": PALETTE["green"]}
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.2
    for i, fam in enumerate(fams):
        vals = [t[(t["model"] == m) & (t["family"] == fam)]["r2"] for m in order]
        ys = [v.iloc[0] if len(v) else np.nan for v in vals]
        bars = ax.bar(np.arange(3) + (i - 1.5) * width, ys, width, color=colors[fam], label=fam.replace("_", " "))
        ax.bar_label(bars, labels=["" if np.isnan(v) else f"{v:.2f}" for v in ys], fontsize=7, padding=2)
    ax.set_xticks(np.arange(3), ["1 variable", "2 variables", "all variables"])
    ax.set(ylim=(0, 1.12), ylabel="Validation R²", title="Univariate, bivariate and multivariate models")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    return fig


def plot_incremental(inc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(inc["variables"], inc["rmse"], marker="o", color=PALETTE["green"])
    for _, r in inc.iterrows():
        ax.annotate(f"+{r['added']}", (r["variables"], r["rmse"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set(xlabel="Variables used (added in order of importance)", ylabel="Validation RMSE", ylim=(0, None),
           title="Error as variables are added one at a time")
    return fig


def plot_group_ablation(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(table["inputs"], table["rmse"], color=[PALETTE["grey"]] * (len(table) - 1) + [PALETTE["green"]])
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="Validation RMSE", xlim=(0, table["rmse"].max() * 1.12), title="What each group of inputs adds")
    return fig


def plot_lag_experiment(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 3))
    labels = [f"{f}\n({u})" for f, u in zip(table["features"], table["usable"])]
    bars = ax.barh(labels, table["rmse"], color=[PALETTE["green"]] + [PALETTE["grey"]] * (len(table) - 1))
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    ax.set(xlabel="Validation RMSE", xlim=(0, table["rmse"].max() * 1.12),
           title="Recent counts help only at short horizons")
    return fig


def plot_test_results(test: dict):
    names = list(test["models"])
    rmse = [test["models"][n]["metrics"]["rmse"] for n in names]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    colors = [PALETTE["red"] if n == "baseline" else PALETTE["green"] if n == "champion" else PALETTE["grey"]
              for n in names]
    labels = dict(TEST_MODEL_LABELS)
    if "label" in test["models"].get("alternative_inputs", {}):
        labels["alternative_inputs"] = test["models"]["alternative_inputs"]["label"].replace(" (", "\n(")
    ax.bar([labels.get(n, n) for n in names], rmse, color=colors)
    ci = test["models"]["champion"]["ci"]["rmse"]
    j = names.index("champion")
    ax.errorbar([j], [rmse[j]], yerr=[[rmse[j] - ci["low"]], [ci["high"] - rmse[j]]], color=PALETTE["ink"],
                capsize=4, fmt="none")
    for i, v in enumerate(rmse):
        ax.text(i, v + 12, f"{v:.0f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.set(ylabel="Test RMSE (vehicles per hour)", ylim=(0, max(rmse) * 1.12),
           title="Test year, every pre-declared model (bar: 95% block-bootstrap CI)")
    return fig


def test_week_frame(frame: pd.DataFrame, clean_test: pd.DataFrame) -> pd.DataFrame:
    """Seven days of the test year that contain a quiet holiday (the first one from May on), so the
    picture shows both ordinary days and the kind of day on which the baseline fails."""
    quiet = clean_test.loc[clean_test["quiet_holiday"] & (clean_test["date_time"].dt.month >= 5), "date_time"]
    stamps = pd.to_datetime(frame["date_time"])
    start = (quiet.min().normalize() - pd.Timedelta(days=3)) if len(quiet) else stamps.iloc[len(frame) // 3].normalize()
    return frame[(stamps >= start) & (stamps < start + pd.Timedelta(days=7))]


def plot_test_week(week: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    t = pd.to_datetime(week["date_time"])
    ax.plot(t, week["actual"], color=PALETTE["ink"], lw=1.8, label="Actual")
    ax.plot(t, week["champion"], color=PALETTE["green"], lw=1.5, label="Champion")
    ax.plot(t, week["baseline"], color=PALETTE["red"], lw=1.1, ls="--", label="Hour-of-week average")
    ax.set(ylabel="Vehicles per hour", title=f"One week of the test year, {t.min():%d %b} to {t.max():%d %b %Y}")
    ax.legend(ncol=3, fontsize=8)
    return fig


def plot_drift(report: pd.DataFrame, title: str, top: int = 15):
    t = report.head(top).iloc[::-1]
    colors = [PALETTE["red"] if v > 0.25 else PALETTE["amber"] if v > 0.1 else PALETTE["green"] for v in t["psi"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(t["feature"], t["psi"], color=colors)
    ax.axvline(0.1, color=PALETTE["grey"], ls=":", label="0.1 moderate shift")
    ax.axvline(0.25, color=PALETTE["ink"], ls="--", label="0.25 large shift")
    ax.set(xlabel="Population stability index", title=title)
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_latency(result: dict):
    lat = np.asarray(result["client_latencies_ms"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(lat[1:] if len(lat) > 1 else lat, bins=40, color=PALETTE["blue"])
    ax.axvline(result["warm_p50_ms"], color=PALETTE["green"], ls="--", label=f"p50 {result['warm_p50_ms']:.0f} ms")
    ax.axvline(result["warm_p95_ms"], color=PALETTE["red"], ls="--", label=f"p95 {result['warm_p95_ms']:.0f} ms")
    ax.set(xlabel="Round-trip latency per request (ms)", ylabel="Requests",
           title=f"Latency of {result['requests']} requests (first request: {result['first_request_ms']:.0f} ms)")
    ax.legend()
    return fig
