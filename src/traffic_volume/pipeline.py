"""Pipeline stages. Each reads its inputs from disk and writes its outputs to disk, so any stage
can be re-run on its own and an interrupted run can resume. Logic lives in the shared modules."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .artifacts import RunContext, load_json, save_json, write_manifest
from .cleaning import clean_traffic, holiday_evidence
from .config import Config
from .data import download_raw, load_raw
from .evaluation import evaluate_regressors
from .experiments import LAG_MODEL, lag_experiment, lag_frame
from .explain import permutation_frame, shap_frame, shap_values_for
from .export import export_for_service, served_predictions
from .feature_study import (
    bivariate_surface,
    group_ablation,
    rank_variables,
    univariate_curves,
    variable_models,
    vif_table,
)
from .features import build_feature_frame
from .log import LOGGER
from .monitoring.drift import drift_report
from .monitoring.latency import replay_requests
from .plotting import (
    finish_figure,
    hour_profiles,
    hour_weekday_table,
    plot_bivariate_surface,
    plot_candidates,
    plot_cleaning_log,
    plot_complexity_curve,
    plot_correlation,
    plot_finalists,
    plot_group_ablation,
    plot_heatmap,
    plot_holiday_evidence,
    plot_hour_profiles,
    plot_importance,
    plot_incremental,
    plot_lag_experiment,
    plot_learning_curve,
    plot_low_volume,
    plot_monthly_trend,
    plot_precipitation_audit,
    plot_residuals,
    plot_search_comparison,
    plot_shap_summary,
    plot_slices,
    plot_target_distribution,
    plot_test_results,
    plot_test_week,
    plot_timeline,
    plot_univariate_fits,
    plot_variable_models,
    plot_variable_ranks,
    plot_weather_effects,
    recorded_hours_per_month,
    set_plot_style,
    test_week_frame,
)
from .split import chronological_split, fast_sample, split_summary
from .tuning import run_all_searches
from .validation import validate_raw
from .workflow import (
    build_frames,
    choose_champion,
    diagnose,
    drop_unserialisable,
    final_test,
    fit_tuned,
    results_report,
    test_pass_models,
)

STAGES = ["download", "validate", "clean", "split", "eda", "train", "tune", "select", "diagnose", "explain",
          "feature-study", "evaluate", "export", "report"]


def raw_csv(ctx: RunContext) -> Path:
    return ctx.data_dir / "raw" / "traffic.csv"


def load_clean(ctx: RunContext) -> pd.DataFrame:
    return pd.read_pickle(ctx.data_dir / "interim" / "clean.pkl")


def frames_on_disk(cfg: Config, ctx: RunContext) -> dict:
    splits = {k: np.array(v) for k, v in load_json(ctx.data_dir / "processed" / "splits.json").items()}
    return build_frames(load_clean(ctx), splits, cfg)


def stage_download(cfg: Config, ctx: RunContext) -> dict:
    info = download_raw(cfg, ctx.data_dir / "raw", repo_root=ctx.repo)
    info["path"] = Path(info["path"]).name
    if info["source_kind"] == "repo_mirror":
        info["source"] = cfg.data.mirror_path
    save_json(info, ctx.data_dir / "raw" / "source.json")
    return info


def stage_validate(cfg: Config, ctx: RunContext) -> dict:
    audit = validate_raw(load_raw(raw_csv(ctx)), cfg)
    save_json(audit, ctx.metrics_dir / "data_audit.json")
    return audit


def stage_clean(cfg: Config, ctx: RunContext) -> dict:
    clean, log = clean_traffic(load_raw(raw_csv(ctx)), cfg)
    hours = len(clean)
    clean = fast_sample(clean, cfg)
    clean.to_pickle(ctx.data_dir / "interim" / "clean.pkl")
    out = {"cleaning_log": log, "hours_after_cleaning": hours, "hours_used": len(clean)}
    save_json(out, ctx.metrics_dir / "cleaning.json")
    return out


def stage_split(cfg: Config, ctx: RunContext) -> dict:
    clean = load_clean(ctx)
    splits = chronological_split(clean, cfg)
    save_json({k: v.tolist() for k, v in splits.items()}, ctx.data_dir / "processed" / "splits.json")
    summary = split_summary(clean, splits, cfg.target)
    save_json(summary, ctx.metrics_dir / "split.json")
    evidence = holiday_evidence(clean.iloc[splits["train"]], cfg.target)
    save_json(evidence.to_dict("records"), ctx.metrics_dir / "holiday_evidence.json")
    return summary


def stage_eda(cfg: Config, ctx: RunContext) -> dict:
    f = frames_on_disk(cfg, ctx)
    trval, t = f["clean_trval"], cfg.target
    raw = load_raw(raw_csv(ctx))
    hourly = raw.drop_duplicates("date_time").assign(date_time=lambda d: pd.to_datetime(d["date_time"]))
    audit = load_json(ctx.metrics_dir / "data_audit.json")
    evidence = pd.DataFrame(load_json(ctx.metrics_dir / "holiday_evidence.json"))
    finish_figure(plot_timeline(raw), "eda_timeline", ctx)
    finish_figure(plot_precipitation_audit(audit["precipitation_amounts_by_year"]), "eda_precipitation_audit", ctx)
    finish_figure(plot_low_volume(hourly, t, cfg.cleaning.min_plausible_volume), "eda_low_volume", ctx)
    finish_figure(plot_cleaning_log(load_json(ctx.metrics_dir / "cleaning.json")["cleaning_log"]), "eda_cleaning_rules", ctx)
    finish_figure(plot_holiday_evidence(evidence, cfg.cleaning.quiet_holidays), "eda_holiday_evidence", ctx)
    finish_figure(plot_target_distribution(trval[t], trval["date_time"].dt.hour), "eda_target_distribution", ctx)
    finish_figure(plot_hour_profiles(trval, t), "eda_hour_profiles", ctx)
    table = hour_weekday_table(trval["date_time"].dt.hour, trval["date_time"].dt.dayofweek, trval[t])
    finish_figure(plot_heatmap(table, "Mean volume by weekday and hour (training and validation years)"),
                  "eda_hour_weekday_heatmap", ctx)
    finish_figure(plot_weather_effects(trval, t), "eda_weather_effects", ctx)
    finish_figure(plot_monthly_trend(trval, t), "eda_monthly_trend", ctx)
    finish_figure(plot_correlation(f["X_trval"], f["y_trval"]), "eda_correlation", ctx)
    per_month = recorded_hours_per_month(raw)
    profiles = hour_profiles(trval, t)
    save_json({"recorded_hours_per_month": {str(k): int(v) for k, v in per_month.items()},
               "hour_profiles": {k: [float(x) for x in profiles[k]] for k in profiles.columns},
               "hour_weekday_mean": [[float(x) for x in row] for row in table.to_numpy()],
               "weekday_daytime_median_by_weather": {
                   k: float(v) for k, v in trval[trval["date_time"].dt.hour.between(7, 18) & (trval["date_time"].dt.dayofweek < 5)]
                   .groupby("weather_main")[t].median().sort_values().items()}},
              ctx.metrics_dir / "eda.json")
    return {"figures": 11}


def stage_train(cfg: Config, ctx: RunContext) -> dict:
    f = frames_on_disk(cfg, ctx)
    table = evaluate_regressors(f["X_train"], f["y_train"], f["X_validation"], f["y_validation"], f["cats"], f["nums"], cfg)
    lags = lag_experiment(f["X"], f["y"], lag_frame(load_clean(ctx), cfg.target), f["splits"], cfg)
    save_json(table.to_dict("records"), ctx.metrics_dir / "candidates.json")
    save_json({"table": lags.to_dict("records"), "lag_available_share": lags.attrs["lag_available_share"],
               "settings": f"{LAG_MODEL}, default settings, every row"}, ctx.metrics_dir / "lags.json")
    finish_figure(plot_candidates(table), "models_candidates", ctx)
    finish_figure(plot_lag_experiment(lags), "models_lag_experiment", ctx)
    return {"best": table.iloc[0]["model"]}


def stage_tune(cfg: Config, ctx: RunContext) -> dict:
    f = frames_on_disk(cfg, ctx)
    tuning = run_all_searches(f["X_train"], f["y_train"], f["cats"], f["nums"], cfg,
                              cache_dir=ctx.artifacts_dir / "tuning_parts" / ("fast" if cfg.is_fast else "full"))
    save_json(tuning, ctx.metrics_dir / "tuning.json")
    finish_figure(plot_search_comparison(tuning), "tuning_strategies", ctx)
    return {k: v["best_rmse"] for k, v in tuning.items()}


def stage_select(cfg: Config, ctx: RunContext) -> dict:
    f = frames_on_disk(cfg, ctx)
    fitted = fit_tuned(load_json(ctx.metrics_dir / "tuning.json"), f, cfg)
    champion = choose_champion(fitted, cfg)
    joblib.dump(fitted[champion["label"]]["model"], ctx.artifacts_dir / "champion_train_fit.joblib")
    joblib.dump(fitted[champion["all_inputs_label"]]["model"], ctx.artifacts_dir / "all_inputs_train_fit.joblib")
    save_json(champion, ctx.metrics_dir / "champion.json")
    finish_figure(plot_finalists(champion["comparison"], champion["label"], cfg.selection.tie_tolerance_rmse),
                  "models_finalists", ctx)
    return {k: v for k, v in champion.items() if k != "comparison"}


def champion_on_disk(ctx: RunContext):
    return joblib.load(ctx.artifacts_dir / "champion_train_fit.joblib"), load_json(ctx.metrics_dir / "champion.json")


def stage_diagnose(cfg: Config, ctx: RunContext) -> dict:
    f = frames_on_disk(cfg, ctx)
    model, champion = champion_on_disk(ctx)
    d = diagnose(model, champion, f, cfg)
    save_json(drop_unserialisable(d), ctx.metrics_dir / "diagnostics.json")
    finish_figure(plot_learning_curve(d["learning_curve"]), "diag_learning_curve", ctx)
    if d["complexity_curve"]:
        finish_figure(plot_complexity_curve(d["complexity_curve"]), "diag_complexity_curve", ctx)
    finish_figure(plot_residuals(d["residuals"]), "diag_residuals", ctx)
    finish_figure(plot_slices(d["slices"], "Validation year"), "diag_slices_validation", ctx)
    return {"val_rmse": d["validation"]["rmse"], "train_rmse": d["train"]["rmse"]}


def stage_explain(cfg: Config, ctx: RunContext) -> dict:
    f = frames_on_disk(cfg, ctx)
    model, _ = champion_on_disk(ctx)
    perm = permutation_frame(model, f["X_validation"], f["y_validation"], cfg)
    values = shap_values_for(model, f["X_validation"], cfg)
    save_json({"permutation": perm.to_dict("records"), "shap": shap_frame(values).to_dict("records")},
              ctx.metrics_dir / "explain.json")
    finish_figure(plot_importance(perm, "rmse_increase", "std", "Permutation importance (increase in validation RMSE)"),
                  "explain_permutation", ctx)
    finish_figure(plot_shap_summary(values), "explain_shap_summary", ctx)
    return {"top_feature": perm.iloc[0]["feature"]}


def stage_feature_study(cfg: Config, ctx: RunContext) -> dict:
    """Variable importance is measured on the model that sees every variable (the champion's family and
    settings on all inputs), so that a variable the champion leaves out can still be ranked."""
    f = frames_on_disk(cfg, ctx)
    _, champion = champion_on_disk(ctx)
    model = joblib.load(ctx.artifacts_dir / "all_inputs_train_fit.joblib")
    params, kind = champion["params"], champion["name"]
    values = shap_values_for(model, f["X_validation"], cfg)
    ranking = rank_variables(f["raw_train"], f["y_train"], f["raw_validation"], f["y_validation"], model, values, cfg)
    study = variable_models(f["X"], f["y"], f["splits"], ranking, params, cfg, kind)
    actual, predicted = bivariate_surface(f["X"], f["y"], f["splits"], study["top2"], params, cfg, kind)
    ablation = group_ablation(f["X"], f["y"], f["splits"], load_clean(ctx), params, cfg, kind)
    Xv, yv = f["X_validation"], f["y_validation"]
    save_json({"ranking": ranking.reset_index().to_dict("records"), "vif": vif_table(f["X_train"]).to_dict("records"),
               "top1": study["top1"], "top2": study["top2"], "table": study["table"].to_dict("records"),
               "incremental": study["incremental"].to_dict("records"),
               "group_ablation": ablation.to_dict("records"), "wet_hours": ablation.attrs["wet_hours"],
               "univariate_curves": univariate_curves(study["curves"], Xv, yv, study["top1"][0]),
               "bivariate_surface": {"actual": [[float(x) for x in r] for r in actual.to_numpy()],
                                     "predicted": [[float(x) for x in r] for r in predicted.to_numpy()]}},
              ctx.metrics_dir / "feature_study.json")
    finish_figure(plot_variable_ranks(ranking), "study_variable_ranks", ctx)
    finish_figure(plot_univariate_fits(study["curves"], Xv, yv, study["top1"][0], "univariate"), "study_univariate", ctx)
    finish_figure(plot_univariate_fits(study["curves"], Xv, yv, cfg.feature_study.contrast_variable,
                                       "univariate (raw weather)"), "study_univariate_weather", ctx)
    finish_figure(plot_bivariate_surface(actual, predicted), "study_bivariate_surface", ctx)
    finish_figure(plot_variable_models(study["table"]), "study_variable_models", ctx)
    finish_figure(plot_incremental(study["incremental"]), "study_incremental", ctx)
    finish_figure(plot_group_ablation(ablation), "study_group_ablation", ctx)
    return {"top1": study["top1"], "top2": study["top2"]}


def stage_evaluate(cfg: Config, ctx: RunContext) -> dict:
    f = frames_on_disk(cfg, ctx)
    _, champion = champion_on_disk(ctx)
    study = load_json(ctx.metrics_dir / "feature_study.json")
    specs = test_pass_models(champion, load_json(ctx.metrics_dir / "tuning.json"), study, f, cfg)
    test = final_test(specs, f, cfg)
    joblib.dump({k: test["fitted"][k] for k in ("champion", "alternative_inputs")},
                ctx.artifacts_dir / "final_models.joblib")
    test["frame"].to_pickle(ctx.artifacts_dir / "test_predictions.pkl")
    week = test_week_frame(test["frame"], f["clean_test"])
    report = drop_unserialisable(test)
    report["week"] = {"date_time": week["date_time"].astype(str).tolist(),
                      **{k: [float(x) for x in week[k]] for k in ("actual", "champion", "baseline")}}
    save_json(report, ctx.metrics_dir / "test.json")
    finish_figure(plot_test_results(test), "test_results", ctx)
    finish_figure(plot_test_week(week), "test_week", ctx)
    finish_figure(plot_slices(test["slices"], "Test year"), "test_slices", ctx)
    return test["success"]


def stage_export(cfg: Config, ctx: RunContext) -> dict:
    f = frames_on_disk(cfg, ctx)
    models = joblib.load(ctx.artifacts_dir / "final_models.joblib")
    _, champion = champion_on_disk(ctx)
    test = load_json(ctx.metrics_dir / "test.json")
    out = ctx.artifacts_dir / "export"
    for stale in out.glob("*.onnx"):
        stale.unlink()
    meta, deployment = export_for_service(models, champion, test["models"], f, cfg, out)
    save_json(meta, out / "metadata.json")
    # Rows and expected answers for the service's own parity test (tests/parity).
    f["clean_validation"].head(200).to_pickle(out / "parity_rows.pkl")
    np.save(out / "parity_expected.npy", models["champion"].predict(f["X_validation"].head(200)))
    np.save(out / "parity_expected_calendar.npy", models["alternative_inputs"].predict(f["X_validation"].head(200))
            if meta["fallback"] else np.array([]))
    save_json(deployment, ctx.metrics_dir / "deployment.json")
    if not deployment["parity_ok"]:
        raise RuntimeError(f"export parity failed: {deployment}")
    return deployment


def stage_report(cfg: Config, ctx: RunContext) -> dict:
    m = ctx.metrics_dir
    cleaning, explain = load_json(m / "cleaning.json"), load_json(m / "explain.json")
    f = frames_on_disk(cfg, ctx)
    parts = {"audit": load_json(m / "data_audit.json"), "cleaning_log": cleaning["cleaning_log"],
             "hours": cleaning["hours_after_cleaning"], "split_summary": load_json(m / "split.json"),
             "holiday_evidence": load_json(m / "holiday_evidence.json"),
             "cats": f["cats"], "nums": f["nums"], "candidates": load_json(m / "candidates.json"),
             "lags": load_json(m / "lags.json"), "tuning": load_json(m / "tuning.json"),
             "champion": load_json(m / "champion.json"), "diagnostics": load_json(m / "diagnostics.json"),
             "permutation": pd.DataFrame(explain["permutation"]), "shap": pd.DataFrame(explain["shap"]),
             "study": load_json(m / "feature_study.json"), "test": load_json(m / "test.json"),
             "deployment": load_json(m / "deployment.json") if (m / "deployment.json").exists() else None}
    save_json(results_report(parts, cfg), m / "metrics.json")
    return {"metrics_file": "metrics.json"}


STAGE_FUNCTIONS = {name: globals()["stage_" + name.replace("-", "_")] for name in STAGES}


def run_stage(name: str, cfg: Config, ctx: RunContext) -> dict:
    set_plot_style()
    ctx.ensure()
    start = time.time()
    LOGGER.info("stage %s: start%s", name, " (fast mode)" if cfg.is_fast else "")
    result = STAGE_FUNCTIONS[name](cfg, ctx)
    seconds = round(time.time() - start, 1)
    LOGGER.info("stage %s: done in %.1fs", name, seconds)
    write_manifest(ctx, name, dataclasses.asdict(cfg), {"seconds": seconds, "result": drop_unserialisable(result)})
    return result


def run_all(cfg: Config, ctx: RunContext) -> None:
    for name in STAGES:
        run_stage(name, cfg, ctx)


def run_latency(cfg: Config, ctx: RunContext, url: str, n_rows: int = 300) -> dict:
    out = ctx.artifacts_dir / "export"
    meta = load_json(out / "metadata.json")
    f = frames_on_disk(cfg, ctx)
    rows, raw = f["clean_validation"].head(n_rows), f["raw_validation"].head(n_rows)
    result = replay_requests(url, rows, served_predictions(raw, meta, out / meta["file"]))
    save_json(result, ctx.metrics_dir / "monitoring.json")
    return result


def run_drift(cfg: Config, ctx: RunContext) -> dict:
    """Feature drift between the training years and the validation year (no target used)."""
    f = frames_on_disk(cfg, ctx)
    report = drift_report(f["X_train"], f["X_validation"], f["cats"])
    save_json(report.to_dict("records"), ctx.metrics_dir / "drift.json")
    return {"largest_shift": report.iloc[0].to_dict()}


__all__ = ["STAGES", "run_all", "run_stage", "run_latency", "run_drift", "build_feature_frame"]
