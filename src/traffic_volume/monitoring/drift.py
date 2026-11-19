"""Data drift between an earlier and a later period, per feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(reference, current, bins: int = 10, categorical: bool = False) -> float:
    """PSI: below 0.1 is stable, 0.1 to 0.25 is a moderate shift, above 0.25 is a large shift."""
    eps = 1e-6
    if categorical:
        ref = pd.Series(reference).astype(str).value_counts(normalize=True)
        cur = pd.Series(current).astype(str).value_counts(normalize=True)
        keys = ref.index.union(cur.index)
        r, c = ref.reindex(keys, fill_value=0).to_numpy(), cur.reindex(keys, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(np.asarray(reference, float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(reference, edges)[0] / len(reference)
        c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """PSI for every feature, plus the two-sample KS test for numeric ones."""
    rows = []
    for col in reference.columns:
        is_cat = col in categorical
        row = {"feature": col, "type": "categorical" if is_cat else "numeric",
               "psi": population_stability_index(reference[col], current[col], categorical=is_cat)}
        if not is_cat:
            ks = ks_2samp(reference[col], current[col])
            row.update({"ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue)})
        row["level"] = "large" if row["psi"] > 0.25 else "moderate" if row["psi"] > 0.1 else "stable"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
