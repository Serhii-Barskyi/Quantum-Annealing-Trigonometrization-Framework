from __future__ import annotations
from .dataset import load_dataset
from .metrics import compute_metrics
from core.csnn_t import fit_csnn_t

def validation_report(dataset_path: str, *, verbose: bool = False):
    ds = load_dataset(dataset_path)
    mdl = fit_csnn_t(ds, n_lambdas=100, lam_range=(-18, 4))
    pred = mdl.predict(ds.X_test)
    rho_dc = ds.params.get("rho_dc_vs_ac")
    result = compute_metrics(pred, ds.y_test, ds.meta_test, ds.non_slack, rho_dc_vs_ac=rho_dc)
    result.update({"lambda": float(mdl.lam_opt), "case": ds.case})
    if verbose:
        print(result)
    return result
