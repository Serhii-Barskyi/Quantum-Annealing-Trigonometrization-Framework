from __future__ import annotations
import numpy as np
from core.gcv_stable_v53 import tikhonov_solve_svd_v53
from .common import *

def verify(ctx:VerificationContext)->dict:
    payload=maybe_json(ctx.evidence_root/"csnnt_fits.json")
    if not payload or not payload.get("fits"): return report("V02",ctx,[check("csnnt_fit_evidence_present",False)])
    checks=[]; ids=[]
    for rec in payload["fits"]:
        eid=str(rec.get("evidence_id","")); ids.append(eid)
        rel=rec.get("audit_npz"); p=ctx.evidence_root/str(rel) if rel else Path("__missing__")
        checks.append(check(f"{eid}:audit_npz",p.is_file()))
        if not p.is_file(): continue
        z=np.load(p); X=np.asarray(z["X"],dtype=np.complex128); y=np.asarray(z["y"]); H=np.asarray(z["H"],dtype=np.complex128); pred=np.asarray(z["pred"],dtype=float)
        lam=float(rec["lambda"]); H2=tikhonov_solve_svd_v53(X,y,lam); pred2=np.real(X@H2)
        tol=float(rec.get("tolerance",1e-9))
        checks.extend([
            check(f"{eid}:finite",np.isfinite(pred).all() and np.isfinite(H.real).all() and np.isfinite(H.imag).all()),
            check(f"{eid}:coefficient_recompute",np.allclose(H,H2,rtol=tol,atol=tol),observed=float(np.max(np.abs(H-H2))),expected=f"<= {tol}"),
            check(f"{eid}:prediction_recompute",np.allclose(pred,pred2,rtol=tol,atol=tol),observed=float(np.max(np.abs(pred-pred2))),expected=f"<= {tol}"),
            check(f"{eid}:approved_call_path",str(rec.get("call_path","")) in {"core.csnn_t.fit_csnn_t","core.csnn_t_adapter.fit_csnn_t_surrogate","core.csnn_t_adapter_v53.fit_csnn_t_surrogate"}),
            check(f"{eid}:fingerprints",all(bool(rec.get(k)) for k in ("feature_fingerprint","target_fingerprint","train_partition_fingerprint","model_coefficient_sha256","surrogate_level"))),
        ])
    return report("V02",ctx,checks,evidence_ids=ids)
