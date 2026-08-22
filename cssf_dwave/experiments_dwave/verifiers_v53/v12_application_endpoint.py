from __future__ import annotations
from bess.case300 import load_case300_mode_a
from .common import *


def verify(ctx:VerificationContext)->dict:
    p=maybe_json(ctx.evidence_root/"application_endpoint.json")
    if not p: return report("V12",ctx,[check("application_endpoint_present",False)])
    checks=[]
    data=load_case300_mode_a(ctx.project_root/"data"/"case300_full_modeA_Barskyi_Serhii.json")
    expected_n1=tuple(int(x) for x in data.params.get("safe_n1_lines",()))
    protected=set(p.get("protected_partition_ids",[])); selection=set(p.get("selection_input_ids",[]))
    checks.append(check("no_ood_n1_leakage",protected.isdisjoint(selection),observed=sorted(protected&selection),expected=[]))
    protocol_hashes={x.get("verification_protocol_sha256") for x in p.get("placements",[])}
    checks.append(check("same_verification_protocol",len(protocol_hashes)==1 and None not in protocol_hashes))
    checks.append(check("complete_frozen_n1_declared",int(p.get("full_n1_count",-1))==len(expected_n1),observed=p.get("full_n1_count"),expected=len(expected_n1)))
    for row in p.get("placements",[]):
        name=str(row.get("method")); n1=tuple(map(int,row.get("n1_line_indices",[])))
        checks.extend([
            check(f"{name}:AC",row.get("ac_complete") is True),
            check(f"{name}:OOD",row.get("ood_complete") is True),
            check(f"{name}:N1",row.get("n1_complete") is True),
            check(f"{name}:N1_full_set",n1==expected_n1,observed=len(n1),expected=len(expected_n1)),
            check(f"{name}:serialized_placement",bool(row.get("selected_buses"))),
            check(f"{name}:rerun_fingerprint",bool(row.get("independent_rerun_fingerprint"))),
            check(f"{name}:ood_rows",len(row.get("ood",{}).get("rows",[]))>0),
            check(f"{name}:n1_rows",len(row.get("n1",{}).get("rows",[]))==len(expected_n1),observed=len(row.get("n1",{}).get("rows",[])),expected=len(expected_n1)),
        ])
        if ctx.execute_expensive_checks:
            try:
                from experiments_dwave.application_endpoint_v38 import EndpointProtocol,evaluate_selected_buses
                pdict=row.get("verification_protocol") or p.get("verification_protocol") or {}
                protocol=EndpointProtocol(
                    validation_scenarios=int(pdict["validation_scenarios"]),ood_scenarios=int(pdict["ood_scenarios"]),
                    validation_seed=int(pdict["validation_seed"]),ood_seed=int(pdict["ood_seed"]),bess_power_mw=float(pdict["bess_power_mw"]),
                    n1_line_indices=tuple(map(int,pdict["n1_line_indices"])),
                    confirmation_scenarios=int(pdict.get("confirmation_scenarios",64)),confirmation_seed=int(pdict.get("confirmation_seed",20260847)),
                    pf_solver=str(pdict.get("pf_solver","pandapower.runpp")),
                    voltage_bounds_pu=tuple(map(float,pdict.get("voltage_bounds_pu",(0.90,1.10)))),loading_limit_percent=float(pdict.get("loading_limit_percent",100.0)),
                )
                rerun=evaluate_selected_buses(ctx.project_root,row["selected_buses"],method=name,power_mw=float(row.get("bess_power_mw",25.0)),protocol=protocol,placement_fingerprint=row.get("placement_fingerprint"))
                checks.append(check(f"{name}:independent_physical_rerun",rerun["independent_rerun_fingerprint"]==row.get("independent_rerun_fingerprint"),observed=rerun["independent_rerun_fingerprint"],expected=row.get("independent_rerun_fingerprint")))
            except Exception as exc:
                checks.append(check(f"{name}:independent_physical_rerun",False,observed=f"{type(exc).__name__}: {exc}",expected="deterministic identical rerun"))
    return report("V12",ctx,checks)
