"""Claim-grade AC/OOD/full-N-1 BESS endpoint for CSSF(QA) v38.

The module is additive and never changes the frozen CSSF scientific sources.  It
serializes enough protocol state to let V12 independently rerun every placement
under the same deterministic physical verification design.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bess.case300 import load_case300_mode_a
from config.loader import load_config
from opf.bess_constraints import BESSPlacement
from opf.case_loader import load_power_case
from opf.scenario_generator import ScenarioBatch, ScenarioGeneratorConfig, generate_scenarios


class ApplicationEndpointError(RuntimeError):
    pass


def _json_hash(value: Any) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False,default=str).encode("utf-8")
    return hashlib.sha256(b"CSSF-application-endpoint-v38\0"+raw).hexdigest()


def _placement_fingerprint(placement: BESSPlacement) -> str:
    return _json_hash({"fleet":placement.fleet.fingerprint(),"selected_buses":list(placement.selected_buses)})


def _total_ac_losses(net: Any) -> float:
    total=0.0
    for name in ("res_line","res_trafo","res_trafo3w"):
        table=getattr(net,name,None)
        if table is not None and not table.empty and "pl_mw" in table.columns:
            total+=float(table["pl_mw"].sum())
    return float(total)


def _network_feasible(net: Any) -> bool:
    if not bool(getattr(net,"converged",False)): return False
    bus=getattr(net,"res_bus",None)
    if bus is not None and not bus.empty and "vm_pu" in bus.columns:
        vm=bus["vm_pu"].to_numpy(dtype=float)
        if np.any(vm<0.90) or np.any(vm>1.10): return False
    for name in ("res_line","res_trafo","res_trafo3w"):
        table=getattr(net,name,None)
        if table is not None and not table.empty and "loading_percent" in table.columns:
            if np.any(table["loading_percent"].to_numpy(dtype=float)>100.0+1e-8): return False
    return True


def _run_pf(net: Any) -> tuple[bool,float,str,str]:
    try:
        import pandapower as pp
    except Exception as exc:  # pragma: no cover - installed by target notebooks
        raise ApplicationEndpointError("pandapower is required for the physical endpoint") from exc
    try:
        pp.runpp(net,calculate_voltage_angles=True,init="auto",numba=True,enforce_q_lims=True)
        loss=_total_ac_losses(net); feasible=_network_feasible(net)
        return bool(feasible),float(loss),"",""
    except Exception as exc:
        return False,float("nan"),type(exc).__name__,str(exc)[:500]


def _apply_buses(net: Any, selected_buses: tuple[int,...], power_mw: float) -> None:
    try:
        import pandapower as pp
    except Exception as exc:  # pragma: no cover
        raise ApplicationEndpointError("pandapower is required for the physical endpoint") from exc
    for bus in selected_buses:
        pp.create_sgen(net,bus=int(bus),p_mw=float(power_mw),q_mvar=0.0,name="CSSF_BESS_fixed_discharge")


def _scenario_rows(selected_buses: tuple[int,...], power_mw: float, batch: ScenarioBatch, loaded_case: Any, partition: str) -> list[dict[str,Any]]:
    rows=[]
    for scenario_id,net in batch.iter_networks(loaded_case):
        _apply_buses(net,selected_buses,power_mw)
        feasible,loss,etype,emsg=_run_pf(net)
        rows.append({"partition":partition,"scenario_id":f"{partition}:{scenario_id}","feasible":feasible,"raw_loss_mw":None if not np.isfinite(loss) else float(loss),"penalized_objective":float(loss) if feasible else 1.0e9,"error_type":etype,"error_message":emsg})
    return rows


def _n1_rows(selected_buses: tuple[int,...], power_mw: float, loaded_case: Any, line_indices: tuple[int,...]) -> list[dict[str,Any]]:
    rows=[]
    for line_index in line_indices:
        # LoadedPowerCase exposes an immutable validated base; deepcopy is deliberate for each contingency.
        import copy
        net=copy.deepcopy(loaded_case.network)
        line=getattr(net,"line",None)
        if line is None or int(line_index) not in line.index:
            raise ApplicationEndpointError(f"Frozen N-1 line {line_index} is absent from case300")
        if "in_service" not in line.columns: line["in_service"]=True
        line.loc[int(line_index),"in_service"]=False
        _apply_buses(net,selected_buses,power_mw)
        feasible,loss,etype,emsg=_run_pf(net)
        rows.append({"partition":"n1","scenario_id":f"n1:line:{int(line_index)}","line_index":int(line_index),"feasible":feasible,"raw_loss_mw":None if not np.isfinite(loss) else float(loss),"penalized_objective":float(loss) if feasible else 1.0e9,"error_type":etype,"error_message":emsg})
    return rows


def _summary(rows: list[dict[str,Any]]) -> dict[str,Any]:
    values=np.asarray([float(r["penalized_objective"]) for r in rows],dtype=float)
    feasible=np.asarray([bool(r["feasible"]) for r in rows],dtype=bool)
    raw=np.asarray([np.nan if r["raw_loss_mw"] is None else float(r["raw_loss_mw"]) for r in rows],dtype=float)
    return {
        "count":len(rows),"feasible_count":int(feasible.sum()),"feasibility_rate":float(feasible.mean()) if len(rows) else 0.0,
        "mean_penalized_objective":float(values.mean()) if len(rows) else None,
        "mean_raw_loss_mw_feasible":float(np.nanmean(raw[feasible])) if np.any(feasible) else None,
        "worst_raw_loss_mw_feasible":float(np.nanmax(raw[feasible])) if np.any(feasible) else None,
    }


@dataclass(frozen=True)
class EndpointProtocol:
    validation_scenarios: int
    ood_scenarios: int
    validation_seed: int
    ood_seed: int
    bess_power_mw: float
    n1_line_indices: tuple[int,...]
    confirmation_scenarios: int = 64
    confirmation_seed: int = 20260847
    pf_solver: str = "pandapower.runpp"
    voltage_bounds_pu: tuple[float,float] = (0.90,1.10)
    loading_limit_percent: float = 100.0

    def as_dict(self)->dict[str,Any]:
        d=asdict(self); d["n1_line_indices"]=list(self.n1_line_indices); return d
    def fingerprint(self)->str: return _json_hash(self.as_dict())


def build_endpoint_protocol(project_root: str|Path) -> EndpointProtocol:
    root=Path(project_root)
    cfg=load_config([root/"config"/"base.yaml",root/"config"/"case300.yaml",root/"config"/"emulator_gpu.yaml"])
    data=load_case300_mode_a(root/"data"/"case300_full_modeA_Barskyi_Serhii.json")
    lines=tuple(int(x) for x in data.params["safe_n1_lines"])
    return EndpointProtocol(
        validation_scenarios=int(cfg.opf.validation_scenarios),ood_scenarios=int(cfg.opf.ood_scenarios),
        validation_seed=int(cfg.random.scenario_seed)+10,ood_seed=int(cfg.random.scenario_seed)+20,
        bess_power_mw=25.0,n1_line_indices=lines,
        confirmation_scenarios=64,confirmation_seed=int(cfg.random.scenario_seed)+30,
    )


def build_endpoint_partitions(project_root: str|Path, protocol: EndpointProtocol) -> tuple[Any,ScenarioBatch,ScenarioBatch]:
    root=Path(project_root); cfg=load_config([root/"config"/"base.yaml",root/"config"/"case300.yaml",root/"config"/"emulator_gpu.yaml"])
    loaded=load_power_case("case300")
    validation=generate_scenarios(loaded,ScenarioGeneratorConfig(n_scenarios=protocol.validation_scenarios,seed=protocol.validation_seed,load_mean=1.0,load_sigma=0.08,renewable_mean=1.0,renewable_sigma=0.20,load_bounds=(cfg.opf.train_load_scale_min,cfg.opf.train_load_scale_max),renewable_bounds=(0.0,2.0)),metadata={"partition":"validation","contract":"v38"})
    ood=generate_scenarios(loaded,ScenarioGeneratorConfig(n_scenarios=protocol.ood_scenarios,seed=protocol.ood_seed,load_mean=(cfg.opf.ood_load_scale_min+cfg.opf.ood_load_scale_max)/2.0,load_sigma=0.06,renewable_mean=0.75,renewable_sigma=0.25,load_bounds=(cfg.opf.ood_load_scale_min,cfg.opf.ood_load_scale_max),renewable_bounds=(0.0,1.5)),metadata={"partition":"ood","contract":"v38"})
    return loaded,validation,ood


def evaluate_selected_buses(project_root: str|Path, selected_buses: tuple[int,...]|list[int], *, method: str, power_mw: float=25.0, protocol: EndpointProtocol|None=None, placement_fingerprint: str|None=None) -> dict[str,Any]:
    protocol=build_endpoint_protocol(project_root) if protocol is None else protocol
    buses=tuple(int(x) for x in selected_buses)
    if not buses or len(set(buses))!=len(buses): raise ApplicationEndpointError("selected_buses must be non-empty and unique")
    loaded,validation,ood=build_endpoint_partitions(project_root,protocol)
    validation_rows=_scenario_rows(buses,float(power_mw),validation,loaded,"validation")
    ood_rows=_scenario_rows(buses,float(power_mw),ood,loaded,"ood")
    n1_rows=_n1_rows(buses,float(power_mw),loaded,protocol.n1_line_indices)
    payload={
        "schema":"CSSF-APPLICATION-ENDPOINT-v38","method":str(method),"selected_buses":list(buses),
        "bess_power_mw":float(power_mw),"placement_fingerprint":placement_fingerprint or _json_hash({"selected_buses":list(buses),"power_mw":float(power_mw)}),"verification_protocol":protocol.as_dict(),"verification_protocol_sha256":protocol.fingerprint(),
        "validation":{"summary":_summary(validation_rows),"rows":validation_rows,"batch_fingerprint":validation.fingerprint()},
        "ood":{"summary":_summary(ood_rows),"rows":ood_rows,"batch_fingerprint":ood.fingerprint()},
        "n1":{"summary":_summary(n1_rows),"rows":n1_rows,"line_indices":list(protocol.n1_line_indices)},
        "ac_complete":True,"ood_complete":True,"n1_complete":len(n1_rows)==len(protocol.n1_line_indices),"n1_line_indices":list(protocol.n1_line_indices),
    }
    payload["independent_rerun_fingerprint"]=_json_hash(payload)
    return payload


def evaluate_serialized_placement(project_root: str|Path, placement: BESSPlacement, *, method: str, protocol: EndpointProtocol|None=None) -> dict[str,Any]:
    return evaluate_selected_buses(project_root,tuple(int(x) for x in placement.selected_buses),method=method,power_mw=float(placement.fleet.unit.power_mw),protocol=protocol,placement_fingerprint=_placement_fingerprint(placement))



def select_placement_on_validation(project_root: str|Path, placements: Mapping[str,BESSPlacement], *, protocol: EndpointProtocol|None=None) -> tuple[str,dict[str,Any]]:
    """Select one candidate placement using validation AC only; OOD/N-1 stay untouched."""
    if not placements: raise ApplicationEndpointError("At least one candidate placement is required")
    protocol=build_endpoint_protocol(project_root) if protocol is None else protocol
    loaded,validation,_=build_endpoint_partitions(project_root,protocol)
    rows=[]
    for key,placement in placements.items():
        result=_scenario_rows(tuple(map(int,placement.selected_buses)),float(placement.fleet.unit.power_mw),validation,loaded,"validation")
        summary=_summary(result)
        rows.append({"candidate_id":str(key),"selected_buses":list(map(int,placement.selected_buses)),"summary":summary,"validation_batch_fingerprint":validation.fingerprint(),"placement_fingerprint":_placement_fingerprint(placement)})
    rows.sort(key=lambda r:(-float(r["summary"]["feasibility_rate"]),float("inf") if r["summary"]["mean_raw_loss_mw_feasible"] is None else float(r["summary"]["mean_raw_loss_mw_feasible"]),r["candidate_id"]))
    winner=rows[0]
    return str(winner["candidate_id"]),{"schema":"CSSF-VALIDATION-PLACEMENT-SELECTION-v38","verification_protocol_sha256":protocol.fingerprint(),"winner":winner,"candidates":rows,"selection_partition":"validation","ood_n1_used":False}


def evaluate_confirmation_buses(
    project_root: str|Path, selected_buses: tuple[int,...]|list[int], *, method: str,
    power_mw: float=25.0, protocol: EndpointProtocol|None=None,
) -> dict[str,Any]:
    """Independent paired AC confirmation partition for application claims.

    This partition is disjoint from validation and OOD through a frozen seed and
    is never used for control selection, surrogate fitting or hyperparameter
    choice.  It is designed for V14 paired application-level confirmation.
    """
    protocol=build_endpoint_protocol(project_root) if protocol is None else protocol
    root=Path(project_root); cfg=load_config([root/"config"/"base.yaml",root/"config"/"case300.yaml",root/"config"/"emulator_gpu.yaml"])
    loaded=load_power_case("case300")
    batch=generate_scenarios(loaded,ScenarioGeneratorConfig(
        n_scenarios=int(protocol.confirmation_scenarios),seed=int(protocol.confirmation_seed),
        load_mean=1.02,load_sigma=0.09,renewable_mean=0.95,renewable_sigma=0.22,
        load_bounds=(cfg.opf.train_load_scale_min,cfg.opf.train_load_scale_max),renewable_bounds=(0.0,2.0),
    ),metadata={"partition":"confirmation","contract":"v38"})
    buses=tuple(map(int,selected_buses)); rows=_scenario_rows(buses,float(power_mw),batch,loaded,"confirmation")
    return {"schema":"CSSF-APPLICATION-CONFIRMATION-v38","method":str(method),"selected_buses":list(buses),
            "bess_power_mw":float(power_mw),"rows":rows,"summary":_summary(rows),
            "batch_fingerprint":batch.fingerprint(),"seed":int(protocol.confirmation_seed),
            "count":int(protocol.confirmation_scenarios),"verification_protocol_sha256":protocol.fingerprint()}

def run_application_endpoint(project_root: str|Path, placements: Mapping[str,BESSPlacement], *, output_path: str|Path|None=None, selection_input_ids: list[str]|None=None) -> dict[str,Any]:
    if not placements: raise ApplicationEndpointError("At least one placement is required")
    protocol=build_endpoint_protocol(project_root)
    rows=[evaluate_serialized_placement(project_root,p,method=name,protocol=protocol) for name,p in placements.items()]
    protected=[f"ood:scenario_{i:06d}" for i in range(protocol.ood_scenarios)]+[f"n1:line:{x}" for x in protocol.n1_line_indices]
    out={"schema":"CSSF-APPLICATION-ENDPOINT-COMPARISON-v38","verification_protocol":protocol.as_dict(),"verification_protocol_sha256":protocol.fingerprint(),"placements":rows,"protected_partition_ids":protected,"selection_input_ids":list(selection_input_ids or []),"full_n1_count":len(protocol.n1_line_indices)}
    if output_path is not None:
        path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2,sort_keys=True,default=str),encoding="utf-8")
    return out


__all__=["ApplicationEndpointError","EndpointProtocol","build_endpoint_protocol","build_endpoint_partitions","evaluate_selected_buses","evaluate_serialized_placement","evaluate_confirmation_buses","select_placement_on_validation","run_application_endpoint"]
