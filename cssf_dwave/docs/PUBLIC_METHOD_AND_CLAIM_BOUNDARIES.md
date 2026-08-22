# CSSF(QA) public method and claim boundaries

## Scope and two evidence protocols

This repository contains two separate computational protocols:

1. `notebooks/CSSF_dwave_case300.ipynb` — the monograph companion for the preserved `K=27`, `B in {2,3}` **real-D-Wave QPU** case300 evidence chain.
2. `notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb` — the scalable `K=80`, `B=15` **GPU/SQA framework** and matched control-evidence program.

The numerical results of one protocol must not be transferred to the other. In particular:

```text
hardware case300 != GPU/SQA case300
```

## Canonical Google Colab repository path

```text
/content/drive/MyDrive/cssf_dwave
```

Both notebooks and all public runbooks preserve this hard path contract.

## Real-QPU hardware boundary

The monograph notebook requires an explicit `Advantage_system4` or `Advantage_system6` Pegasus solver, a runtime-provided Leap token, the matching calibration schedule, and an online QPU. It has no classical/simulator fallback for the hardware evidence path. Credentials are not stored in the repository.

A new QPU run may differ from retained evidence because solver working graph, calibration epoch, embedding and sampling distribution are physical parts of the experiment.

## DC-baseline correction

The frozen case300 input does not contain a measured matched `rho_dc_vs_ac` value. Public code therefore represents that baseline as unavailable (`None`) and does not infer `beats_DC` from comparison with an artificial zero.

## Historical support-package boundary

The preserved case300 notebook depended on external historical support modules that are not present in the supplied source artifacts. The public release therefore provides `case300_compat/` as a tested compatibility reconstruction of the required runtime API. This reconstruction is not represented as byte-identical to the missing support package.

Consequently, preserved historical outputs and fresh regenerated values are labeled separately. The original run recorded `rho_OOD = 0.9914438669`; the public compatibility layer, using an explicit flattened non-slack Pearson definition, regenerates approximately `0.99053` on CPU. This is a disclosed reproducibility difference, not a new claim of historical identity.

The same non-identity is visible in the reconstructed QUBO penalty scale: the compatibility layer currently derives a conservative `lam_pen ≈ 81.4358`, while the preserved historical notebook output recorded approximately `1079.86`. The reconstructed QUBO passes exact-cardinality checks, reproduces the CPU brute-force optima `[236,57]` for `B=2` and `[236,57,281]` for `B=3`, and passes the QUBO↔Ising energy identity to below `1e-8`; nevertheless, the penalty-scale difference means the release does **not** claim exact numerical reconstruction of the missing historical QUBO builder.

## Scalable simulator evidence boundary

The simulator notebook produces GPU/SQA evidence only after fail-closed repository, dependency, CUDA and CPU analytical-preflight gates. GPU/SQA evidence and real-QPU evidence are distinct evidence classes; emulator repeatability is not hardware reproducibility.

The current case300 non-QZero GPU/SQA program is preflight-cleared with its prospectively locked design. The QZero comparator remains unavailable without its declared pretraining corpus. IEEE-33, Few-FEM and EEG application branches require their declared real application assets before those branches can constitute application-level evidence.

## Scientific interpretation

Internal surrogate metrics, QUBO energy or execution time are not sufficient by themselves to establish application superiority. Claims are limited by held-out, fidelity, provenance, feasibility, physical-validation and fail-closed gates.

The retained case300 result is application/portfolio evidence for one fixed network/protocol. It is not a claim of universal quantum speedup or universal superiority of quantum annealing.
