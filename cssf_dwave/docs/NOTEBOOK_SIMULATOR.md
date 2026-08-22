# CSSF_QA_DWave_Evidence_Simulator_v56.ipynb — scalable framework runbook

## Role

`notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb` is the canonical scalable CSSF(QA) evidence environment. It is **not** the preserved real-QPU case300 notebook.

The notebook implements the locked sequence:

```text
repository/dependency gates
→ CPU analytical preflight
→ Pegasus-constrained GPU/SQA
→ matched response/control experiments
→ independent confirmation
→ application claim gates
```

## Canonical path

```text
/content/drive/MyDrive/cssf_dwave
```

## Execution boundary

The production simulator path requires NVIDIA CUDA and fails closed if the declared GPU backend is unavailable. GPU/SQA evidence must never be reported as D-Wave QPU evidence.

The current case300 non-QZero program is CPU-preflight cleared under its locked prospective design. QZero remains blocked without its declared pretraining corpus. IEEE-33, Few-FEM and EEG application-level claims remain blocked until their declared assets and fidelity gates exist and pass.
