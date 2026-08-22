# CSSF_dwave_case300.ipynb — monograph / real-QPU runbook

## Role

`notebooks/CSSF_dwave_case300.ipynb` is the public companion notebook for the preserved case300 hardware protocol discussed in *Quantum Annealing Trigonometrization*.

It targets **real D-Wave Pegasus hardware only**. The allowed solver families are `Advantage_system4` and `Advantage_system6`. There is no simulator fallback in the hardware path.

## Canonical path

```text
/content/drive/MyDrive/cssf_dwave
```

## Credentials

Store these values in **Colab Secrets** (or environment variables outside version control):

```text
DWAVE_API_TOKEN
CSSF_DWAVE_SOLVER_ID
```

`CSSF_DWAVE_SOLVER_ID` must explicitly select `Advantage_system4` or `Advantage_system6` (a provider suffix is accepted). The repository contains **no token or credential value**.

## Runtime sequence

1. Mount Google Drive.
2. Install `requirements-case300.txt`.
3. Restart the Colab runtime, remount Drive, and continue.
4. Pass the explicit Leap/QPU/Pegasus gate.
5. Pass repository and calibration-asset checks.
6. Run Level-1 CSNN-T and candidate screening.
7. Construct and audit QUBO/Ising.
8. Load the matching System4/System6 calibration schedule from `calibration/`.
9. Run the optional control-response exploration and the real-QPU BESS sampling stage.
10. Compare the frozen QPU portfolio with the HiGHS reference only after independent nonlinear AC/OOD/N-1 verification.

## Important correction relative to the historical saved notebook

A missing matched DC-LSF baseline is represented as `None` / `not available`. It is **not** replaced by zero, and the notebook makes no CSNN-T-vs-DC superiority claim until such a baseline is actually computed on the same target/scenarios.

## Compatibility reconstruction

The original saved notebook referenced historical external support modules that are not available in the supplied release artifacts. `case300_compat/` provides the narrow public runtime API needed by the normalized notebook. It is validated for the frozen dataset, candidate order, QUBO/Ising consistency and fail-closed QPU routing, but it is not claimed to reproduce the missing support package byte-for-byte.

The retained `rho_OOD = 0.9914438669` is therefore labeled as an original-run result. Under the compatibility layer's explicit flattened non-slack Pearson metric, the CPU regeneration is approximately `0.99053`.

The same non-identity is visible in the reconstructed QUBO penalty scale: the compatibility layer currently derives a conservative `lam_pen ≈ 81.4358`, while the preserved historical notebook output recorded approximately `1079.86`. The reconstructed QUBO passes exact-cardinality checks, reproduces the CPU brute-force optima `[236,57]` for `B=2` and `[236,57,281]` for `B=3`, and passes the QUBO↔Ising energy identity to below `1e-8`; nevertheless, the penalty-scale difference means the release does **not** claim exact numerical reconstruction of the missing historical QUBO builder.

## Evidence boundary

The preserved numerical result belongs to the historical `K=27`, `B in {2,3}` real-QPU protocol. A new live run is a new hardware observation and can differ because solver working graph, calibration epoch, embedding and sampling are physical parts of the experiment.
