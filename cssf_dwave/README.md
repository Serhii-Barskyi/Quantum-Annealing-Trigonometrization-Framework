# CSSF(QA) — Technical Repository Guide

> This is the **technical README** for the public framework package. For the technology overview, D-Wave relevance, BESS value proposition, and the *Quantum Annealing Trigonometrization* concept, see the [GitHub repository landing page](https://github.com/Serhii-Barskyi/Quantum-Annealing-Trigonometrization-Framework).

CSSF(QA) is the public reproducible implementation of the Complex Spectral Surrogate Framework for quantum annealing. The repository contains **two distinct computational protocols** with different evidence roles. They must not be interpreted as one experiment.

---

## 1. Which notebook should I run?

| Notebook | Purpose | Backend / evidence | Relationship to the monograph |
|---|---|---|---|
| [`notebooks/CSSF_dwave_case300.ipynb`](notebooks/CSSF_dwave_case300.ipynb) | Public security- and path-normalized execution adaptation of the preserved case300 BESS workflow: CSSF → QUBO/Ising → real D-Wave QPU → HiGHS reference → AC/OOD/N-1 verification | **Preserved real-D-Wave evidence + explicit live-QPU execution path** | **Yes. It is based on the case300 notebook accompanying _Quantum Annealing Trigonometrization_.** |
| [`notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb`](notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb) | Scalable CSSF(QA) validation program: analytical preflight, GPU/SQA, matched competitors, control-response experiments, transfer, and application gates | **GPU/SQA simulator evidence; not real-QPU evidence** | Uses and extends the methodology of the monograph, but is a separate computational protocol |

### Short rule

If you are reading the monograph and want to follow its **case300 / real-QPU evidence chain**, start with:

```text
notebooks/CSSF_dwave_case300.ipynb
```

If you want to investigate the **current scalable CSSF(QA) framework, CPU preflight, GPU/SQA, and matched control experiments**, use:

```text
notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb
```

Do not mix numerical results from the two protocols:

```text
real-QPU case300 evidence != GPU/SQA framework evidence
```

---

## 2. Canonical Google Colab path — do not change it

The public framework uses one fixed runtime root:

```text
/content/drive/MyDrive/cssf_dwave
```

After copying the framework directory to Google Drive, the layout must begin exactly as follows:

```text
/content/drive/MyDrive/cssf_dwave/
├── README.md
├── .github/
├── notebooks/
├── core/
├── config/
├── qubo/
├── qaoa/
├── dwave_backend/
├── opf/
├── bess/
├── benchmarks/
├── experiments_dwave/
├── calibration/
├── data/
├── tests/
├── docs/
└── releases/
```

Do not rename `cssf_dwave`, and do not move `core/`, `notebooks/`, `experiments_dwave/`, or other runtime directories outside it.

Both public notebooks use the same canonical root:

```python
Path('/content/drive/MyDrive/cssf_dwave')
```

---

## 3. Scientific foundation: Quantum Annealing Trigonometrization

The first notebook is directly associated with the monograph **_Quantum Annealing Trigonometrization_**, which develops the mathematical and physical basis of CSSF(QA):

- toric and phase geometry;
- complex-Hermitian spectral representations;
- CSNN-T / GCV identification;
- AC sensitivity and BESS placement;
- QUBO / Ising construction;
- continuous quantum annealing;
- calibration-resolved annealing coordinates;
- D-Wave hardware semantics;
- embedding, chain, and provenance requirements;
- statistical and physical confirmation gates.

**[Quantum Annealing Trigonometrization](https://www.linkedin.com/in/serhii-barskyi/)**

---

# 4. Notebook 1 — `CSSF_dwave_case300.ipynb`

## 4.1 Purpose

This notebook is the **case300 computational artifact associated with the monograph**.

It follows the application pipeline for Battery Energy Storage System placement on the IEEE 300-bus test system:

```text
AC power-system data
    ↓
CSNN-T periodic / spectral representation
    ↓
BESS candidate screening
    ↓
QUBO / Ising construction
    ↓
real D-Wave quantum annealing
    ↓
elite BESS portfolio
    ↓
HiGHS / MILP-DC reference
    ↓
nonlinear AC verification
    ↓
OOD + N-1 validation
```

The final result is not judged only by QUBO energy. Selected BESS placements undergo independent AC/OOD/N-1 physical verification.

### 4.1.1 Historical reproducibility boundary

The public notebook is a **security- and path-normalized execution adaptation** of the preserved notebook: the embedded credential was removed, the Colab root was normalized to `/content/drive/MyDrive/cssf_dwave`, and an unavailable DC baseline was changed to fail closed rather than being represented by an artificial zero.

The original external support modules referenced by the historical run are not present among the preserved source artifacts. The release therefore includes a tested `case300_compat/` module that reconstructs the narrow public runtime API required by the notebook. It is **not claimed to be a byte-for-byte or numerically identical reconstruction of the missing historical support package**.

The preserved value `rho_OOD = 0.9914438669` belongs to the historical run and its original external metric implementation. A CPU regeneration in this release, using the explicitly published flattened non-slack Pearson-correlation formula, gives approximately `0.99053`. This difference does not overwrite the preserved hardware evidence; it is disclosed as a reproducibility boundary. Any new live-QPU execution is a new hardware observation.

Likewise, the reconstructed QUBO builder produces `lam_pen ≈ 81.4358`, while the preserved historical output reported approximately `1079.86`. The CPU smoke path still reproduces brute-force optima `[236,57]` for `B=2` and `[236,57,281]` for `B=3`, and the QUBO↔Ising identity check remains below `1e-8`. The public compatibility path is therefore executable and auditable, but it is not presented as an exact numerical reconstruction of the missing historical QUBO support package.

## 4.2 Requirements

To reproduce the live-QPU branch you need:

- Google Colab or a compatible Python environment;
- the complete `cssf_dwave` directory at the canonical path;
- dependencies installed from the notebook/release requirements;
- valid D-Wave Leap credentials with access to the required QPU;
- the calibration assets included in the public release;
- the frozen case300 input under `data/`;
- an allowed solver that passes the notebook runtime gate.

The API token must never be stored in the repository.

## 4.3 Run procedure

1. Place the complete framework directory at:

   ```text
   /content/drive/MyDrive/cssf_dwave
   ```

2. Open:

   ```text
   /content/drive/MyDrive/cssf_dwave/notebooks/CSSF_dwave_case300.ipynb
   ```

3. Run cells **strictly from top to bottom**.

4. Do not skip repository, solver, calibration, or integrity gates.

5. Do not replace a real-QPU stage with a CPU/GPU sampler and interpret the fallback as hardware evidence.

## 4.4 Main notebook stages

<details>
<summary><strong>Show the detailed execution map</strong></summary>

| Stage | What happens | What is checked |
|---|---|---|
| Environment / dependencies | Mount Drive, install dependencies, import packages | Required environment and packages are available |
| D-Wave access | Resolve credentials and connect to the QPU | Connection targets an allowed D-Wave context |
| Repository gate | Verify required paths and assets | The framework package is complete |
| case300 data | Load the frozen dataset | Dataset and experiment inputs are available |
| Level 1 | CSNN-T spectral surrogate for AC-LSF | Periodic representation and held-out behavior |
| Level 2 | Candidate screening + QUBO + Ising | The discrete problem is constructed consistently |
| Diagnostics | Scaling, sensitivity, AC checks, higher-order diagnostics | Internal pipeline consistency |
| Hardware context | Solver + calibration schedule | Real QPU context and calibration artifact are identified |
| Level 3 | QA-response / periodic-control diagnostics | Control-model evidence is separated from application-surrogate evidence |
| Real QPU | D-Wave annealing for the BESS budgets | Hardware SampleSet / candidate portfolio |
| Classical reference | HiGHS / MILP-DC | Independent classical reference |
| Physical confirmation | AC / OOD comparison | Downstream engineering value |
| N-1 | Contingency sweep | Resilience boundary |
| Final integrity | Consolidated report | Final interpretation is allowed only after upstream gates |

</details>

## 4.5 Preserved case300 reference results

These values belong to the **preserved real-QPU case300 protocol** and serve as reference evidence for the `quantum portfolio -> physical verification` chain.

### BESS budget `B = 2`

CSSF/D-Wave placement:

```text
[236, 57]
```

HiGHS/MILP-DC reference:

```text
[182, 184]
```

| Scenario | CSSF(QA) / D-Wave [MW] | HiGHS reference [MW] |
|---|---:|---:|
| normal | **+3.3993** | +2.5665 |
| peak | **+4.9263** | +3.7589 |
| OOD 1.30 | **+7.6972** | +5.9300 |
| OOD 1.40 | **+12.2204** | +9.5209 |
| **mean** | **+7.0608** | +5.4441 |
| **mean OOD** | **+9.9588** | +7.7255 |

### BESS budget `B = 3`

CSSF/D-Wave placement:

```text
[236, 57, 90]
```

HiGHS/MILP-DC reference:

```text
[236, 182, 184]
```

| Scenario | CSSF(QA) / D-Wave [MW] | HiGHS reference [MW] |
|---|---:|---:|
| normal | **+4.7046** | +4.3130 |
| peak | **+6.8077** | +6.2836 |
| OOD 1.30 | **+10.5934** | +9.8551 |
| OOD 1.40 | **+16.7072** | +15.6766 |
| **mean** | **+9.7032** | +9.0321 |
| **mean OOD** | **+13.6503** | +12.7659 |

### N-1 contingency reference

| BESS budget | Metric | CSSF(QA) / D-Wave | HiGHS reference |
|---:|---|---:|---:|
| 2 | mean ΔL | **+3.5296** | +2.6800 |
| 2 | worst-case ΔL | **+3.3040** | +2.4358 |
| 3 | mean ΔL | **+4.8825** | +4.4900 |
| 3 | worst-case ΔL | **+4.5990** | +4.1740 |

### Correct interpretation

These numbers support **application/portfolio evidence** for the preserved hardware experiment. They do not imply that:

- every future D-Wave solver will return the same placements;
- every new calibration epoch will produce the same probabilities;
- the current CSSF annealing-control surrogate has already demonstrated superiority;
- the GPU/SQA experiment in Notebook 2 reproduces the real-QPU experiment;
- the case300 result automatically transfers to other grids, BESS budgets, or application domains.

---

# 5. Notebook 2 — `CSSF_QA_DWave_Evidence_Simulator_v56.ipynb`

## 5.1 Purpose

This notebook is the **scalable CSSF(QA) evidence and validation environment**.

It separates causal questions that must not be collapsed into one claim:

1. Does periodic/trigonometric application representation help?
2. Is the planned control experiment analytically identifiable before annealer execution?
3. Does a structured QA response emerge in GPU/SQA?
4. Do CSSF-guided controls improve matched annealing outcomes?
5. Does the effect survive independent application validation?
6. Can response knowledge transfer across related calibration/hardware contexts?

## 5.2 Execution environment

Production SQA stages are designed for **Google Colab with an NVIDIA CUDA GPU**.

Before the first GPU annealer call, a **CPU-only Analytical Preflight** checks, among other items:

- schedule admissibility;
- information geometry and identifiability;
- D0-D3 causal-contrast completeness;
- QUBO consistency;
- statistical resolution and read budgets;
- mandatory experiment gates.

CPU preflight is not a substitute for GPU evidence.

## 5.3 Canonical launch

Open:

```text
/content/drive/MyDrive/cssf_dwave/notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb
```

The notebook checks the canonical repository root before production stages.

Run cells top-to-bottom. Do not skip:

```text
repository integrity
→ dependency / CUDA gate
→ analytical preflight
→ simulator construction
→ matched experiments
→ independent confirmation
→ application gates
```

## 5.4 Main blocks

<details>
<summary><strong>Show the evidence-program structure</strong></summary>

### Block I / II — environment and application-domain evidence

- Colab / CUDA / dependency checks;
- frozen source/input integrity;
- canonical CSNN-T/GCV application evidence;
- representation diagnostics.

### Block III — integrated CSSF(QA) chain

```text
periodic application physics
→ CSSF representation
→ BESS / QUBO
→ calibration-resolved annealing controls
→ Analytical Preflight
→ GPU SQA response
→ CSSF response model
→ independently verified application endpoint
```

### Mandatory Analytical Preflight

It runs before GPU/SQA response generation and cannot be redefined after observing results.

### Block IV — matched control frontier

Contains causal-mechanism experiments, matched competitors, production confirmation, and economic/application metrics.

### Block V — four-task application program

- case300 AC/BESS;
- IEEE-33 resilience;
- Few-FEM outer-rotor BLDC;
- EEG phase/syntax microstates.

Each application branch has its own data, fidelity, and confirmation gates.

</details>

## 5.5 Current evidence boundary

### Available in the current case300 framework path

- CPU analytical preflight;
- case300 application/QUBO assets;
- GPU/SQA program for allowed non-QZero branches;
- matched control experiments under the frozen protocol;
- independent production sampling where the protocol requires it.

### Blocked until declared inputs are available

**QZero** must not run without its real pretraining corpus.

**IEEE-33**, **Few-FEM**, and **EEG** are not completed application evidence until their declared assets and fidelity gates are available and passed.

A blocked branch is a valid fail-closed result, not a reason to substitute synthetic or proxy evidence.

## 5.6 GPU/SQA is not real-QPU evidence

`local_sqa_gpu` / GPU path-integral SQA models QA response in the declared simulator infrastructure.

It must not be described as:

- a run on a real D-Wave QPU;
- a hardware-calibration result;
- a direct replication of the preserved case300 hardware experiment.

A hardware claim requires a separate live-QPU execution with an actual solver, working graph, calibration context, and runtime provenance.

---

# 6. Scientific distinction between Notebook 1 and Notebook 2

| Question | Notebook 1 | Notebook 2 |
|---|---|---|
| Primary role | Preserved monograph case study | Scalable framework validation |
| Backend | Real D-Wave QPU for the hardware branch | GPU/SQA simulator for production evidence stages |
| BESS scale / protocol | Preserved case300 hardware protocol | Separate scalable framework protocol |
| Primary result type | Quantum portfolio + AC/OOD/N-1 application evidence | Causal/matched validation of representation, QA-response, and control mechanisms |
| May numerical results be transferred between the two? | **No** | **No** |
| May SQA be called a hardware result? | — | **No** |
| Direct accompanying-notebook relationship to the monograph? | **Yes** | No; it is a separate framework program |

---

# 7. Repository integrity and reproducibility

The public release must preserve its declared scientific/runtime files and fingerprints.

Before interpreting results, verify:

- frozen source manifests;
- frozen input manifests;
- repository layout;
- dependency versions;
- selected solver/backend;
- calibration artifact;
- embedding/topology identity where applicable;
- QUBO/Ising fingerprints;
- output provenance;
- declared evidence-gate status.

Runtime success does not override a failed scientific gate.

---

# 8. D-Wave credentials and hardware access

Notebook 1 requires valid D-Wave Leap access to a hardware solver that passes its runtime gate.

Use the following rules:

- keep `DWAVE_API_TOKEN` only in Colab Secrets or environment variables;
- never save the token in notebook output;
- never commit `.env`, credential files, or local configuration containing secrets;
- set `CSSF_DWAVE_SOLVER_ID` explicitly to `Advantage_system4` or `Advantage_system6` for the frozen Pegasus protocol;
- do not silently substitute another topology family when the requested solver is unavailable;
- retain returned solver identity and QPU timing/provenance metadata.

---

# 9. Troubleshooting

## `Repository root not found`

Verify that this exact path exists:

```text
/content/drive/MyDrive/cssf_dwave
```

Do not add another directory level and do not rename `cssf_dwave`.

## `CUDA unavailable`

This is a hard gate for production GPU/SQA stages in Notebook 2. Select an NVIDIA GPU runtime in Colab. Do not use a CPU fallback for the claim-execution path.

## QZero unavailable

This is the expected fail-closed state when the declared real pretraining corpus is missing. Do not fabricate a synthetic replacement to fill a benchmark row.

## IEEE-33 / motor / EEG branch blocked

Check the application assets and the public claim-boundary document. A missing required input means `BLOCK_MISSING_ASSET`; it is not a defect to be hidden.

## QPU solver unavailable

Do not substitute another solver automatically. Preserve solver/topology identity or define a new experiment context explicitly.

---

# 10. Documentation map

```text
Repository root README.md
    public technology / research showcase (GitHub wrapper)

cssf_dwave/README.md
    this technical repository guide

docs/PUBLIC_METHOD_AND_CLAIM_BOUNDARIES.md
    scientific interpretation and public claim boundaries

docs/NOTEBOOK_CASE300.md
    detailed monograph / real-QPU notebook runbook

docs/NOTEBOOK_SIMULATOR.md
    detailed GPU/SQA evidence notebook runbook
```

---

# 11. Quick start

### A. Reproduce the monograph case300 workflow

```text
1. Copy the framework directory -> /content/drive/MyDrive/cssf_dwave
2. Open notebooks/CSSF_dwave_case300.ipynb
3. Provide D-Wave credentials at runtime
4. Run top-to-bottom
5. Preserve solver/QPU provenance
6. Inspect AC/OOD/N-1 confirmation
```

### B. Run the current scalable framework evidence program

```text
1. Copy the framework directory -> /content/drive/MyDrive/cssf_dwave
2. Start Google Colab with an NVIDIA GPU
3. Open notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb
4. Run top-to-bottom
5. Pass repository/CUDA/integrity gates
6. Pass CPU Analytical Preflight
7. Execute only cleared GPU/SQA branches
8. Preserve blocked branches as blocked
```

---

# 12. Further reading

- **Public CSSF(QA) showcase:** [GitHub repository landing page](https://github.com/Serhii-Barskyi/Quantum-Annealing-Trigonometrization-Framework)
- **Monograph:** [Quantum Annealing Trigonometrization](https://www.linkedin.com/in/serhii-barskyi/)
- **Monograph case300 notebook:** [`notebooks/CSSF_dwave_case300.ipynb`](notebooks/CSSF_dwave_case300.ipynb)
- **Framework simulator/evidence notebook:** [`notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb`](notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb)
- **Public claim boundaries:** [`docs/PUBLIC_METHOD_AND_CLAIM_BOUNDARIES.md`](docs/PUBLIC_METHOD_AND_CLAIM_BOUNDARIES.md)

---

## License

See [`LICENSE`](LICENSE).
