# ⚛️ Quantum Annealing Trigonometrization Framework

**CSSF(QA)** is the quantum-annealing specialization of the [Complex Spectral Surrogate Framework (CSSF)](https://github.com/Serhii-Barskyi/CSSF-Complex-Spectral-Surrogate-Framework), extending its complex-Hermitian spectral modeling line to quantum-annealing trigonometrization, calibration-resolved response identification, surrogate modeling, and D-Wave QPU / GPU-SQA experiments.

Its complex-Hermitian core develops from the Aizenberg MVN/MLMVN complex-valued neural-network line and is extended here to physically structured quantum-annealing control and response spaces.

### From physical periodicity to controllable probability of application-useful quantum-annealing outcomes

The framework goes beyond constructing a BQM/QUBO and sending it to a solver. It introduces a structured intelligence layer between the application and the hardware experiment:

```text
application phase geometry
→ BQM / Ising
→ annealing controls
→ spectral QPU-response identification
→ elite-solution probability
→ independently confirmed application value
```

The practical goal is to determine **which physically admissible annealing controls increase the probability of obtaining solutions that remain valuable after independent application-level verification**.

For D-Wave infrastructure, this creates a candidate **annealing-control intelligence layer** around programmable schedules, solver/calibration context, SampleSet distributions, and downstream validation. The intended value is measurable: higher elite probability, fewer control experiments or reads to reach a target, better reproducibility, and lower **cost-to-confirmed-target**.

---

## 🧭 Quantum Annealing Trigonometrization

CSSF(QA) is built on the methodology developed in the monograph **[Quantum Annealing Trigonometrization](https://www.linkedin.com/in/serhii-barskyi/)**.

The central representation principle is simple: variables with genuine periodic or phase geometry should not be flattened unnecessarily into ordinary Euclidean coordinates. CSSF preserves that structure with complex harmonic features such as

$$
\Phi_{\omega}(\theta)=e^{i\omega^\top\theta}.
$$

For quantum annealing, the same idea is applied to the control process itself. An executable schedule $s(t)$ is combined with calibration-resolved D-Wave functions $A(s),B(s)$, producing accumulated operator-action coordinates

$$
\beta_k=\frac{1}{\hbar}\int_{I_k}A(s(t))\,dt,
\qquad
\gamma_k=\frac{1}{\hbar}\int_{I_k}B(s(t))\,dt.
$$

CSSF(QA) can therefore represent an annealing control through the **physical action accumulated by the driver and problem Hamiltonians**, not only through the geometric shape of the schedule. A Hermitian complex-spectral model then identifies the multi-output response of the annealer over that physically admissible control space.

---

## Why this fits D-Wave infrastructure

A D-Wave experiment is more than one BQM. Its physical context can include the programmable schedule, solver and working graph, calibration-resolved $A(s),B(s)$, embedding, chain strength, gauge policy, SampleSet distribution, and hardware timing.

CSSF(QA) treats these elements as part of one response-identification and control problem. Instead of asking only

> *Which sample has the minimum proxy energy?*

it asks

> **Which admissible control makes the QPU more likely to return solutions that are useful after independent physical or application validation?**

The response vector can include

$$
P_{\mathrm{elite}},\quad p_\gamma,\quad P_{\mathrm{feas}},\quad \mathrm{CVaR}_\alpha,\quad U_{\mathrm{application}}.
$$

The resource objective is correspondingly broader than energy-only optimization:

$$
C=(N_{\mathrm{controls}},N_{\mathrm{reads}},T_{\text{QPU-access}},N_{\text{physical validations}},T_{\mathrm{classical}}).
$$

The validation hypothesis is

```text
better physical representation
→ better response identification
→ higher elite probability
→ fewer control experiments / reads
→ lower cost-to-confirmed-target
```

This is the main reason the framework is relevant to D-Wave users: it is designed to turn a programmable annealing control surface into a reproducible, application-aware experimental system rather than a sequence of isolated parameter sweeps.

---

## ⚡ BESS placement: a physically verifiable reference case

The first major application reference is **Battery Energy Storage System placement on the IEEE 300-bus system (`case300`)**.

Here the quantum result is evaluated **outside the QUBO**:

```text
QPU distribution
→ elite BESS portfolio
→ nonlinear AC verification
→ OOD validation
→ N-1 contingencies
→ engineering value
```

The preserved real-D-Wave case300 protocol produced the following downstream results:

| BESS budget | Metric | CSSF(QA D-Wave) | HiGHS reference |
|---:|---|---:|---:|
| 2 | Mean AC loss reduction | **7.0608 MW** | 5.4441 MW |
| 2 | Mean OOD | **9.9588 MW** | 7.7255 MW |
| 2 | N-1 mean | **3.5296 MW** | 2.6800 MW |
| 3 | Mean AC loss reduction | **9.7032 MW** | 9.0321 MW |
| 3 | Mean OOD | **13.6503 MW** | 12.7659 MW |
| 3 | N-1 mean | **4.8825 MW** | 4.4900 MW |

These numbers are **fixed case300 application/portfolio evidence**. They are not a claim of universal quantum speedup, and they do not by themselves prove superiority of the newer annealing-control surrogate layer.

The $B=3$ result is particularly informative: the physically strong QPU placement `[236, 57, 90]` did not coincide with the brute-force QUBO optimum `[236, 57, 281]`. The experiment therefore illustrates why **a quantum portfolio followed by independent physical selection can be more useful than treating one proxy-energy ground state as the entire engineering objective**.

---

## 🔬 What CSSF(QA) provides

CSSF(QA) is intended for problems where the cost of discovering a good control is itself significant. The framework provides infrastructure for:

- calibration-resolved QPU-response identification;
- elite-probability and feasibility modeling;
- information-efficient next-control selection;
- independent confirmatory sampling;
- application-aware validation rather than energy-only scoring;
- residual and transfer modeling across related calibration contexts;
- reproducible provenance and cost-to-target accounting.

The framework is deliberately **fail-closed**. If the surrogate, fidelity, data, hardware, or physical-validation gate is not passed, the corresponding claim remains blocked.

The currently frozen hardware-validation pair is Pegasus `Advantage_system4` / `Advantage_system6`. 

---

## 📘 Scientific foundation

### [Quantum Annealing Trigonometrization](https://www.linkedin.com/in/serhii-barskyi/)

The monograph develops the mathematical and physical foundation of CSSF(QA), including complex/Hermitian spectral modeling, toric application geometry, AC sensitivity, BESS placement, QUBO/Ising construction, continuous and digitized quantum annealing, D-Wave Pegasus semantics, calibration-resolved operator-action coordinates, statistical validation, and independent physical confirmation.

Its central evidence chain is

```text
quantum response
→ spectral geometry
→ elite portfolio
→ physical verification
→ engineering decision
```

---

## Two reproducible computational protocols

The repository intentionally contains **two different notebooks**. Their evidence must not be conflated.

### 1. `CSSF_dwave_case300.ipynb` - Monograph / Real-QPU Evidence

This is the public execution adaptation of the case300 notebook associated with the monograph. 

```text
case300 AC data → CSNN-T → candidate screening → QUBO/Ising
→ real D-Wave Pegasus QPU → quantum portfolio → HiGHS reference
→ AC/OOD/N-1 confirmation
```

**[Open the case300 notebook](cssf_dwave/notebooks/CSSF_dwave_case300.ipynb)**

### 2. `CSSF_QA_DWave_Evidence_Simulator_v56.ipynb` - Scalable Framework Evidence

This is the separate scalable CSSF(QA) validation program: analytical CPU preflight, Pegasus-constrained GPU/SQA, matched control competitors, independent production confirmation, residual/transfer experiments, and application gates.

```text
real-QPU case300 evidence != GPU/SQA framework evidence
```

**[Open the scalable evidence notebook](cssf_dwave/notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb)**

---

## Broader application program

The same representation-and-control principle is being investigated across four application families:

**case300 BESS / AC** - preserved physical reference;  
**IEEE-33 resilience** - loss-to-resilience transfer and resilience-aware formulation;  
**Few-FEM outer-rotor BLDC** - periodic electromagnetic surrogate plus discrete manufacturable design;  
**EEG phase / microstate syntax** - native analytic phase plus separately validated discrete syntax.

The last three are **validation programs, not completed application-evidence claims** in this public release.

---

## Where the approach is most applicable

CSSF(QA) is most natural when a problem has genuine periodic/phase/cyclic geometry, admits a BQM/QUBO/Ising bridge, benefits from the annealer's distribution of candidate states, has an independent high-fidelity evaluator, and makes QPU experimentation expensive enough that data-efficient control selection matters.

$$
\boxed{\text{CSSF(QA)}=\text{physics-informed trigonometrization}+\text{response intelligence}+\text{quantum portfolio}+\text{physical confirmation}}
$$

---

## 🧩 Industry optimization use cases

These examples cover discrete and combinatorial optimization problems that can be formulated for the D-Wave quantum-classical stack using [quantum annealing](https://docs.dwavequantum.com/en/latest/quantum_research/quantum_annealing_intro.html), [QUBO / Ising models](https://docs.dwavequantum.com/en/latest/quantum_research/qubo_ising.html), and [hybrid CQM / nonlinear solvers](https://docs.dwavequantum.com/en/latest/industrial_optimization/index_hybrid.html). Repeated industry references are consolidated below so that each sector appears only once.

- **Energy, utilities, and power systems:** unit commitment; BESS siting, sizing, and charge/discharge scheduling; microgrid coordination; generation-reserve allocation; grid observability and islanding; maintenance scheduling; robust post-fault restoration; and multi-objective optimization of CAPEX, losses, voltage deviation, risk, and reliability.

- **Manufacturing and industrial operations:** production-process configuration; lot sizing; Job-Shop and Flow-Shop scheduling; industrial-robot and AGV coordination; assignment of jobs to equipment; production-cell formation; batch and flow planning; line activation and output planning; predictive-maintenance feature selection; and optimization under uncertain demand or equipment failures.

- **Logistics, supply chain, and warehousing:** vehicle routing; fleet sizing; container-operation sequencing; courier and order assignment; warehouse and distribution-center location; bin packing and load consolidation; inventory optimization; spare-parts logistics; multi-echelon supply-chain planning; and robust planning under supplier disruption or shortage risk.

- **Telecommunications and wireless networks:** 4G/5G network configuration; channel, spectrum, and bandwidth allocation; base-station selection for coverage; interference-free transmitter or link selection; edge-node and data-center placement; network partitioning; and graph-coloring formulations for frequency assignment.

- **Transportation, mobility, and aviation:** aircraft, flight, and crew scheduling; aircraft-to-flight assignment; baggage and cargo allocation; last-mile and field-service routing; public-transport and rail scheduling; road-network partitioning; time-slot allocation for conflicting operations; and schedules designed to remain feasible under delays or asset unavailability.

- **Healthcare and life sciences:** physician and nurse rostering; staffing-level optimization; assignment of patients to beds and procedures; operating-room, equipment, and personnel allocation; healthcare-facility location; medical-inventory planning; biomarker feature selection; donor-recipient matching; and constrained clinical scheduling.

- **Computing, cloud, data centers, and HPC:** server and GPU-node sizing; CPU/GPU/memory allocation; VM and container placement; assignment of compute jobs to servers; communication-graph partitioning; mesh and workload partitioning; and capacity-constrained packing of workloads onto physical infrastructure.

- **Semiconductors and electronics:** FPGA logical-block placement; VLSI partitioning; discrete component and circuit configuration; wafer and component inventory planning; semiconductor supply planning; and fault diagnosis through QUBO/Ising or constraint-satisfaction formulations.

- **Finance, banking, and investment management:** equity, ETF, bond, and credit-portfolio optimization; multi-period asset allocation; portfolio construction under Expected Shortfall or ESG constraints; CAPEX-constrained project selection; and robust portfolio selection under uncertain returns and correlations.

- **Retail, e-commerce, and marketing:** assortment selection under shelf-space constraints; warehouse and micro-fulfillment location; safety-stock optimization; demand-prediction feature selection; parcel and order packing; delivery-driver scheduling; order-to-provider matching; and campaign selection under fixed budgets.

- **Cybersecurity, software, and graph analytics:** minimum-risk security configuration; intrusion-detection feature selection; network segmentation; selection of security controls under budget constraints; minimum test-set selection for software coverage; SAT/CSP-based verification; community detection; knowledge-graph clustering; and critical-node analysis.

- **Aerospace, space, and defense:** satellite-observation scheduling; compatible observation selection under resource conflicts; payload selection under mass and volume constraints; spare-parts and MRO logistics; mission-resource configuration; and multi-level logistics for materials, components, and repair resources.

- **Agriculture and food systems:** coordination and routing of autonomous tractors and harvesters; food and perishables supply planning; cold-chain optimization; inventory planning for perishable raw materials and finished goods; and robust resource planning under operational uncertainty.

- **Infrastructure, construction, and smart cities:** selection of infrastructure assets and connections; allocation of cranes, excavators, trucks, and crews; construction scheduling; sensor placement for area coverage; facility-location models; and discrete planning of interconnected urban or utility assets.

- **Public services, emergency response, education, and events:** ambulance, fire-engine, and rescue-team allocation; evacuation routing; emergency-resource siting and reserve planning; exam scheduling under student conflicts; instructor-to-course and room assignment; and selection of mutually compatible events under shared-resource constraints.

Across these sectors, the recurring mathematical structures include *integer and mixed-integer optimization, routing, scheduling, assignment, resource allocation, knapsack, bin packing, set cover / set packing, facility location, supply-chain and inventory optimization, portfolio optimization, feature selection, graph optimization, Max-Cut, Maximum Independent Set, graph matching, graph coloring, constraint satisfaction, multi-objective optimization, and robust discrete optimization*.

---

## 🚀 Get started

- **[Quantum Annealing Trigonometrization](https://www.linkedin.com/in/serhii-barskyi/)**
- **[Open the adapted case300 real-QPU notebook](cssf_dwave/notebooks/CSSF_dwave_case300.ipynb)**
- **[Run the scalable GPU/SQA evidence program](cssf_dwave/notebooks/CSSF_QA_DWave_Evidence_Simulator_v56.ipynb)**
- **[Technical setup and Colab guide](cssf_dwave/README.md)**
- **[Public method and claim boundaries](cssf_dwave/docs/PUBLIC_METHOD_AND_CLAIM_BOUNDARIES.md)**

---

## 🙏 Acknowledgments

- [Professor Igor Aizenberg](https://scholar.google.com/citations?hl=en&user=ZjfN_9AAAAAJ)
- [Potomac Quantum Innovation Center](https://www.pqic.org/)
- [Aqora](https://aqora.io/)
- [Connected DMV](https://www.connecteddmv.org/)
- [D-Wave](https://www.dwavequantum.com/)
- [Google Colab](https://colab.research.google.com/)

---

## 📞 Contact

**Serhii Barskyi**  
Data Scientist (Spectral Methods) | Quantum Optimization: QUBO, QAOA, Quantum Annealing | Spectral Analysis | Fourier-based ML | Qiskit | Django REST | Smart Grid Energy Systems Planning | Data Science Mentor @ Preply

- [Preply](https://preply.com/en/tutor/7756455)
- [LinkedIn](https://www.linkedin.com/in/serhii-barskyi/)
- [Sigma Publishing](https://www.linkedin.com/company/sigma-publishinq)
- 🏆 [Kaggle Competition: Quantum Hybrid BESS Placement - CSSF vs HiGHS](https://www.kaggle.com/competitions/quantum-hybrid-bess-placement-cssf-vs-hi-ghs)

