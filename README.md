# QSimEdge — Quantum Computing Research from ARM Edge Devices

**170+ real IBM Quantum experiments. All job IDs verifiable at quantum.ibm.com.**

Raspberry Pi 5 (ARM Cortex-A76) orchestrating IBM Heron r2 quantum processors.  
NVIDIA Jetson Orin Nano (1024-core Ampere GPU) for hybrid GPU+QPU experiments.

---

## Core Findings

### Finding 1 — QAOA Parameter Transfer Works Within Same Hardware Generation
Parameters from ibm_kingston transfer to ibm_fez at **95.4% quality**.  
Statistical test: p=0.638 — cannot distinguish transferred from fresh optimisation.  
**Saves 15 IBM Quantum jobs per optimisation cycle.**

### Finding 2 — Topology Breaks Parameter Transfer
Ring → Chain: **50.5%** (gamma shifts +0.506 rad)  
Ring → Triangle: **55.5%** (gamma shifts +0.359 rad)  
**Graph topology requires fresh optimisation — parameters are topology-specific.**

### Finding 3 — Scale Breaks Parameter Transfer  
4-node → 6-node: **41.1%** transfer quality  
**Larger graphs need fresh optimisation from scratch.**

### Finding 4 — QAOA Performance Varies 17.7% by Time of Day
Bell noise correlates with QAOA advantage: r=−0.521  
Best time: **evening (post-IBM-calibration)** → QAOA opt prob ~0.44  
Worst time: **morning (post-overnight-drift)** → QAOA opt prob ~0.38  
**IBM calibration cycle directly affects algorithm performance.**

### Finding 5 — Qubit Fidelity Rankings Stable After 5 Weeks
AI-selected best qubit pair (1,3) from April 2026 is still best in May 2026.  
Absolute fidelity dropped ~0.02 but relative ranking preserved.  
**One-time qubit selection valid for months.**

### Finding 6 — ZNE Gives +12% on QAOA Circuits
Richardson ZNE (1x/2x/3x) extrapolation improves QAOA optimal probability by 12%.

---

## Hardware

| Device | Spec | Role |
|---|---|---|
| Raspberry Pi 5 8GB | ARM Cortex-A76, 2.4 GHz | Quantum orchestration, experiments |
| NVIDIA Jetson Orin Nano 8GB | ARM Cortex-A78AE, Ampere GPU | GPU+QPU hybrid, QEC decoder |
| IBM ibm_fez | 156Q Heron r2 | Primary quantum backend |
| IBM ibm_kingston | 156Q Heron r2 | Cross-backend transfer experiments |

---

## IBM Job IDs — Selected Results

| Experiment | Backend | Result | Job ID |
|---|---|---|---|
| QAOA 20-job sweep | ibm_kingston | r=0.9944 sim-HW | d7gvoh7b |
| Parameter transfer | ibm_fez | 95.4% quality, p=0.638 | d8agn7lg |
| Reverse transfer | ibm_kingston | 94.2% quality | d8ahlrgp |
| Graph topology chain | ibm_fez | 50.5% — fails | d8alrg2s |
| Graph topology triangle | ibm_fez | 55.5% — fails | d8alspas |
| 4→6 node scaling | ibm_fez | 41.1% — fails | d8arq3aa |
| ZNE on QAOA | ibm_fez | +12% improvement | d8b6ht4e |
| Fidelity drift study | ibm_fez | Ranking stable 5 weeks | d8bi972j |
| GPU+QPU hybrid | ibm_fez | 35.55% optimal | d7gad6tp |
| VQE H2 | ibm_kingston | −1.7886 Ha | d78c25jc |

**Total: 170+ IBM Quantum job IDs**

---

## Certification
Quantum Fundamentals Program — Qubitech, WISER, APSCHE | Feb 2026  
Top 4.6% of 65,000+ participants | ID: B9D7CBB2

*varuntej3626@gmail.com | github.com/varuntej0007/qsim-ibm-experiments*
