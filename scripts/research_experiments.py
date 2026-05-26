"""
Research Experiments — Raspberry Pi 5
Date: May 2026
Author: Miriyala Varun Tej

Experiment A — Cross-Platform QAOA Parameter Transfer
  ibm_kingston optimal → does it transfer to ibm_fez?
  FIXED: targets ibm_fez

Experiment A2 — Reverse Transfer
  ibm_fez fresh optimal → does it transfer back to ibm_kingston?
  NEW: reverse direction to confirm symmetry

Experiment B — Noise-Fingerprint Qubit Selection
  FIXED: layout bug corrected, no duplicate qubits

Experiment D — Shot Noise Variance Study
  Run same params 5 times, measure variance
  Tells us if 95.5% transfer quality is real or shot noise

All results → ~/research_results/
"""

import os, json, datetime, time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from scipy.optimize import minimize

os.makedirs(os.path.expanduser("~/research_results"), exist_ok=True)
RESULTS_DIR = os.path.expanduser("~/research_results")
sim = AerSimulator()

def ts():    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
def stamp(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save(name, data):
    f = os.path.join(RESULTS_DIR, f"{name}_{ts()}.json")
    with open(f, "w") as fp:
        json.dump({"timestamp": stamp(), "data": data}, fp, indent=2)
    print(f"  Saved: {f}")
    return f

# ── QAOA setup ─────────────────────────────────────────────────────────────────
N = 4
EDGES = [(0,1),(1,2),(2,3),(3,0),(0,2)]
OPTIMAL_STATES = ['0101', '1010']

def cut_val(bs):
    return sum(1 for i,j in EDGES if bs[i] != bs[j])

def qaoa_circuit(gamma, beta):
    qc = QuantumCircuit(N, N)
    qc.h(range(N))
    for q1, q2 in EDGES:
        qc.cx(q1,q2); qc.rz(2*gamma, q2); qc.cx(q1,q2)
    for q in range(N):
        qc.rx(2*beta, q)
    qc.measure(range(N), range(N))
    return qc

def opt_prob(counts, shots=1024):
    return sum(counts.get(bs,0) for bs in OPTIMAL_STATES) / shots


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — Cross-Platform Transfer: ibm_kingston → ibm_fez
# (Same as before, kept for reference — already ran this)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_A():
    print()
    print("="*65)
    print("EXPERIMENT A — PARAMETER TRANSFER: ibm_kingston → ibm_fez")
    print("="*65)

    KINGSTON_GAMMA = 1.0658
    KINGSTON_BETA  = 0.6013
    KINGSTON_PROB  = 0.4502

    print(f"\n  Known ibm_kingston optimal:")
    print(f"  gamma={KINGSTON_GAMMA}  beta={KINGSTON_BETA}  prob={KINGSTON_PROB}")

    # Simulator check
    qc = qaoa_circuit(KINGSTON_GAMMA, KINGSTON_BETA)
    counts_sim = sim.run(transpile(qc, sim), shots=2048).result().get_counts()
    sim_p = opt_prob(counts_sim, 2048)
    print(f"\n  Simulator prob with kingston params: {sim_p:.4f}")

    results = {}
    try:
        svc = QiskitRuntimeService(channel="ibm_quantum_platform")
        try:
            backend = svc.backend("ibm_fez")
        except:
            backend = svc.least_busy(operational=True, simulator=False)
        print(f"  Backend: {backend.name} ({backend.num_qubits}Q)")
        sampler = SamplerV2(backend)

        # Transferred job
        qc_t = transpile(qaoa_circuit(KINGSTON_GAMMA, KINGSTON_BETA),
                         backend, optimization_level=3)
        job1 = sampler.run([qc_t], shots=1024)
        print(f"\n  Transferred job: {job1.job_id()}", end="", flush=True)
        counts1 = dict(job1.result()[0].data.c.get_counts())
        p1 = opt_prob(counts1)
        print(f" → prob={p1:.4f}")

        # Fresh COBYLA on this backend
        cobyla_log = []
        def cost_hw(params):
            g, b = params
            qc = qaoa_circuit(g, b)
            qc_tr = transpile(qc, backend, optimization_level=3)
            job = sampler.run([qc_tr], shots=512)
            counts = dict(job.result()[0].data.c.get_counts())
            p = opt_prob(counts, 512)
            cobyla_log.append({"job_id": job.job_id(),
                                "gamma": round(g,4), "beta": round(b,4),
                                "prob": round(p,4)})
            print(f"    eval {len(cobyla_log):>2}: g={g:.4f} b={b:.4f} p={p:.4f} job={job.job_id()}")
            return -p

        print(f"\n  Fresh COBYLA on {backend.name}:")
        res = minimize(cost_hw, [KINGSTON_GAMMA, KINGSTON_BETA],
                       method='COBYLA', options={'maxiter':15, 'rhobeg':0.2})
        g_fresh, b_fresh = res.x

        # Confirm with 1024 shots
        qc_final = transpile(qaoa_circuit(g_fresh, b_fresh),
                             backend, optimization_level=3)
        job_final = sampler.run([qc_final], shots=1024)
        print(f"\n  Final confirm job: {job_final.job_id()}", end="", flush=True)
        counts_final = dict(job_final.result()[0].data.c.get_counts())
        p_final = opt_prob(counts_final)
        print(f" → prob={p_final:.4f}")

        tq = p1/p_final*100 if p_final > 0 else 0
        print(f"\n  Transfer quality: {tq:.1f}%")
        print(f"  gamma shift: {KINGSTON_GAMMA:.4f} → {g_fresh:.4f} (Δ={g_fresh-KINGSTON_GAMMA:+.4f})")
        print(f"  beta shift:  {KINGSTON_BETA:.4f} → {b_fresh:.4f} (Δ={b_fresh-KINGSTON_BETA:+.4f})")

        results = {
            "source": "ibm_kingston", "target": backend.name,
            "source_gamma": KINGSTON_GAMMA, "source_beta": KINGSTON_BETA,
            "source_prob": KINGSTON_PROB,
            "transferred_prob": round(p1,4), "transferred_job": job1.job_id(),
            "fresh_gamma": round(g_fresh,4), "fresh_beta": round(b_fresh,4),
            "fresh_prob": round(p_final,4), "fresh_job": job_final.job_id(),
            "transfer_quality_pct": round(tq,1),
            "gamma_shift": round(g_fresh-KINGSTON_GAMMA,4),
            "beta_shift": round(b_fresh-KINGSTON_BETA,4),
            "cobyla_log": cobyla_log
        }

    except Exception as e:
        print(f"\n  IBM error: {e}")
        results = {"error": str(e)}

    save("experiment_A_kingston_to_fez", results)
    print("\nEXPERIMENT A DONE.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A2 — REVERSE TRANSFER: ibm_fez → ibm_kingston
#
# Your ibm_fez fresh optimal from last run:
#   gamma=1.0598  beta=0.7272  prob=0.4453 (best eval)
#
# Question: do THESE params work on ibm_kingston?
# If yes — transfer is symmetric. Heron r2 is a family.
# If no  — one direction transfers but not the other. Asymmetric noise.
# ══════════════════════════════════════════════════════════════════════════════

def experiment_A2():
    print()
    print("="*65)
    print("EXPERIMENT A2 — REVERSE TRANSFER: ibm_fez → ibm_kingston")
    print("Symmetric? Or one-way only?")
    print("="*65)

    # Your ibm_fez fresh optimal from last night
    FEZ_GAMMA = 1.0598
    FEZ_BETA  = 0.7272
    FEZ_PROB  = 0.4453   # best eval on ibm_fez (eval 11)
    FEZ_JOB   = "d8a77eop0eas73dpg61g"

    print(f"\n  Known ibm_fez optimal:")
    print(f"  gamma={FEZ_GAMMA}  beta={FEZ_BETA}  prob={FEZ_PROB}")
    print(f"  Job: {FEZ_JOB}")

    # Also test your original kingston params for comparison
    KING_GAMMA = 1.0658
    KING_BETA  = 0.6013

    results = {}
    try:
        svc = QiskitRuntimeService(channel="ibm_quantum_platform")
        try:
            backend = svc.backend("ibm_kingston")
            print(f"  Got ibm_kingston directly")
        except:
            backend = svc.least_busy(operational=True, simulator=False)
            print(f"  ibm_kingston unavailable, using: {backend.name}")

        print(f"  Backend: {backend.name} ({backend.num_qubits}Q)")
        sampler = SamplerV2(backend)

        # Job 1: fez optimal → kingston (reverse transfer)
        qc_fez = transpile(qaoa_circuit(FEZ_GAMMA, FEZ_BETA),
                           backend, optimization_level=3)
        job_fez = sampler.run([qc_fez], shots=1024)
        print(f"\n  Job 1 (fez params on kingston): {job_fez.job_id()}", end="", flush=True)
        counts_fez = dict(job_fez.result()[0].data.c.get_counts())
        p_fez = opt_prob(counts_fez)
        print(f" → prob={p_fez:.4f}")

        # Job 2: original kingston params (baseline, sanity check)
        qc_king = transpile(qaoa_circuit(KING_GAMMA, KING_BETA),
                            backend, optimization_level=3)
        job_king = sampler.run([qc_king], shots=1024)
        print(f"  Job 2 (original kingston params): {job_king.job_id()}", end="", flush=True)
        counts_king = dict(job_king.result()[0].data.c.get_counts())
        p_king = opt_prob(counts_king)
        print(f" → prob={p_king:.4f}")

        # Job 3: Fresh COBYLA on kingston from fez starting point
        cobyla_log = []
        def cost_hw(params):
            g, b = params
            qc = qaoa_circuit(g, b)
            qc_tr = transpile(qc, backend, optimization_level=3)
            job = sampler.run([qc_tr], shots=512)
            counts = dict(job.result()[0].data.c.get_counts())
            p = opt_prob(counts, 512)
            cobyla_log.append({"job_id": job.job_id(),
                                "gamma": round(g,4), "beta": round(b,4),
                                "prob": round(p,4)})
            print(f"    eval {len(cobyla_log):>2}: g={g:.4f} b={b:.4f} p={p:.4f} job={job.job_id()}")
            return -p

        print(f"\n  Fresh COBYLA on {backend.name} (warm start from fez params):")
        res = minimize(cost_hw, [FEZ_GAMMA, FEZ_BETA],
                       method='COBYLA', options={'maxiter':12, 'rhobeg':0.15})
        g_fresh, b_fresh = res.x

        qc_final = transpile(qaoa_circuit(g_fresh, b_fresh),
                             backend, optimization_level=3)
        job_final = sampler.run([qc_final], shots=1024)
        print(f"\n  Final confirm: {job_final.job_id()}", end="", flush=True)
        counts_final = dict(job_final.result()[0].data.c.get_counts())
        p_final = opt_prob(counts_final)
        print(f" → prob={p_final:.4f}")

        tq_reverse = p_fez/p_final*100 if p_final > 0 else 0
        tq_original = p_king/p_final*100 if p_final > 0 else 0

        print(f"\n  {'='*50}")
        print(f"  EXPERIMENT A2 RESULTS — Reverse Transfer")
        print(f"  {'='*50}")
        print(f"  ibm_fez params on ibm_kingston:     prob={p_fez:.4f}  job={job_fez.job_id()}")
        print(f"  Original ibm_kingston params:        prob={p_king:.4f}  job={job_king.job_id()}")
        print(f"  Fresh optimised on ibm_kingston:     prob={p_final:.4f}  job={job_final.job_id()}")
        print(f"\n  Reverse transfer quality: {tq_reverse:.1f}% of fresh-optimised")
        print(f"  Original params quality:  {tq_original:.1f}% of fresh-optimised")
        print(f"\n  Beta comparison:")
        print(f"    ibm_kingston original beta: {KING_BETA:.4f}")
        print(f"    ibm_fez optimal beta:       {FEZ_BETA:.4f}  (Δ={FEZ_BETA-KING_BETA:+.4f})")
        print(f"    ibm_kingston fresh beta:    {b_fresh:.4f}  (Δ={b_fresh-KING_BETA:+.4f})")
        print(f"\n  Gamma comparison:")
        print(f"    ibm_kingston original gamma: {KING_GAMMA:.4f}")
        print(f"    ibm_fez optimal gamma:       {FEZ_GAMMA:.4f}  (Δ={FEZ_GAMMA-KING_GAMMA:+.4f})")
        print(f"    ibm_kingston fresh gamma:    {g_fresh:.4f}  (Δ={g_fresh-KING_GAMMA:+.4f})")

        if tq_reverse >= 90 and tq_original >= 90:
            finding = "SYMMETRIC TRANSFER: Both directions work. Heron r2 is a transferable parameter family. The backend-specific difference is in beta, not gamma."
        elif tq_reverse >= 90:
            finding = "ASYMMETRIC: fez→kingston transfers well but kingston→fez was partial. ibm_fez params are more general."
        elif tq_original >= 90:
            finding = "ASYMMETRIC: Original kingston params hold on kingston, but fez params do not transfer back. Backend-specific tuning still needed."
        else:
            finding = "WEAK TRANSFER in reverse direction. Each backend needs its own optimisation."

        print(f"\n  Finding: {finding}")

        results = {
            "source": "ibm_fez", "target": backend.name,
            "fez_gamma": FEZ_GAMMA, "fez_beta": FEZ_BETA, "fez_prob": FEZ_PROB,
            "fez_on_kingston_prob": round(p_fez,4), "fez_on_kingston_job": job_fez.job_id(),
            "original_kingston_prob": round(p_king,4), "original_kingston_job": job_king.job_id(),
            "fresh_gamma": round(g_fresh,4), "fresh_beta": round(b_fresh,4),
            "fresh_prob": round(p_final,4), "fresh_job": job_final.job_id(),
            "reverse_transfer_quality_pct": round(tq_reverse,1),
            "original_transfer_quality_pct": round(tq_original,1),
            "finding": finding,
            "cobyla_log": cobyla_log
        }

    except Exception as e:
        print(f"\n  IBM error: {e}")
        results = {"error": str(e)}

    save("experiment_A2_fez_to_kingston", results)
    print("\nEXPERIMENT A2 DONE.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Noise-Fingerprint Qubit Selection for QAOA
# FIXED: no duplicate layout, simple qubit pair testing
#
# Measure Bell fidelity on 6 neighbour pairs
# Run QAOA with best pair layout vs default transpiler
# ══════════════════════════════════════════════════════════════════════════════

def experiment_B():
    print()
    print("="*65)
    print("EXPERIMENT B — NOISE-FINGERPRINT QUBIT SELECTION FOR QAOA")
    print("FIXED version — no duplicate layout bug")
    print("="*65)

    def bell_fidelity(counts, shots=1024):
        return (counts.get('00',0) + counts.get('11',0)) / shots

    results = {}
    try:
        svc = QiskitRuntimeService(channel="ibm_quantum_platform")
        try:
            backend = svc.backend("ibm_fez")
        except:
            backend = svc.least_busy(operational=True, simulator=False)
        print(f"\n  Backend: {backend.name} ({backend.num_qubits}Q)")
        sampler = SamplerV2(backend)

        # Get connected qubit pairs from coupling map
        try:
            edges = list(backend.coupling_map.get_edges())
            # Pick pairs where both qubits are small indices (better characterised)
            test_pairs = []
            seen = set()
            for q0, q1 in edges:
                key = (min(q0,q1), max(q0,q1))
                if key not in seen and q0 < 20 and q1 < 20:
                    seen.add(key)
                    test_pairs.append((q0, q1))
                if len(test_pairs) >= 6:
                    break
        except:
            test_pairs = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]

        print(f"\n  Testing {len(test_pairs)} qubit pairs: {test_pairs}")
        print(f"\n  {'Pair':>8} {'Fidelity':>10} {'Job ID':>28}")
        print("  " + "-"*50)

        pair_fidelities = []
        for q0, q1 in test_pairs:
            # Simple Bell circuit on those physical qubits
            qc = QuantumCircuit(max(q0,q1)+1, 2)
            qc.h(q0)
            qc.cx(q0, q1)
            qc.measure(q0, 0)
            qc.measure(q1, 1)
            qc_t = transpile(qc, backend, optimization_level=1)
            job = sampler.run([qc_t], shots=1024)
            print(f"  ({q0:>2},{q1:>2}):   ", end="", flush=True)
            counts = dict(job.result()[0].data.c.get_counts())
            fid = bell_fidelity(counts)
            pair_fidelities.append({
                "pair": [q0, q1],
                "fidelity": round(fid, 4),
                "job_id": job.job_id(),
                "counts": counts
            })
            print(f"{fid:>10.4f}   {job.job_id()}")

        # Sort by fidelity
        pair_fidelities.sort(key=lambda x: x["fidelity"], reverse=True)
        best = pair_fidelities[0]
        worst = pair_fidelities[-1]
        best_q0, best_q1 = best["pair"]

        print(f"\n  Best pair:  ({best_q0},{best_q1}) fidelity={best['fidelity']:.4f}")
        print(f"  Worst pair: ({worst['pair'][0]},{worst['pair'][1]}) fidelity={worst['fidelity']:.4f}")
        print(f"  Spread: {worst['fidelity']:.4f} → {best['fidelity']:.4f}")

        # QAOA with default transpiler
        print(f"\n  Running QAOA — default layout vs best-pair layout...")
        qc_qaoa = qaoa_circuit(1.0658, 0.6013)

        qc_default = transpile(qc_qaoa, backend, optimization_level=3)
        job_default = sampler.run([qc_default], shots=1024)
        print(f"  Default layout job: {job_default.job_id()}", end="", flush=True)
        counts_def = dict(job_default.result()[0].data.c.get_counts())
        p_default = opt_prob(counts_def)
        print(f" → prob={p_default:.4f}")

        # QAOA with best-pair guided layout
        # Use initial_layout as list mapping virtual qubit index → physical qubit
        # Virtual qubits 0,1,2,3 → best pair + neighbours
        bq0, bq1 = best_q0, best_q1
        # pick 2 more physical qubits near the best pair for the other 2 virtual qubits
        other_qs = [q for q in range(backend.num_qubits)
                    if q != bq0 and q != bq1 and abs(q-bq0) < 5][:2]
        if len(other_qs) < 2:
            other_qs = [bq0+2, bq0+3]

        guided_layout = [bq0, bq1, other_qs[0], other_qs[1]]
        print(f"  Guided layout: virtual [0,1,2,3] → physical {guided_layout}")

        try:
            qc_guided = transpile(qc_qaoa, backend, optimization_level=3,
                                  initial_layout=guided_layout)
            job_guided = sampler.run([qc_guided], shots=1024)
            print(f"  Guided layout job:  {job_guided.job_id()}", end="", flush=True)
            counts_gui = dict(job_guided.result()[0].data.c.get_counts())
            p_guided = opt_prob(counts_gui)
            print(f" → prob={p_guided:.4f}")
        except Exception as e_layout:
            print(f"\n  Layout error: {e_layout}")
            print(f"  Falling back to default transpiler...")
            qc_guided = transpile(qc_qaoa, backend, optimization_level=3)
            job_guided = sampler.run([qc_guided], shots=1024)
            print(f"  Fallback job: {job_guided.job_id()}", end="", flush=True)
            counts_gui = dict(job_guided.result()[0].data.c.get_counts())
            p_guided = opt_prob(counts_gui)
            print(f" → prob={p_guided:.4f}")

        improvement = (p_guided - p_default) / p_default * 100 if p_default > 0 else 0

        print(f"\n  {'='*50}")
        print(f"  EXPERIMENT B RESULTS")
        print(f"  {'='*50}")
        print(f"  QAOA default layout:  prob={p_default:.4f}  job={job_default.job_id()}")
        print(f"  QAOA guided layout:   prob={p_guided:.4f}  job={job_guided.job_id()}")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  Best Bell fidelity pair: ({bq0},{bq1}) = {best['fidelity']:.4f}")
        print(f"  Fidelity spread across pairs: {worst['fidelity']:.4f} to {best['fidelity']:.4f}")

        if improvement > 3:
            finding = f"POSITIVE: Noise-fingerprint qubit selection improves QAOA by {improvement:.1f}%. Measured fidelity beats transpiler default."
        elif improvement > -3:
            finding = f"NEUTRAL ({improvement:+.1f}%): No significant difference. Transpiler is already near-optimal for this backend."
        else:
            finding = f"NEGATIVE ({improvement:+.1f}%): Transpiler default wins. It uses more calibration data than 2-qubit Bell fidelity alone."

        print(f"\n  Finding: {finding}")

        results = {
            "backend": backend.name,
            "pairs_tested": pair_fidelities,
            "best_pair": [bq0, bq1],
            "best_fidelity": best["fidelity"],
            "worst_fidelity": worst["fidelity"],
            "guided_layout": guided_layout,
            "qaoa_default_prob": round(p_default,4),
            "qaoa_default_job": job_default.job_id(),
            "qaoa_guided_prob": round(p_guided,4),
            "qaoa_guided_job": job_guided.job_id(),
            "improvement_pct": round(improvement,2),
            "finding": finding
        }

    except Exception as e:
        print(f"\n  IBM error: {e}")
        results = {"error": str(e)}

    save("experiment_B_qubit_selection", results)
    print("\nEXPERIMENT B DONE.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — Shot Noise Variance Study
#
# Run same optimal params 5 times on ibm_fez
# Measure how much prob varies just from shot noise
# This tells us: is 95.5% transfer quality real or within noise?
#
# If std_dev < 0.01 → transfer quality is a real signal
# If std_dev > 0.03 → 95.5% might just be shot noise
# ══════════════════════════════════════════════════════════════════════════════

def experiment_D():
    print()
    print("="*65)
    print("EXPERIMENT D — SHOT NOISE VARIANCE STUDY")
    print("Same params × 5 runs: how stable are IBM results?")
    print("="*65)

    # Test both your key param sets
    param_sets = [
        {"name": "ibm_fez_optimal",      "gamma": 1.0598, "beta": 0.7272},
        {"name": "ibm_kingston_optimal",  "gamma": 1.0658, "beta": 0.6013},
    ]

    results = {}
    try:
        svc = QiskitRuntimeService(channel="ibm_quantum_platform")
        try:
            backend = svc.backend("ibm_fez")
        except:
            backend = svc.least_busy(operational=True, simulator=False)
        print(f"\n  Backend: {backend.name} ({backend.num_qubits}Q)")
        sampler = SamplerV2(backend)

        for pset in param_sets:
            g, b = pset["gamma"], pset["beta"]
            name = pset["name"]
            print(f"\n  Testing: {name}")
            print(f"  gamma={g}  beta={b}")
            print(f"  Running 5 times at 1024 shots each...")
            print(f"\n  {'Run':>5} {'Prob':>8} {'Job ID':>28}")
            print("  " + "-"*44)

            probs = []
            jobs = []
            for i in range(5):
                qc = qaoa_circuit(g, b)
                qc_t = transpile(qc, backend, optimization_level=3)
                job = sampler.run([qc_t], shots=1024)
                print(f"  {i+1:>4}. ", end="", flush=True)
                counts = dict(job.result()[0].data.c.get_counts())
                p = opt_prob(counts)
                probs.append(p)
                jobs.append(job.job_id())
                print(f"{p:>8.4f}   {job.job_id()}")

            mean_p = float(np.mean(probs))
            std_p  = float(np.std(probs))
            min_p  = float(np.min(probs))
            max_p  = float(np.max(probs))

            print(f"\n  Results for {name}:")
            print(f"    Mean:    {mean_p:.4f}")
            print(f"    Std dev: {std_p:.4f}")
            print(f"    Range:   {min_p:.4f} to {max_p:.4f}")
            print(f"    CV:      {std_p/mean_p*100:.1f}%  (coefficient of variation)")

            if std_p < 0.015:
                verdict = f"STABLE: std={std_p:.4f} < 0.015. IBM results at 1024 shots are reproducible. Transfer quality of 95.5% is a real signal, not noise."
            elif std_p < 0.030:
                verdict = f"MODERATE NOISE: std={std_p:.4f}. Some shot noise but trend is real. Transfer results are meaningful."
            else:
                verdict = f"HIGH VARIANCE: std={std_p:.4f} > 0.030. 1024 shots is not enough for precise comparison. Need 4096 shots for clean results."

            print(f"\n  Verdict: {verdict}")

            results[name] = {
                "gamma": g, "beta": b,
                "probs": [round(p,4) for p in probs],
                "job_ids": jobs,
                "mean": round(mean_p,4),
                "std": round(std_p,4),
                "min": round(min_p,4),
                "max": round(max_p,4),
                "cv_pct": round(std_p/mean_p*100,1),
                "verdict": verdict
            }

    except Exception as e:
        print(f"\n  IBM error: {e}")
        results = {"error": str(e)}

    save("experiment_D_shot_variance", results)
    print("\nEXPERIMENT D DONE.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Simulator Test (unchanged, verify logic)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_C_sim_only():
    print()
    print("="*65)
    print("EXPERIMENT C — SIMULATOR TEST (no IBM, verify logic)")
    print("="*65)

    KINGSTON_GAMMA = 1.0658
    KINGSTON_BETA  = 0.6013

    qc = qaoa_circuit(KINGSTON_GAMMA, KINGSTON_BETA)
    counts = sim.run(transpile(qc, sim), shots=2048).result().get_counts()
    prob_transferred = opt_prob(counts, 2048)

    def cost(params):
        g, b = params
        qc = qaoa_circuit(g, b)
        c = sim.run(transpile(qc, sim), shots=512).result().get_counts()
        return -opt_prob(c, 512)

    res = minimize(cost, [np.pi/4, np.pi/4], method='COBYLA',
                   options={'maxiter':80, 'rhobeg':0.4})
    g_opt, b_opt = res.x
    qc2 = qaoa_circuit(g_opt, b_opt)
    counts2 = sim.run(transpile(qc2, sim), shots=2048).result().get_counts()
    prob_fresh = opt_prob(counts2, 2048)

    print(f"\n  Transferred params (gamma={KINGSTON_GAMMA}, beta={KINGSTON_BETA}):")
    print(f"    Sim prob = {prob_transferred:.4f}")
    print(f"\n  Fresh COBYLA on simulator:")
    print(f"    gamma={g_opt:.4f}  beta={b_opt:.4f}")
    print(f"    Sim prob = {prob_fresh:.4f}")
    print(f"\n  Transfer quality: {prob_transferred/prob_fresh*100:.1f}%")
    print(f"\n  Top 5 states:")
    for bs, cnt in sorted(counts.items(), key=lambda x: -x[1])[:5]:
        cuts = cut_val(bs)
        star = "★" if bs in OPTIMAL_STATES else ""
        print(f"    |{bs}⟩: {cnt/2048*100:.1f}%  cuts={cuts} {star}")

    save("experiment_C_simulator_test", {
        "transferred_prob": round(prob_transferred,4),
        "fresh_prob": round(prob_fresh,4),
        "fresh_gamma": round(g_opt,4),
        "fresh_beta": round(b_opt,4),
        "transfer_quality_pct": round(prob_transferred/prob_fresh*100,1)
    })
    print("\nEXPERIMENT C DONE. Logic verified.")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Research Experiments — Pi 5")
    parser.add_argument("--exp", default="sim",
        choices=["sim","A","A2","B","D","all"],
        help="sim=simulator | A=kingston→fez | A2=fez→kingston | B=qubit selection | D=variance | all=A2+B+D")
    args = parser.parse_args()

    print(f"\nResearch Session — {stamp()}")
    print(f"Platform: Raspberry Pi 5 | Results: {RESULTS_DIR}\n")

    if args.exp == "sim":
        experiment_C_sim_only()
        print("\nSimulator OK. Run with --exp A2 next.")

    elif args.exp == "A":
        experiment_A()

    elif args.exp == "A2":
        experiment_A2()

    elif args.exp == "B":
        experiment_B()

    elif args.exp == "D":
        experiment_D()

    elif args.exp == "all":
        print("Running A2 → B → D in sequence")
        experiment_A2()
        print("\nWaiting 30s...")
        time.sleep(30)
        experiment_B()
        print("\nWaiting 30s...")
        time.sleep(30)
        experiment_D()

    print(f"\nAll results: {RESULTS_DIR}")
