"""
QSimEdge QAOA on Real IBM Quantum Hardware
Same Max-Cut problem — now on real superconducting qubits
Comparing ideal simulation vs real hardware noise effects on QAOA
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import json, datetime, time

edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
n_qubits = 4

def cost_function(bitstring, edges):
    return sum(1 for i,j in edges if bitstring[i] != bitstring[j])

gamma = Parameter('γ')
beta  = Parameter('β')

def build_qaoa_circuit(n, edges, gamma, beta):
    qc = QuantumCircuit(n, n)
    for i in range(n): qc.h(i)
    for i,j in edges:
        qc.cx(i,j); qc.rz(2*gamma,j); qc.cx(i,j)
    for i in range(n): qc.rx(2*beta, i)
    qc.measure(range(n), range(n))
    return qc

# Best parameters from simulator (reuse)
G_OPT = 1.3963  # ~0.444*pi
B_OPT = 0.6981  # ~0.222*pi

print("="*60)
print("QSimEdge QAOA — Real IBM Quantum Hardware")
print("="*60)
print(f"Graph: {n_qubits} nodes, {len(edges)} edges")
print(f"Using pre-optimised params: gamma={G_OPT:.4f}, beta={B_OPT:.4f}")
print()

# ── 1. PERFECT SIMULATION ──────────────────────────────────────────────────
print("--- Step 1: Perfect Simulation (AerSimulator) ---")
qc = build_qaoa_circuit(n_qubits, edges, gamma, beta)
bound_qc = qc.assign_parameters({gamma: G_OPT, beta: B_OPT})
sim = AerSimulator()
sim_result = sim.run(transpile(bound_qc, sim), shots=1024).result()
sim_counts = sim_result.get_counts()
sim_sorted = sorted(sim_counts.items(), key=lambda x: -x[1])
print(f"  Top result: {sim_sorted[0][0]} with {sim_sorted[0][1]} counts")
print(f"  Cut value:  {cost_function(sim_sorted[0][0], edges)}")
print()

# ── 2. REAL IBM HARDWARE ───────────────────────────────────────────────────
print("--- Step 2: Real IBM Quantum Hardware ---")
service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.least_busy(operational=True, simulator=False)
print(f"  Using: {backend.name} ({backend.configuration().n_qubits}Q)")

qc_real = build_qaoa_circuit(n_qubits, edges, gamma, beta)
bound_real = qc_real.assign_parameters({gamma: G_OPT, beta: B_OPT})
qc_t = transpile(bound_real, backend, optimization_level=3)

print(f"  Transpiled depth: {qc_t.depth()}")
print(f"  Transpiled gates: {sum(qc_t.count_ops().values())}")
print(f"  Submitting to real hardware...")

sampler = SamplerV2(backend)
job = sampler.run([qc_t], shots=1024)
print(f"  Job ID: {job.job_id()}")
print("  Waiting for real hardware result...")

real_result = job.result()
real_counts = dict(real_result[0].data.c.get_counts())
real_sorted = sorted(real_counts.items(), key=lambda x: -x[1])

print(f"\n  Real hardware result:")
print(f"  Top result: {real_sorted[0][0]} with {real_sorted[0][1]} counts")
print(f"  Cut value:  {cost_function(real_sorted[0][0], edges)}")

# ── 3. COMPARISON ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("QAOA COMPARISON: Simulator vs Real Hardware")
print(f"{'='*60}")
print(f"{'Bitstring':<12} {'Sim Count':<12} {'Sim %':<10} {'Real Count':<12} {'Real %':<10} {'Cut'}")
print("-"*66)

all_strings = set(list(sim_counts.keys()) + list(real_counts.keys()))
all_strings = sorted(all_strings, key=lambda x: -sim_counts.get(x,0))

for bs in list(all_strings)[:8]:
    sc = sim_counts.get(bs, 0)
    rc = real_counts.get(bs, 0)
    cut = cost_function(bs, edges)
    print(f"{bs:<12} {sc:<12} {sc/1024*100:<10.1f} {rc:<12} {rc/1024*100:<10.1f} {cut}")

# Approximation ratios
sim_best_cut = cost_function(sim_sorted[0][0], edges)
real_best_cut = cost_function(real_sorted[0][0], edges)
classical_opt = 4  # known optimal

print(f"\nApproximation Ratios (vs classical optimal = {classical_opt} cuts):")
print(f"  Simulator:     {sim_best_cut}/{classical_opt} = {sim_best_cut/classical_opt:.3f}")
print(f"  Real hardware: {real_best_cut}/{classical_opt} = {real_best_cut/classical_opt:.3f}")

# Noise analysis
sim_optimal_pct = sum(v for k,v in sim_counts.items() if cost_function(k,edges)==classical_opt)/1024*100
real_optimal_pct = sum(v for k,v in real_counts.items() if cost_function(k,edges)==classical_opt)/1024*100
print(f"\nOptimal solution found in:")
print(f"  Simulator:     {sim_optimal_pct:.1f}% of shots")
print(f"  Real hardware: {real_optimal_pct:.1f}% of shots")
print(f"  Noise impact:  {sim_optimal_pct - real_optimal_pct:.1f}% reduction due to quantum noise")

# ── SAVE ───────────────────────────────────────────────────────────────────
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    "timestamp": ts,
    "problem": "Max-Cut QAOA p=1",
    "graph": {"nodes": n_qubits, "edges": edges},
    "classical_optimal_cuts": classical_opt,
    "parameters": {"gamma": G_OPT, "beta": B_OPT},
    "backend": backend.name,
    "job_id": job.job_id(),
    "simulator": {
        "top_result": sim_sorted[0][0],
        "top_cut": sim_best_cut,
        "approx_ratio": round(sim_best_cut/classical_opt, 4),
        "optimal_pct": round(sim_optimal_pct, 2),
        "counts": dict(sim_sorted[:10])
    },
    "real_hardware": {
        "top_result": real_sorted[0][0],
        "top_cut": real_best_cut,
        "approx_ratio": round(real_best_cut/classical_opt, 4),
        "optimal_pct": round(real_optimal_pct, 2),
        "counts": dict(real_sorted[:10])
    }
}
with open(f"results/qaoa_real_{ts}.json","w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: results/qaoa_real_{ts}.json")
print(f"Job ID: {job.job_id()} — verifiable at quantum.ibm.com")
print("\nDONE.")
