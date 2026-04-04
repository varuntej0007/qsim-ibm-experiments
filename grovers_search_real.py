"""
QSimEdge Grover's Search — Real Problem
Finding a specific item in an unsorted database
4 qubits = 16 items, target item = 1011
Classical: 8 queries average
Grover's: 3 queries (sqrt speedup)
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import json, datetime, time

TARGET = '1011'  # searching for this 4-bit item
n = 4            # 4 qubits = 16 items

print("="*60)
print("QSimEdge Grover's Search — 4-Qubit (16 items)")
print("="*60)
print(f"Database size: 2^{n} = {2**n} items")
print(f"Target item: |{TARGET}⟩")
print(f"Classical expected queries: {2**(n-1)} (average)")
print(f"Grover's expected queries: ~{int(np.pi/4*np.sqrt(2**n))} (√N * π/4)\n")

def build_oracle(n, target):
    """Oracle that marks the target state with a phase flip."""
    qc = QuantumCircuit(n)
    # Flip qubits where target bit is 0 (to make target = |111...1>)
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            qc.x(i)
    # Multi-controlled Z (phase flip on |111...1>)
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    # Unflip
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            qc.x(i)
    return qc

def build_diffuser(n):
    """Grover diffusion operator: 2|s><s| - I"""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))
    return qc

def build_grover(n, target, iterations):
    qc = QuantumCircuit(n, n)
    # Initial superposition
    qc.h(range(n))
    # Grover iterations
    oracle = build_oracle(n, target)
    diffuser = build_diffuser(n)
    for _ in range(iterations):
        qc = qc.compose(oracle)
        qc = qc.compose(diffuser)
    qc.measure(range(n), range(n))
    return qc

# Optimal iterations for 4 qubits = 3
n_iterations = 3
qc = build_grover(n, TARGET, n_iterations)

print("--- Grover's Circuit ---")
print(f"Iterations: {n_iterations}")
print(f"Circuit depth: {qc.depth()}")
print()

# ── SIMULATOR ──────────────────────────────────────────────────────────────
print("--- Simulator Run ---")
sim = AerSimulator()
sim_counts = sim.run(transpile(qc, sim), shots=1024).result().get_counts()
sim_sorted = sorted(sim_counts.items(), key=lambda x: -x[1])
sim_target_pct = sim_counts.get(TARGET, 0) / 1024 * 100
print(f"  Target |{TARGET}⟩ found in: {sim_target_pct:.1f}% of shots")
print(f"  Top results: {sim_sorted[:3]}")
print(f"  Classical random search would find in ~50% average\n")

# ── REAL HARDWARE ──────────────────────────────────────────────────────────
print("--- Real IBM Hardware ---")
service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.least_busy(operational=True, simulator=False)
print(f"  Using: {backend.name}")

qc_t = transpile(qc, backend, optimization_level=1)
print(f"  Transpiled depth: {qc_t.depth()}")

job = SamplerV2(backend).run([qc_t], shots=1024)
print(f"  Job ID: {job.job_id()}")
print("  Waiting...")

real_counts = dict(job.result()[0].data.c.get_counts())
real_sorted = sorted(real_counts.items(), key=lambda x: -x[1])
real_target_pct = real_counts.get(TARGET, 0) / 1024 * 100

print(f"\n{'='*60}")
print("GROVER'S SEARCH RESULTS")
print(f"{'='*60}")
print(f"{'Metric':<35} {'Simulator':>12} {'Real HW':>12}")
print("-"*60)
print(f"{'Target |'+TARGET+'> probability':<35} {sim_target_pct:>11.1f}% {real_target_pct:>11.1f}%")
print(f"{'Top result':<35} {sim_sorted[0][0]:>12} {real_sorted[0][0]:>12}")
print(f"{'Target found as top result?':<35} {'YES' if sim_sorted[0][0]==TARGET else 'NO':>12} {'YES' if real_sorted[0][0]==TARGET else 'NO':>12}")
print(f"{'Classical random (expected %)':<35} {'50.0%':>12} {'50.0%':>12}")
print(f"{'Quantum advantage (ratio)':<35} {sim_target_pct/50:>12.2f}x {real_target_pct/50:>12.2f}x")

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json.dump({"timestamp":ts,"target":TARGET,"n_qubits":n,"iterations":n_iterations,
           "backend":backend.name,"job_id":job.job_id(),
           "simulator":{"target_pct":round(sim_target_pct,2),"counts":dict(sim_sorted[:8])},
           "real_hardware":{"target_pct":round(real_target_pct,2),"counts":dict(real_sorted[:8])}},
          open(f"results/grovers_4q_{ts}.json","w"), indent=2)
print(f"\nJob ID: {job.job_id()}")
print("DONE.")
