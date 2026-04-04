"""
QSimEdge QAOA — Max-Cut Problem
Quantum Approximate Optimisation Algorithm
Solving graph partitioning on AerSimulator then real IBM hardware

The Max-Cut Problem:
Given a graph with nodes and edges, divide nodes into two groups
such that the maximum number of edges cross between the groups.
Classical: NP-hard for large graphs
Quantum: QAOA provides approximate solution with quantum speedup
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from qiskit import transpile
import json, datetime, time

# ── THE PROBLEM: MAX-CUT ON A 4-NODE GRAPH ───────────────────────────────────
# Graph edges: (0,1), (1,2), (2,3), (3,0), (0,2)
# Goal: split 4 nodes into 2 groups to maximise edges between groups
# Classical optimal: known to be 4 cuts

edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
n_qubits = 4

print("="*60)
print("QSimEdge QAOA — Max-Cut Problem")
print("="*60)
print(f"Graph: {n_qubits} nodes, {len(edges)} edges")
print(f"Edges: {edges}")
print(f"Classical optimal cut: 4 edges\n")

def cost_function(bitstring, edges):
    """Count how many edges are cut by this bitstring partition."""
    cut = 0
    for i, j in edges:
        if bitstring[i] != bitstring[j]:
            cut += 1
    return cut

def brute_force_maxcut(n, edges):
    """Find optimal solution by checking all 2^n partitions."""
    best_cut = 0
    best_partition = None
    for i in range(2**n):
        bits = format(i, f'0{n}b')
        cut = cost_function(bits, edges)
        if cut > best_cut:
            best_cut = cut
            best_partition = bits
    return best_partition, best_cut

# ── CLASSICAL BRUTE FORCE SOLUTION ───────────────────────────────────────────
print("--- Classical Brute Force ---")
t0 = time.perf_counter()
best_partition, best_cut = brute_force_maxcut(n_qubits, edges)
t_classical = time.perf_counter() - t0
print(f"Optimal partition: {best_partition} (group 0: nodes with bit=0, group 1: nodes with bit=1)")
print(f"Maximum cuts: {best_cut}")
print(f"Classical time: {t_classical*1000:.3f}ms\n")

# ── BUILD QAOA CIRCUIT (p=1 layer) ────────────────────────────────────────────
gamma = Parameter('γ')  # problem unitary parameter
beta  = Parameter('β')  # mixer unitary parameter

def build_qaoa_circuit(n, edges, gamma, beta):
    qc = QuantumCircuit(n, n)
    
    # Step 1: Initial state — equal superposition
    for i in range(n):
        qc.h(i)
    
    # Step 2: Problem unitary (cost operator)
    # For each edge (i,j): apply e^{-i*gamma*(1-Z_i*Z_j)/2}
    for i, j in edges:
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)
    
    # Step 3: Mixer unitary
    # Apply e^{-i*beta*X} to each qubit
    for i in range(n):
        qc.rx(2 * beta, i)
    
    # Measure
    qc.measure(range(n), range(n))
    return qc

qc = build_qaoa_circuit(n_qubits, edges, gamma, beta)
print("--- QAOA Circuit ---")
print(qc.draw(output='text'))
print(f"Circuit depth: {qc.depth()}")
print(f"Gate count: {sum(qc.count_ops().values())}\n")

# ── QAOA PARAMETER OPTIMISATION ───────────────────────────────────────────────
print("--- Optimising QAOA Parameters ---")
sim = AerSimulator()

def qaoa_expectation(params, shots=1024):
    """Evaluate QAOA circuit with given parameters, return average cut value."""
    g, b = params
    bound_qc = qc.assign_parameters({gamma: g, beta: b})
    t_qc = transpile(bound_qc, sim)
    result = sim.run(t_qc, shots=shots).result()
    counts = result.get_counts()
    
    # Calculate expected cut value
    total_cut = 0
    total_shots = sum(counts.values())
    for bitstring, count in counts.items():
        cut = cost_function(bitstring, edges)
        total_cut += cut * count
    return -total_cut / total_shots  # negative because we minimise

# Grid search over parameters
best_params = None
best_val = float('inf')
results_grid = []

print("  Running parameter grid search (gamma x beta)...")
for g in np.linspace(0, np.pi, 10):
    for b in np.linspace(0, np.pi/2, 10):
        val = qaoa_expectation([g, b])
        results_grid.append((g, b, -val))
        if val < best_val:
            best_val = val
            best_params = [g, b]

print(f"  Best gamma: {best_params[0]:.4f} rad")
print(f"  Best beta:  {best_params[1]:.4f} rad")
print(f"  Best expected cut: {-best_val:.3f}")
print()

# ── FINAL RUN WITH BEST PARAMETERS ───────────────────────────────────────────
print("--- Final QAOA Run (Simulator, 1024 shots) ---")
g_opt, b_opt = best_params
bound_qc = qc.assign_parameters({gamma: g_opt, beta: b_opt})
t_qc = transpile(bound_qc, sim)
result = sim.run(t_qc, shots=1024).result()
counts = result.get_counts()

# Sort by count
sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
print("  Top 5 measurement outcomes:")
print(f"  {'Bitstring':<12} {'Count':<8} {'Cut Value':<12} {'% of shots'}")
print("  " + "-"*46)
for bitstring, count in sorted_counts[:5]:
    cut = cost_function(bitstring, edges)
    pct = count/1024*100
    print(f"  {bitstring:<12} {count:<8} {cut:<12} {pct:.1f}%")

# Check if QAOA found optimal
top_bitstring = sorted_counts[0][0]
top_cut = cost_function(top_bitstring, edges)
print(f"\n  QAOA best result: partition={top_bitstring}, cuts={top_cut}")
print(f"  Classical optimal: partition={best_partition}, cuts={best_cut}")
print(f"  QAOA approximation ratio: {top_cut/best_cut:.3f}")

# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    "timestamp": ts,
    "problem": "Max-Cut",
    "graph": {"nodes": n_qubits, "edges": edges},
    "classical_optimal": {"partition": best_partition, "cuts": best_cut, "time_ms": round(t_classical*1000,3)},
    "qaoa": {
        "layers_p": 1,
        "best_gamma": round(g_opt, 4),
        "best_beta": round(b_opt, 4),
        "expected_cut": round(-best_val, 4),
        "top_result": top_bitstring,
        "top_cut": top_cut,
        "approximation_ratio": round(top_cut/best_cut, 4),
        "measurement_counts": dict(sorted_counts[:10])
    }
}
with open(f"results/qaoa_simulator_{ts}.json","w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: results/qaoa_simulator_{ts}.json")
print("\nDONE — Simulator complete. Run qaoa_real_hardware.py for IBM results.")
