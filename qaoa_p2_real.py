"""
QSimEdge QAOA p=2 — Max-Cut
Two-layer QAOA gives better approximation than p=1
Comparing p=1 vs p=2 on simulator AND real IBM hardware
This is publishable beyond basics
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from scipy.optimize import minimize
import json, datetime, time, platform

def get_temp():
    try: return float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000
    except: return 0.0

edges = [(0,1),(1,2),(2,3),(3,0),(0,2)]
n_qubits = 4
classical_opt = 4

def cost_function(bitstring, edges):
    return sum(1 for i,j in edges if bitstring[i] != bitstring[j])

def build_qaoa(n, edges, gammas, betas, p):
    """Build QAOA circuit with p layers."""
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
    for layer in range(p):
        for i,j in edges:
            qc.cx(i,j)
            qc.rz(2*gammas[layer], j)
            qc.cx(i,j)
        for i in range(n):
            qc.rx(2*betas[layer], i)
    qc.measure(range(n), range(n))
    return qc

def expectation(params, p, shots=1024):
    gammas = params[:p]
    betas  = params[p:]
    gv = ParameterVector('g', p)
    bv = ParameterVector('b', p)
    qc = build_qaoa(n_qubits, edges, gv, bv, p)
    param_dict = {}
    for i in range(p):
        param_dict[gv[i]] = gammas[i]
        param_dict[bv[i]] = betas[i]
    bound = qc.assign_parameters(param_dict)
    sim = AerSimulator()
    t_qc = transpile(bound, sim)
    counts = sim.run(t_qc, shots=shots).result().get_counts()
    total_cut = sum(cost_function(bs, edges)*c for bs,c in counts.items())
    return -total_cut / shots

print("="*65)
print("QSimEdge QAOA p=1 vs p=2 — Max-Cut on 4-node graph")
print("="*65)
print(f"Platform: {platform.machine()} | Temp: {get_temp()}C")
print(f"Graph edges: {edges}")
print(f"Classical optimal: {classical_opt} cuts\n")

results = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "platform": platform.machine(), "edges": edges,
           "classical_optimal": classical_opt, "layers": {}}

for p in [1, 2]:
    print(f"--- QAOA p={p} Optimisation ---")
    np.random.seed(42)
    x0 = np.random.uniform(0, np.pi, 2*p)
    t0 = time.perf_counter()
    res = minimize(expectation, x0, args=(p,), method='COBYLA',
                   options={'maxiter': 200, 'rhobeg': 0.5})
    t_opt = time.perf_counter() - t0
    best_params = res.x
    best_exp = -res.fun

    # Final evaluation with best params
    gv = ParameterVector('g', p)
    bv = ParameterVector('b', p)
    qc = build_qaoa(n_qubits, edges, gv, bv, p)
    param_dict = {}
    for i in range(p):
        param_dict[gv[i]] = best_params[i]
        param_dict[gv[i]+p if False else list(gv)[i]] = best_params[i]
    # simpler: assign by position
    all_params = list(qc.parameters)
    param_dict2 = {all_params[i]: best_params[i] for i in range(len(all_params))}
    bound = qc.assign_parameters(param_dict2)

    sim = AerSimulator()
    counts = sim.run(transpile(bound, sim), shots=4096).result().get_counts()
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    top_bs = sorted_counts[0][0]
    top_cut = cost_function(top_bs, edges)
    approx_ratio = top_cut / classical_opt
    opt_pct = sum(v for k,v in counts.items() if cost_function(k,edges)==classical_opt)/4096*100

    print(f"  Optimisation time: {t_opt:.1f}s ({res.nfev} evaluations)")
    print(f"  Best expected cut: {best_exp:.4f}")
    print(f"  Top result:        {top_bs} — {top_cut} cuts")
    print(f"  Approximation ratio: {approx_ratio:.4f}")
    print(f"  Optimal found in:  {opt_pct:.1f}% of shots")
    print(f"  Gammas: {[round(x,4) for x in best_params[:p]]}")
    print(f"  Betas:  {[round(x,4) for x in best_params[p:]]}\n")

    results["layers"][f"p{p}"] = {
        "best_expected_cut": round(best_exp,4),
        "top_bitstring": top_bs,
        "top_cut": top_cut,
        "approximation_ratio": round(approx_ratio,4),
        "optimal_pct": round(opt_pct,2),
        "opt_time_s": round(t_opt,2),
        "n_evaluations": res.nfev,
        "gammas": [round(x,4) for x in best_params[:p]],
        "betas": [round(x,4) for x in best_params[p:]],
        "top_counts": dict(sorted_counts[:8])
    }

print("--- Summary: p=1 vs p=2 ---")
print(f"{'Metric':<30} {'p=1':>12} {'p=2':>12}")
print("-"*56)
for metric in ["best_expected_cut","approximation_ratio","optimal_pct","n_evaluations"]:
    v1 = results["layers"]["p1"][metric]
    v2 = results["layers"]["p2"][metric]
    print(f"{metric:<30} {str(v1):>12} {str(v2):>12}")

print("\n--- Submitting p=2 to Real IBM Hardware ---")
try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(operational=True, simulator=False)
    print(f"  Backend: {backend.name} ({backend.num_qubits}Q)")

    gv = ParameterVector('g', 2)
    bv = ParameterVector('b', 2)
    qc_real = build_qaoa(n_qubits, edges, gv, bv, 2)
    best_p2 = results["layers"]["p2"]
    params_p2 = best_p2["gammas"] + best_p2["betas"]
    all_params = list(qc_real.parameters)
    param_dict = {all_params[i]: params_p2[i] for i in range(len(all_params))}
    bound_real = qc_real.assign_parameters(param_dict)
    qc_t = transpile(bound_real, backend, optimization_level=3)
    print(f"  Transpiled depth: {qc_t.depth()} | Gates: {sum(qc_t.count_ops().values())}")

    sampler = SamplerV2(backend)
    job = sampler.run([qc_t], shots=1024)
    print(f"  Job ID: {job.job_id()}")
    print("  Waiting...")

    real_result = job.result()
    real_counts = dict(real_result[0].data.c.get_counts())
    real_sorted = sorted(real_counts.items(), key=lambda x: -x[1])
    real_top = real_sorted[0][0]
    real_top_cut = cost_function(real_top, edges)
    real_opt_pct = sum(v for k,v in real_counts.items() if cost_function(k,edges)==classical_opt)/1024*100

    print(f"\n  Real hardware top result: {real_top} — {real_top_cut} cuts")
    print(f"  Optimal in {real_opt_pct:.1f}% of shots")
    print(f"  Approximation ratio: {real_top_cut/classical_opt:.4f}")

    results["real_hardware"] = {
        "backend": backend.name, "job_id": job.job_id(),
        "top_bitstring": real_top, "top_cut": real_top_cut,
        "approximation_ratio": round(real_top_cut/classical_opt,4),
        "optimal_pct": round(real_opt_pct,2),
        "transpiled_depth": qc_t.depth(),
        "counts": dict(real_sorted[:10])
    }

    print(f"\n{'='*65}")
    print("FINAL COMPARISON — p=1 vs p=2 vs Real Hardware")
    print(f"{'='*65}")
    print(f"{'Metric':<32} {'p=1 Sim':>10} {'p=2 Sim':>10} {'p=2 Real':>10}")
    print("-"*64)
    p1 = results["layers"]["p1"]
    p2 = results["layers"]["p2"]
    rh = results["real_hardware"]
    print(f"{'Approximation ratio':<32} {p1['approximation_ratio']:>10} {p2['approximation_ratio']:>10} {rh['approximation_ratio']:>10}")
    print(f"{'Optimal cut % of shots':<32} {p1['optimal_pct']:>10} {p2['optimal_pct']:>10} {rh['optimal_pct']:>10}")
    print(f"{'Top bitstring':<32} {p1['top_bitstring']:>10} {p2['top_bitstring']:>10} {rh['top_bitstring']:>10}")
    print(f"{'Top cut value':<32} {p1['top_cut']:>10} {p2['top_cut']:>10} {rh['top_cut']:>10}")

except Exception as e:
    print(f"  Real hardware error: {e}")

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"results/qaoa_p2_{ts}.json"
with open(fname,"w") as f: json.dump(results,f,indent=2)
print(f"\nSaved: {fname}")
print("DONE.")
