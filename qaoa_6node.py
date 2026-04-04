"""
QSimEdge QAOA — 6-Node Graph Max-Cut
Larger problem than 4-node — 6 qubits, 8 edges
Classical brute force: 2^6 = 64 combinations
Quantum: QAOA finds near-optimal faster
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

# 6-node graph — more complex problem
edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,4)]
n_qubits = 6
print("="*65)
print("QSimEdge QAOA — 6-Node Graph Max-Cut")
print("="*65)
print(f"Platform: {platform.machine()} | Temp: {get_temp()}C")
print(f"Nodes: {n_qubits} | Edges: {len(edges)}")
print(f"Edges: {edges}\n")

def cost_fn(bitstring, edges):
    return sum(1 for i,j in edges if bitstring[i]!=bitstring[j])

# Brute force classical
t0 = time.perf_counter()
best_cut, best_part = 0, None
for i in range(2**n_qubits):
    bs = format(i, f'0{n_qubits}b')
    c = cost_fn(bs, edges)
    if c > best_cut: best_cut=c; best_part=bs
t_classical = time.perf_counter()-t0
print(f"Classical brute force: {best_cut} cuts, partition={best_part}, time={t_classical*1000:.1f}ms\n")

def build_qaoa6(edges, gammas, betas, p):
    n = 6
    gv = ParameterVector('g', p)
    bv = ParameterVector('b', p)
    qc = QuantumCircuit(n, n)
    for i in range(n): qc.h(i)
    for layer in range(p):
        for i,j in edges:
            qc.cx(i,j); qc.rz(2*gv[layer],j); qc.cx(i,j)
        for i in range(n): qc.rx(2*bv[layer],i)
    qc.measure(range(n),range(n))
    all_p = list(qc.parameters)
    vals = list(gammas) + list(betas)
    return qc.assign_parameters({all_p[i]:vals[i] for i in range(len(all_p))})

sim = AerSimulator()

def expectation6(params, p=1, shots=1024):
    g = params[:p]; b = params[p:]
    qc = build_qaoa6(edges, g, b, p)
    counts = sim.run(transpile(qc,sim),shots=shots).result().get_counts()
    return -sum(cost_fn(bs,edges)*c for bs,c in counts.items())/shots

print("--- QAOA p=1 Optimisation (6 nodes) ---")
np.random.seed(0)
x0 = np.random.uniform(0,np.pi,2)
t0 = time.perf_counter()
res = minimize(expectation6, x0, args=(1,), method='COBYLA',
               options={'maxiter':200,'rhobeg':0.5})
t_p1 = time.perf_counter()-t0
g1,b1 = res.x[0], res.x[1]

qc_p1 = build_qaoa6(edges,[g1],[b1],1)
counts_p1 = sim.run(transpile(qc_p1,sim),shots=4096).result().get_counts()
sorted_p1 = sorted(counts_p1.items(),key=lambda x:-x[1])
top_p1 = sorted_p1[0][0]; top_cut_p1 = cost_fn(top_p1,edges)
opt_pct_p1 = sum(v for k,v in counts_p1.items() if cost_fn(k,edges)==best_cut)/4096*100
ar_p1 = top_cut_p1/best_cut

print(f"  Time: {t_p1:.1f}s | Top: {top_p1} ({top_cut_p1} cuts)")
print(f"  Approximation ratio: {ar_p1:.4f}")
print(f"  Optimal in {opt_pct_p1:.1f}% of shots\n")

print("--- QAOA p=2 Optimisation (6 nodes) ---")
x0_2 = np.random.uniform(0,np.pi,4)
t0 = time.perf_counter()
res2 = minimize(expectation6, x0_2, args=(2,), method='COBYLA',
                options={'maxiter':300,'rhobeg':0.5})
t_p2 = time.perf_counter()-t0
g2,b2 = res2.x[:2], res2.x[2:]

qc_p2 = build_qaoa6(edges,g2,b2,2)
counts_p2 = sim.run(transpile(qc_p2,sim),shots=4096).result().get_counts()
sorted_p2 = sorted(counts_p2.items(),key=lambda x:-x[1])
top_p2 = sorted_p2[0][0]; top_cut_p2 = cost_fn(top_p2,edges)
opt_pct_p2 = sum(v for k,v in counts_p2.items() if cost_fn(k,edges)==best_cut)/4096*100
ar_p2 = top_cut_p2/best_cut

print(f"  Time: {t_p2:.1f}s | Top: {top_p2} ({top_cut_p2} cuts)")
print(f"  Approximation ratio: {ar_p2:.4f}")
print(f"  Optimal in {opt_pct_p2:.1f}% of shots\n")

print(f"{'='*65}")
print("6-NODE QAOA RESULTS SUMMARY")
print(f"{'='*65}")
print(f"{'Metric':<30} {'Classical':>12} {'QAOA p=1':>12} {'QAOA p=2':>12}")
print("-"*68)
print(f"{'Best cut found':<30} {best_cut:>12} {top_cut_p1:>12} {top_cut_p2:>12}")
print(f"{'Approximation ratio':<30} {'1.0000':>12} {ar_p1:>12.4f} {ar_p2:>12.4f}")
print(f"{'Optimal solution %':<30} {'100%':>12} {opt_pct_p1:>11.1f}% {opt_pct_p2:>11.1f}%")
print(f"{'Search time':<30} {t_classical*1000:>10.1f}ms {t_p1:>10.1f}s {t_p2:>10.1f}s")

print("\n--- Submitting p=2 (6-node) to Real IBM Hardware ---")
try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(operational=True, simulator=False)
    print(f"  Backend: {backend.name}")
    qc_real = build_qaoa6(edges,g2,b2,2)
    qc_t = transpile(qc_real, backend, optimization_level=3)
    print(f"  Transpiled depth: {qc_t.depth()} | Gates: {sum(qc_t.count_ops().values())}")
    job = SamplerV2(backend).run([qc_t], shots=1024)
    print(f"  Job ID: {job.job_id()} — waiting...")
    real_c = dict(job.result()[0].data.c.get_counts())
    real_s = sorted(real_c.items(),key=lambda x:-x[1])
    real_top = real_s[0][0]; real_cut = cost_fn(real_top,edges)
    real_opt_pct = sum(v for k,v in real_c.items() if cost_fn(k,edges)==best_cut)/1024*100
    print(f"\n  Real HW top: {real_top} ({real_cut} cuts)")
    print(f"  Approximation ratio: {real_cut/best_cut:.4f}")
    print(f"  Optimal in {real_opt_pct:.1f}% of shots")
    print(f"  Job ID: {job.job_id()}")
except Exception as e:
    print(f"  Real hardware: {e}")

ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname=f"results/qaoa_6node_{ts}.json"
with open(fname,"w") as f:
    json.dump({"timestamp":ts,"platform":platform.machine(),
               "graph":{"nodes":n_qubits,"edges":edges},
               "classical":{"cuts":best_cut,"partition":best_part,"time_ms":round(t_classical*1000,2)},
               "qaoa_p1":{"approx_ratio":round(ar_p1,4),"optimal_pct":round(opt_pct_p1,2)},
               "qaoa_p2":{"approx_ratio":round(ar_p2,4),"optimal_pct":round(opt_pct_p2,2)}},f,indent=2)
print(f"\nSaved: {fname}")
print("DONE.")
