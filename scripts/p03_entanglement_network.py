import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from arq_utils import *
import numpy as np
from itertools import combinations
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

print_header(3, "ENTANGLEMENT NETWORK — AI META-ENTANGLEMENT MANAGER")
sim = AerSimulator(); N = 5; SHOTS = 1024

def bell_fidelity(q1, q2, n, noise_model=None):
    qc = QuantumCircuit(n, 2)
    qc.h(q1); qc.cx(q1, q2)
    qc.measure(q1, 0); qc.measure(q2, 1)
    backend = AerSimulator(noise_model=noise_model) if noise_model else sim
    counts = backend.run(transpile(qc, backend), shots=SHOTS).result().get_counts()
    good = counts.get('00',0) + counts.get('11',0)
    return round(good/SHOTS, 4)

class EntanglementManager:
    def __init__(self, n):
        self.n = n
        self.fid = np.zeros((n,n))

    def characterise(self, noise_model=None):
        pairs = list(combinations(range(self.n), 2))
        for q1, q2 in pairs:
            f = bell_fidelity(q1, q2, self.n, noise_model)
            self.fid[q1][q2] = f; self.fid[q2][q1] = f
        return self.fid

    def select_topology(self, min_fid=0.88):
        pairs = [(self.fid[i][j], i, j) for i in range(self.n)
                 for j in range(i+1, self.n)]
        pairs.sort(reverse=True)
        selected = []; used = set()
        for f, q1, q2 in pairs:
            if f >= min_fid and q1 not in used and q2 not in used:
                selected.append((q1, q2, f))
                used.update([q1, q2])
        return selected

    def print_matrix(self):
        print(f"\n  Fidelity Matrix:")
        print("  " + "  ".join([f"  Q{j}" for j in range(self.n)]))
        for i in range(self.n):
            row = f"  Q{i}"
            for j in range(self.n):
                row += "  ----" if i==j else f"  {self.fid[i][j]:.3f}"
            print(row)

mgr = EntanglementManager(N)

print("\n--- Phase 1: Ideal Characterisation ---")
mgr.characterise()
mgr.print_matrix()

print("\n--- Phase 2: Heterogeneous Noise ---")
np.random.seed(42); temp = get_temp(); base = compute_thermal_noise_rate(temp)
nm = NoiseModel()
for q in range(N):
    nm.add_quantum_error(depolarizing_error(base*np.random.uniform(0.8,1.5),1),['h'],[q])
for q1 in range(N):
    for q2 in range(q1+1,N):
        nm.add_quantum_error(depolarizing_error(base*np.random.uniform(5,12),2),['cx'],[q1,q2])

mgr2 = EntanglementManager(N)
mgr2.characterise(nm)
mgr2.print_matrix()

print("\n--- Phase 3: AI Topology Selection ---")
topo = mgr2.select_topology(min_fid=0.85)
print(f"  Selected {len(topo)} pairs:")
for q1,q2,f in topo:
    print(f"    Q{q1}-Q{q2}: {f:.4f}")

print("\n--- Phase 4: Dynamic Adaptation ---")
print(f"{'Cycle':>6} {'Temp':>6} {'Best Pair':>12} {'Fidelity':>10} {'Pairs Used'}")
print("-"*55)
for cycle in range(5):
    t = 40+cycle*8; rate = compute_thermal_noise_rate(t)
    nm_c = NoiseModel()
    nm_c.add_all_qubit_quantum_error(depolarizing_error(rate,1),['h'])
    nm_c.add_all_qubit_quantum_error(depolarizing_error(rate*8,2),['cx'])
    mc = EntanglementManager(N); mc.characterise(nm_c)
    topo_c = mc.select_topology(min_fid=0.83)
    if topo_c:
        bp = f"Q{topo_c[0][0]}-Q{topo_c[0][1]}"; bf = topo_c[0][2]
        tp = ", ".join(f"Q{a}-Q{b}" for a,b,_ in topo_c)
    else:
        bp = "NONE"; bf = 0; tp = "ALL ISOLATED"
    print(f"{cycle+1:>6} {t:>5}°C {bp:>12} {bf:>10.4f} {tp}")

print("\n--- Phase 5: Real IBM Hardware ---")
try:
    svc = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = svc.least_busy(operational=True, simulator=False)
    print(f"  Backend: {backend.name}")
    hw = {}; sampler = SamplerV2(backend)
    for q1,q2 in [(0,1),(1,2),(2,3),(0,3),(1,3)]:
        n_hw = max(q2+1,4)
        qc = QuantumCircuit(n_hw, 2)
        qc.h(q1); qc.cx(q1,q2); qc.measure(q1,0); qc.measure(q2,1)
        job = sampler.run([transpile(qc, backend, optimization_level=3)], shots=1024)
        print(f"  Pair ({q1},{q2}) job: {job.job_id()} — waiting...")
        c = dict(job.result()[0].data.c.get_counts())
        f = (c.get('00',0)+c.get('11',0))/1024
        hw[f"({q1},{q2})"] = {"fidelity": round(f,4), "job_id": job.job_id()}
        print(f"    Fidelity: {f:.4f}")
    best = max(hw.items(), key=lambda x: x[1]["fidelity"])
    print(f"  AI selects: pair {best[0]} fidelity {best[1]['fidelity']:.4f}")
except Exception as e:
    print(f"  IBM error: {e}"); hw = {"error": str(e)}

fname = save_result(3, "Entanglement Network", {"topology": topo, "hw": hw})
print(f"\nSaved: {fname}")
print("PRINCIPLE 3 DONE.")
