"""
QSimEdge Zero-Noise Extrapolation (ZNE)
Error mitigation technique — runs circuit at 1x, 2x, 3x noise
then extrapolates back to zero noise to get better answer
This is what IBM's own error mitigation uses
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import json, datetime, time, platform

def get_temp():
    try: return float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000
    except: return 0.0

print("="*65)
print("QSimEdge Zero-Noise Extrapolation (ZNE)")
print("Error mitigation on Bell State + QAOA")
print("="*65)
print(f"Platform: {platform.machine()} | Temp: {get_temp()}C\n")

results = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "platform": platform.machine(),
    "method": "Zero Noise Extrapolation (ZNE)",
    "experiments": []
}

def build_noise_model(error_rate):
    """Build depolarizing noise model at given error rate."""
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ['h','rx','ry','rz'])
    noise.add_all_qubit_quantum_error(depolarizing_error(error_rate*2, 2), ['cx','cz'])
    return noise

def measure_bell_fidelity(sim, shots=4096):
    """Measure Bell State |00> + |11> fidelity — ideal = 1.0, noise reduces it."""
    qc = QuantumCircuit(2,2)
    qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])
    counts = sim.run(transpile(qc,sim), shots=shots).result().get_counts()
    ideal = (counts.get('00',0) + counts.get('11',0)) / shots
    return ideal

# ── ZNE ON SIMULATOR WITH ARTIFICIAL NOISE ─────────────────────────────
print("--- ZNE on Bell State (Simulated Noise Model) ---")
base_error = 0.01  # 1% base error rate
noise_factors = [1, 2, 3]
fidelities = []

for factor in noise_factors:
    error_rate = base_error * factor
    noise_model = build_noise_model(error_rate)
    sim_noisy = AerSimulator(noise_model=noise_model)
    fid = measure_bell_fidelity(sim_noisy)
    fidelities.append(fid)
    print(f"  Noise factor {factor}x (error={error_rate:.3f}): Fidelity = {fid:.4f}")

# Linear extrapolation to zero noise
# Fit line through points (noise_factor, fidelity) and extrapolate to 0
coeffs = np.polyfit(noise_factors, fidelities, 1)
zne_fidelity_linear = coeffs[1]  # y-intercept = value at noise=0

# Richardson extrapolation (more accurate)
# For 3 points with factors 1,2,3: Richardson gives weighted combination
rich = (3*fidelities[0] - 3*fidelities[1] + fidelities[2]) / 1
# Simpler: use 2-point Richardson
rich_2pt = 2*fidelities[0] - fidelities[1]

ideal_sim = measure_bell_fidelity(AerSimulator())

print(f"\n  Ideal (no noise):         {ideal_sim:.4f}")
print(f"  Raw (1x noise):           {fidelities[0]:.4f}")
print(f"  ZNE linear extrapolation: {zne_fidelity_linear:.4f}")
print(f"  ZNE Richardson (2pt):     {rich_2pt:.4f}")
print(f"\n  Error without ZNE: {abs(ideal_sim-fidelities[0]):.4f}")
print(f"  Error with ZNE:    {abs(ideal_sim-zne_fidelity_linear):.4f}")
improvement = abs(ideal_sim-fidelities[0]) / max(abs(ideal_sim-zne_fidelity_linear),0.0001)
print(f"  ZNE improvement:   {improvement:.1f}x closer to ideal")

results["experiments"].append({
    "circuit": "Bell State",
    "method": "ZNE Linear Extrapolation",
    "noise_factors": noise_factors,
    "raw_fidelities": [round(f,4) for f in fidelities],
    "ideal_fidelity": round(ideal_sim,4),
    "zne_linear": round(zne_fidelity_linear,4),
    "zne_richardson_2pt": round(rich_2pt,4),
    "error_without_zne": round(abs(ideal_sim-fidelities[0]),4),
    "error_with_zne": round(abs(ideal_sim-zne_fidelity_linear),4),
    "improvement_factor": round(improvement,2)
})

# ── ZNE ON QAOA COST FUNCTION ────────────────────────────────────────────
print("\n--- ZNE on QAOA Expected Cut Value ---")
edges = [(0,1),(1,2),(2,3),(3,0),(0,2)]
n = 4
gamma_opt, beta_opt = 2.0944, 0.8727

def build_qaoa_p1(edges, gamma, beta):
    from qiskit.circuit import Parameter
    g = Parameter('g'); b = Parameter('b')
    qc = QuantumCircuit(n, n)
    for i in range(n): qc.h(i)
    for i,j in edges:
        qc.cx(i,j); qc.rz(2*g,j); qc.cx(i,j)
    for i in range(n): qc.rx(2*b,i)
    qc.measure(range(n),range(n))
    return qc.assign_parameters({g: gamma, b: beta})

def cost_fn(bitstring):
    return sum(1 for i,j in edges if bitstring[i]!=bitstring[j])

def measure_qaoa_cut(sim, gamma, beta, shots=2048):
    qc = build_qaoa_p1(edges, gamma, beta)
    counts = sim.run(transpile(qc,sim), shots=shots).result().get_counts()
    return sum(cost_fn(bs)*c for bs,c in counts.items()) / shots

noise_factors_qaoa = [1, 2, 3]
qaoa_cuts = []

for factor in noise_factors_qaoa:
    nm = build_noise_model(0.005 * factor)
    sim_n = AerSimulator(noise_model=nm)
    cut = measure_qaoa_cut(sim_n, gamma_opt, beta_opt)
    qaoa_cuts.append(cut)
    print(f"  Noise factor {factor}x: Expected cut = {cut:.4f}")

ideal_cut = measure_qaoa_cut(AerSimulator(), gamma_opt, beta_opt)
coeffs_q = np.polyfit(noise_factors_qaoa, qaoa_cuts, 1)
zne_cut = coeffs_q[1]
rich_cut = 2*qaoa_cuts[0] - qaoa_cuts[1]

print(f"\n  Ideal cut (no noise):     {ideal_cut:.4f}")
print(f"  Raw cut (1x noise):       {qaoa_cuts[0]:.4f}")
print(f"  ZNE linear:               {zne_cut:.4f}")
print(f"  ZNE Richardson:           {rich_cut:.4f}")
print(f"  Classical optimal cut:    4.0000")
print(f"\n  Error without ZNE: {abs(ideal_cut-qaoa_cuts[0]):.4f}")
print(f"  Error with ZNE:    {abs(ideal_cut-zne_cut):.4f}")

results["experiments"].append({
    "circuit": "QAOA Max-Cut p=1",
    "noise_factors": noise_factors_qaoa,
    "raw_cuts": [round(c,4) for c in qaoa_cuts],
    "ideal_cut": round(ideal_cut,4),
    "zne_linear": round(zne_cut,4),
    "zne_richardson": round(rich_cut,4),
    "classical_optimal": 4.0
})

# ── ZNE ON REAL IBM HARDWARE ─────────────────────────────────────────────
print("\n--- ZNE on Real IBM Hardware ---")
print("Running Bell State at 1x noise (native gates)")
print("Then 3x noise (folding gates 3 times)")
try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(operational=True, simulator=False)
    print(f"  Backend: {backend.name}")
    sampler = SamplerV2(backend)

    # Native circuit
    qc_bell = QuantumCircuit(2,2)
    qc_bell.h(0); qc_bell.cx(0,1); qc_bell.measure([0,1],[0,1])
    qc_1x = transpile(qc_bell, backend, optimization_level=1)

    # 3x noise: fold gates (repeat each gate 3 times: G G† G = G)
    def fold_gates(qc, factor=3):
        """Simple gate folding for noise scaling — fold each gate n times."""
        from qiskit import QuantumCircuit
        qc_folded = QuantumCircuit(qc.num_qubits, qc.num_clbits)
        for instruction in qc.data:
            op = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
            if inst.name == 'measure':
                qc_folded.measure(qargs, cargs)
                continue
            # Add gate, then its inverse, then gate again (factor-1)/2 times extra
            qc_folded.append(op, qargs, cargs)
            extra = (factor - 1) // 2
            for _ in range(extra):
                qc_folded.append(inst, qargs, cargs)
                qc_folded.append(inst, qargs, cargs)
        return qc_folded

    qc_3x = QuantumCircuit(2, 2)
    qc_3x.h(0)
    qc_3x.cx(0, 1)
    qc_3x.cx(0, 1)
    qc_3x.cx(0, 1)
    qc_3x.measure([0,1],[0,1])
    qc_3x = transpile(qc_3x, backend, optimization_level=1)

    real_jobs = {}
    for label, qc_hw in [("1x", qc_1x), ("3x", qc_3x)]:
        job = sampler.run([qc_hw], shots=1024)
        print(f"  {label} noise job: {job.job_id()} — waiting...", flush=True)
        res = job.result()
        counts = dict(res[0].data.c.get_counts())
        fid = (counts.get('00',0) + counts.get('11',0)) / 1024
        real_jobs[label] = {"job_id": job.job_id(), "fidelity": round(fid,4), "counts": counts}
        print(f"  {label} fidelity: {fid:.4f}")

    zne_real = 2*real_jobs["1x"]["fidelity"] - real_jobs["3x"]["fidelity"]
    print(f"\n  1x fidelity:  {real_jobs['1x']['fidelity']:.4f}")
    print(f"  3x fidelity:  {real_jobs['3x']['fidelity']:.4f}")
    print(f"  ZNE estimate: {zne_real:.4f} (extrapolated to zero noise)")
    print(f"  Ideal target: 1.0000")
    print(f"  ZNE job IDs: {real_jobs['1x']['job_id']} (1x), {real_jobs['3x']['job_id']} (3x)")

    results["real_hardware_zne"] = {
        "backend": backend.name,
        "bell_1x": real_jobs["1x"],
        "bell_3x": real_jobs["3x"],
        "zne_extrapolated": round(zne_real,4),
        "ideal_target": 1.0
    }

except Exception as e:
    print(f"  Real hardware error: {e}")
    import traceback; traceback.print_exc()

print(f"\n{'='*65}")
print("ZNE SUMMARY TABLE")
print(f"{'='*65}")
print(f"{'Circuit':<20} {'Raw (1x)':<12} {'ZNE Linear':<14} {'ZNE Rich':<12} {'Ideal'}")
print("-"*68)
for exp in results["experiments"]:
    circ = exp["circuit"]
    if "fidelity" in str(exp):
        raw = exp.get("raw_fidelities",[0])[0]
        zne = exp.get("zne_linear",0)
        rich = exp.get("zne_richardson_2pt",0)
        ideal = exp.get("ideal_fidelity",1.0)
    else:
        raw = exp.get("raw_cuts",[0])[0]
        zne = exp.get("zne_linear",0)
        rich = exp.get("zne_richardson",0)
        ideal = exp.get("ideal_cut",0)
    print(f"{circ:<20} {raw:<12.4f} {zne:<14.4f} {rich:<12.4f} {ideal:.4f}")

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"results/zne_{ts}.json"
with open(fname,"w") as f: json.dump(results,f,indent=2)
print(f"\nSaved: {fname}")
print("DONE.")
