"""
QSimEdge VQE — H2 Molecule Ground State
Complete fixed version — all imports correct
Simulator optimisation + real hardware evaluation
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize
import json, datetime, time, platform

def get_temp():
    try: return float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000
    except: return 0.0

print("="*65)
print("QSimEdge VQE — H2 Molecule Ground State Energy")
print("="*65)
print(f"Platform: {platform.machine()} | Temp: {get_temp()}C\n")

# H2 Hamiltonian STO-3G Jordan-Wigner
H_TERMS = [
    ("II", -1.0523732),
    ("IZ",  0.3979374),
    ("ZI", -0.3979374),
    ("ZZ", -0.0112801),
    ("XX",  0.1809312),
]
EXACT_ENERGY = -1.8572750

print("H2 Hamiltonian terms:")
for term,coeff in H_TERMS:
    print(f"  {term}: {coeff:+.7f}")
print(f"\nExact FCI energy: {EXACT_ENERGY} Hartree\n")

def build_ansatz(params):
    """Hardware-efficient 2-qubit ansatz."""
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 0)
    qc.ry(params[3], 1)
    return qc

def measure_pauli(qc_in, pauli_str, sim, shots=4096):
    """Measure expectation value of a Pauli string."""
    qc = qc_in.copy()
    n = len(pauli_str)
    # Basis rotation
    for i, p in enumerate(reversed(pauli_str)):
        if p == 'X': qc.h(i)
        elif p == 'Y': qc.sdg(i); qc.h(i)
    qc.measure_all()
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
    exp_val = 0.0
    total = sum(counts.values())
    for bitstring, count in counts.items():
        clean = bitstring.replace(' ','')
        parity = 1
        for i, p in enumerate(reversed(pauli_str)):
            if p != 'I' and i < len(clean) and clean[-(i+1)] == '1':
                parity *= -1
        exp_val += parity * count / total
    return exp_val

def energy_from_params(params, sim, shots=4096):
    qc = build_ansatz(params)
    energy = 0.0
    for pauli_str, coeff in H_TERMS:
        if all(p == 'I' for p in pauli_str):
            energy += coeff
        else:
            exp_val = measure_pauli(qc, pauli_str, sim, shots)
            energy += coeff * exp_val
    return energy

# ── SIMULATOR OPTIMISATION ────────────────────────────────────────────────
print("--- VQE Simulator Optimisation ---")
sim = AerSimulator()
call_count = [0]

def cost(params):
    call_count[0] += 1
    if call_count[0] % 10 == 0:
        print(f"    Iteration {call_count[0]}...", flush=True)
    return energy_from_params(params, sim, shots=2048)

x0 = np.array([0.5, 1.0, -0.5, 0.3])
t0 = time.perf_counter()
result = minimize(cost, x0, method='COBYLA',
                  options={'maxiter': 80, 'rhobeg': 0.3})
t_vqe = time.perf_counter() - t0

vqe_energy = result.fun
opt_params = result.x
error_ha = abs(vqe_energy - EXACT_ENERGY)
error_ev = error_ha * 27.211396
chem_acc = error_ha < 0.0016

print(f"\n  VQE energy:      {vqe_energy:.6f} Hartree")
print(f"  Exact energy:    {EXACT_ENERGY:.6f} Hartree")
print(f"  Error:           {error_ha:.6f} Hartree ({error_ev:.4f} eV)")
print(f"  Chemical acc:    {'YES (<1 kcal/mol)' if chem_acc else 'NO (needs more layers)'}")
print(f"  Time:            {t_vqe:.1f}s ({result.nfev} evaluations)")
print(f"  Opt parameters:  {[round(x,4) for x in opt_params]}\n")

results = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "platform": platform.machine(),
    "hamiltonian": H_TERMS,
    "exact_energy_hartree": EXACT_ENERGY,
    "simulator": {
        "vqe_energy": round(vqe_energy,6),
        "error_hartree": round(error_ha,6),
        "error_ev": round(error_ev,4),
        "chemical_accuracy": chem_acc,
        "time_s": round(t_vqe,2),
        "n_evaluations": result.nfev,
        "opt_params": [round(x,4) for x in opt_params]
    }
}

# ── REAL HARDWARE EVALUATION ─────────────────────────────────────────────
print("--- Real IBM Hardware Evaluation ---")
try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(operational=True, simulator=False)
    print(f"  Backend: {backend.name} ({backend.num_qubits}Q)")

    sampler = SamplerV2(backend)
    real_energies = {}

    for pauli_str, coeff in H_TERMS:
        if all(p=='I' for p in pauli_str):
            real_energies[pauli_str] = {"coeff": coeff, "exp_val": 1.0,
                                         "contribution": coeff, "job_id": "identity"}
            print(f"  {pauli_str}: identity term = {coeff:.7f}")
            continue

        qc = build_ansatz(opt_params)
        for i, p in enumerate(reversed(pauli_str)):
            if p == 'X': qc.h(i)
            elif p == 'Y': qc.sdg(i); qc.h(i)
        qc.measure_all()
        qc_t = transpile(qc, backend, optimization_level=1)

        job = sampler.run([qc_t], shots=1024)
        print(f"  {pauli_str}: Job {job.job_id()} — waiting...", flush=True)
        res = job.result()
        counts = dict(res[0].data.meas.get_counts())

        exp_val = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            clean = bitstring.replace(' ','')
            parity = 1
            for i, p in enumerate(reversed(pauli_str)):
                if p != 'I' and i < len(clean) and clean[-(i+1)] == '1':
                    parity *= -1
            exp_val += parity * count / total

        contribution = coeff * exp_val
        real_energies[pauli_str] = {
            "coeff": coeff, "exp_val": round(exp_val,4),
            "contribution": round(contribution,6), "job_id": job.job_id()
        }
        print(f"  {pauli_str}: exp_val={exp_val:.4f} contribution={contribution:.6f}")

    energy_real = sum(v["contribution"] for v in real_energies.values())
    error_real = abs(energy_real - EXACT_ENERGY)

    print(f"\n  Real hardware energy: {energy_real:.6f} Hartree")
    print(f"  Exact energy:         {EXACT_ENERGY:.6f} Hartree")
    print(f"  Error (real HW):      {error_real:.6f} Hartree ({error_real*27.211:.4f} eV)")

    results["real_hardware"] = {
        "backend": backend.name,
        "energy_hartree": round(energy_real,6),
        "error_hartree": round(error_real,6),
        "pauli_terms": real_energies,
        "job_ids": {k: v["job_id"] for k,v in real_energies.items()}
    }

    print(f"\n{'='*65}")
    print("VQE FINAL RESULTS")
    print(f"{'='*65}")
    print(f"{'Method':<28} {'Energy (Ha)':>14} {'Error (Ha)':>12} {'Error (eV)':>12}")
    print("-"*68)
    print(f"{'Exact FCI':<28} {EXACT_ENERGY:>14.6f} {'0.000000':>12} {'0.0000':>12}")
    print(f"{'VQE Simulator':<28} {vqe_energy:>14.6f} {error_ha:>12.6f} {error_ev:>12.4f}")
    print(f"{'VQE Real IBM Hardware':<28} {energy_real:>14.6f} {error_real:>12.6f} {error_real*27.211:>12.4f}")

    print(f"\nPauli term contributions (real hardware):")
    print(f"{'Term':<8} {'Coeff':>12} {'Exp Val':>10} {'Contribution':>14} {'Job ID'}")
    print("-"*70)
    for term, data in real_energies.items():
        print(f"{term:<8} {data['coeff']:>12.7f} {data['exp_val']:>10.4f} {data['contribution']:>14.6f}  {data['job_id']}")

except Exception as e:
    print(f"  Real hardware error: {e}")
    import traceback; traceback.print_exc()

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"results/vqe_h2_complete_{ts}.json"
with open(fname,"w") as f: json.dump(results,f,indent=2)
print(f"\nSaved: {fname}")
print("DONE.")
