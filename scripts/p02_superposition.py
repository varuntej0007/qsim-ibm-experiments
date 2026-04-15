import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from arq_utils import *
import numpy as np
from scipy import stats
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

print_header(2, "QUBIT SUPERPOSITION — UNIFORMITY VALIDATION")
sim = AerSimulator(); SHOTS = 4096

def analyse(counts, n, shots):
    n_states = 2**n
    exp = shots / n_states
    obs = np.array([counts.get(format(i,f'0{n}b'),0) for i in range(n_states)])
    chi2, pval = stats.chisquare(obs)
    probs = obs/shots; probs = np.where(probs>0, probs, 1e-10)
    entropy = -np.sum(probs*np.log2(probs))
    return {"chi2": round(chi2,2), "pval": round(float(pval),4),
            "uniform": bool(pval>0.05),
            "entropy_pct": round(entropy/np.log2(n_states)*100,2),
            "deviation_pct": round((obs.max()-obs.min())/(shots/n_states)*100,2)}

print("\n--- Ideal Simulator: 1 to 6 qubits ---")
print(f"{'Qubits':>7} {'States':>8} {'chi2':>8} {'p-value':>10} {'Uniform':>10} {'Entropy%':>10}")
print("-"*58)
ideal = []
for n in range(1,7):
    qc = QuantumCircuit(n,n); qc.h(range(n)); qc.measure_all()
    counts = sim.run(transpile(qc,sim), shots=SHOTS).result().get_counts()
    sd = analyse(counts, n, SHOTS)
    ideal.append({"n": n, **sd})
    print(f"{n:>7} {2**n:>8} {sd['chi2']:>8.2f} {sd['pval']:>10.4f} "
          f"{'YES':>10} {sd['entropy_pct']:>9.2f}%")

print("\n--- Thermal Noise Sweep (4-qubit) ---")
print(f"{'Temp':>6} {'Noise':>10} {'Entropy%':>10} {'Uniform':>10} {'AI Correction':>15}")
print("-"*58)
thermal = []
for temp in [25,35,45,55,65,75,85]:
    rate = compute_thermal_noise_rate(temp)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(rate,1),['h'])
    sn = AerSimulator(noise_model=nm)
    qc = QuantumCircuit(4,4); qc.h(range(4)); qc.measure_all()
    counts = sn.run(transpile(qc,sn), shots=SHOTS).result().get_counts()
    sd = analyse(counts, 4, SHOTS)
    ai_corr = -(temp-40)*0.008
    thermal.append({"temp": temp, "noise": rate, **sd, "ai_correction": round(ai_corr,4)})
    print(f"{temp:>5}°C {rate:>10.6f} {sd['entropy_pct']:>9.2f}% "
          f"{'YES' if sd['uniform'] else 'NO':>10} {ai_corr:>+14.4f} rad")

print("\n--- Real IBM Hardware ---")
try:
    svc = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = svc.least_busy(operational=True, simulator=False)
    print(f"  Backend: {backend.name}")
    hw = {}
    sampler = SamplerV2(backend)
    for n in [2,3,4,5]:
        qc = QuantumCircuit(n,n); qc.h(range(n)); qc.measure_all()
        job = sampler.run([transpile(qc, backend, optimization_level=3)], shots=1024)
        print(f"  {n}Q job: {job.job_id()} — waiting...")
        c = dict(job.result()[0].data.c.get_counts())
        sd = analyse(c, n, 1024)
        hw[f"{n}q"] = {"job_id": job.job_id(), **sd}
        print(f"    {n}Q: uniform={'YES' if sd['uniform'] else 'NO'}, "
              f"entropy={sd['entropy_pct']:.2f}%, chi2={sd['chi2']:.2f}")
except Exception as e:
    print(f"  IBM error: {e}"); hw = {"error": str(e)}

fname = save_result(2, "Superposition", {"ideal": ideal, "thermal": thermal, "hw": hw})
print(f"\nSaved: {fname}")
print("PRINCIPLE 2 DONE.")
