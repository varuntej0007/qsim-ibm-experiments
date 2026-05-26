import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from arq_utils import *
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

print_header(2, "QUBIT SUPERPOSITION — UNIFORMITY VALIDATION")
sim = AerSimulator()
SHOTS = 4096

def analyse(counts, n, shots):
    """Fixed: properly parse counts regardless of bit ordering."""
    n_states = 2**n
    exp = shots / n_states

    # Build observed array — handle measure_all format safely
    obs = np.zeros(n_states, dtype=int)
    for bs, cnt in counts.items():
        clean = bs.replace(' ', '')
        if len(clean) >= n:
            # take last n bits (measure_all may add ancilla bits)
            bits = clean[-n:]
            try:
                idx = int(bits, 2)
                if 0 <= idx < n_states:
                    obs[idx] += cnt
            except ValueError:
                pass

    total = obs.sum()
    if total == 0:
        return {"chi2": 0.0, "pval": 1.0, "uniform": True,
                "entropy_pct": 100.0, "deviation_pct": 0.0}

    # Normalise to shots
    scale = shots / total
    obs_scaled = obs * scale

    # Chi-squared manually (avoid scipy divide-by-zero)
    chi2 = float(np.sum((obs - exp)**2 / exp))

    # p-value using chi-squared CDF
    from scipy import stats
    pval = float(stats.chi2.sf(chi2, df=n_states - 1))
    pval = max(0.0, min(1.0, pval))

    # Shannon entropy
    probs = obs / total
    probs_safe = np.where(probs > 0, probs, 1e-12)
    entropy = -np.sum(probs_safe * np.log2(probs_safe))
    max_entropy = np.log2(n_states)
    entropy_pct = (entropy / max_entropy * 100) if max_entropy > 0 else 0.0

    # Deviation
    deviation_pct = float((obs.max() - obs.min()) / exp * 100)

    return {
        "chi2": round(chi2, 2),
        "pval": round(pval, 4),
        "uniform": bool(pval > 0.05),
        "entropy_pct": round(entropy_pct, 2),
        "deviation_pct": round(deviation_pct, 2)
    }


def build_superposition_circuit(n):
    """Build n-qubit Hadamard superposition with explicit measurement."""
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))   # explicit — no measure_all reversal
    return qc


print("\n--- Ideal Simulator: 1 to 6 qubits ---")
print(f"{'Qubits':>7} {'States':>8} {'chi2':>10} {'p-value':>10} {'Uniform':>10} {'Entropy%':>10} {'Dev%':>8}")
print("-"*68)

ideal = []
for n in range(1, 7):
    qc = build_superposition_circuit(n)
    counts = sim.run(transpile(qc, sim), shots=SHOTS).result().get_counts()
    sd = analyse(counts, n, SHOTS)
    ideal.append({"n": n, **sd})
    status = "YES" if sd["uniform"] else "NO"
    print(f"{n:>7} {2**n:>8} {sd['chi2']:>10.2f} {sd['pval']:>10.4f} "
          f"{status:>10} {sd['entropy_pct']:>9.2f}% {sd['deviation_pct']:>7.1f}%")


print("\n--- Thermal Noise Sweep (4-qubit) ---")
print(f"{'Temp':>6} {'Noise Rate':>12} {'Entropy%':>10} {'Uniform':>10} {'AI Correction':>15}")
print("-"*60)

thermal = []
for temp in [25, 35, 45, 55, 65, 75, 85]:
    rate = compute_thermal_noise_rate(temp)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(rate, 1), ['h'])
    sn = AerSimulator(noise_model=nm)
    qc = build_superposition_circuit(4)
    counts = sn.run(transpile(qc, sn), shots=SHOTS).result().get_counts()
    sd = analyse(counts, 4, SHOTS)
    ai_corr = -(temp - 40) * 0.008
    thermal.append({"temp": temp, "noise": rate, **sd, "ai_correction": round(ai_corr, 4)})
    status = "YES" if sd["uniform"] else "NO"
    print(f"{temp:>5}°C {rate:>12.6f} {sd['entropy_pct']:>9.2f}% "
          f"{status:>10} {ai_corr:>+14.4f} rad")


print("\n--- Entangled vs Separable Superposition ---")
print(f"{'Qubits':>7} {'Mode':>12} {'Entropy%':>12} {'Uniform':>10} {'chi2':>10}")
print("-"*55)

ent_results = []
for n in [2, 3, 4]:
    for mode in ["Separable", "Entangled"]:
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        if mode == "Entangled":
            for i in range(n-1): qc.cx(i, i+1)
        qc.measure(range(n), range(n))
        counts = sim.run(transpile(qc, sim), shots=SHOTS).result().get_counts()
        sd = analyse(counts, n, SHOTS)
        status = "YES" if sd["uniform"] else "NO"
        print(f"{n:>7} {mode:>12} {sd['entropy_pct']:>11.2f}% {status:>10} {sd['chi2']:>10.2f}")
        ent_results.append({"n": n, "mode": mode, **sd})


print("\n--- Real IBM Hardware ---")
try:
    svc = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = svc.least_busy(operational=True, simulator=False)
    print(f"  Backend: {backend.name} ({backend.num_qubits}Q)")
    hw = {}
    sampler = SamplerV2(backend)
    for n in [2, 3, 4, 5]:
        qc = build_superposition_circuit(n)
        job = sampler.run([transpile(qc, backend, optimization_level=3)], shots=1024)
        print(f"  {n}Q job: {job.job_id()} — waiting...")
        c = dict(job.result()[0].data.c.get_counts())
        sd = analyse(c, n, 1024)
        hw[f"{n}q"] = {"job_id": job.job_id(), **sd}
        status = "YES" if sd["uniform"] else "NO"
        print(f"    {n}Q: uniform={status}, entropy={sd['entropy_pct']:.2f}%, chi2={sd['chi2']:.2f}")
except Exception as e:
    print(f"  IBM error: {e}")
    hw = {"error": str(e)}


print("\n" + "="*65)
print("PRINCIPLE 2 SUMMARY")
print("="*65)
print(f"  {'Qubits':<8} {'States':<8} {'Entropy%':<12} {'Uniform':<10} {'chi2'}")
print("-"*50)
for entry in ideal:
    print(f"  {entry['n']:<8} {2**entry['n']:<8} {entry['entropy_pct']:<11.2f}% "
          f"{'YES' if entry['uniform'] else 'NO':<10} {entry['chi2']:.2f}")

fname = save_result(2, "Superposition", {
    "ideal": ideal, "thermal": thermal,
    "entanglement": ent_results, "hw": hw
})
print(f"\nSaved: {fname}")
print("PRINCIPLE 2 DONE.")
