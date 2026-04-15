import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from arq_utils import *
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from scipy.optimize import minimize
import time

print_header(1, "AI-DRIVEN ADAPTIVE SUPERPOSITION MANAGEMENT")

class AdiabaticRegulator:
    def __init__(self, n=4):
        self.n = n
        self.angles = np.ones(n) * np.pi/2

    def read_env(self):
        t = get_temp()
        return {"temp": t, "load": get_cpu_load(),
                "noise": compute_thermal_noise_rate(t),
                "t2": thermal_decoherence_time(t)}

    def compute_angles(self, env):
        base = np.pi/2
        tc = (env["temp"] - 40) * 0.008
        lc = env["load"] * 0.05
        angles = np.array([
            np.clip(base - tc + lc * np.sin(i * np.pi * 0.03), np.pi/4, 3*np.pi/4)
            for i in range(self.n)
        ])
        self.angles = angles
        return angles

    def build_circuit(self, angles):
        qc = QuantumCircuit(self.n, self.n)
        for i, a in enumerate(angles): qc.ry(a, i)
        qc.measure_all()
        return qc

    def build_perfect(self):
        qc = QuantumCircuit(self.n, self.n)
        qc.h(range(self.n)); qc.measure_all()
        return qc

    def uniformity(self, counts, shots):
        n_states = 2**self.n
        exp = shots / n_states
        dev = sum(abs(counts.get(format(i, f'0{self.n}b'), 0) - exp)
                  for i in range(n_states))
        return max(0.0, 1.0 - dev / (2 * shots))

reg = AdiabaticRegulator(4)
sim = AerSimulator()
SHOTS = 2048

print("\n--- Environment Reading ---")
env = reg.read_env()
for k,v in env.items(): print(f"  {k}: {v}")

print("\n--- AI Angle Computation ---")
angles = reg.compute_angles(env)
for i,a in enumerate(angles):
    print(f"  Q{i}: {a:.4f} rad  (P0={np.cos(a/2)**2:.3f}  P1={np.sin(a/2)**2:.3f})")

print("\n--- Temperature Sweep ---")
print(f"{'Temp':>6} {'Noise':>10} {'Perfect U':>12} {'AI Adapt U':>12} {'Gain':>8}")
print("-"*52)
results = []
for temp in [30,40,45,50,55,65,75,85]:
    rate = compute_thermal_noise_rate(temp)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(rate,1),['ry','h'])
    nm.add_all_qubit_quantum_error(depolarizing_error(rate*8,2),['cx'])
    sn = AerSimulator(noise_model=nm)
    fake_env = {"temp": temp, "load": 0.5, "noise": rate, "t2": thermal_decoherence_time(temp)}
    ai_ang = reg.compute_angles(fake_env)
    qcp = reg.build_perfect()
    cp = sn.run(transpile(qcp,sn), shots=SHOTS).result().get_counts()
    up = reg.uniformity(cp, SHOTS)
    qca = reg.build_circuit(ai_ang)
    ca = sn.run(transpile(qca,sn), shots=SHOTS).result().get_counts()
    ua = reg.uniformity(ca, SHOTS)
    gain = ua - up
    results.append({"temp": temp, "noise": rate, "u_perfect": round(up,4),
                    "u_adaptive": round(ua,4), "gain": round(gain,4)})
    print(f"{temp:>5}°C {rate:>10.6f} {up:>12.4f} {ua:>12.4f} {gain:>+8.4f}")

print("\n--- Optimisation ---")
def cost(a):
    t = get_temp(); r = compute_thermal_noise_rate(t)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(r,1),['ry'])
    s = AerSimulator(noise_model=nm)
    qc = reg.build_circuit(a)
    c = s.run(transpile(qc,s), shots=512).result().get_counts()
    return 1 - reg.uniformity(c, 512)

res = minimize(cost, angles, method='COBYLA', options={'maxiter':60,'rhobeg':0.2})
print(f"  Converged: {res.nfev} evals | Uniformity: {1-res.fun:.4f}")

print("\n--- Real IBM Hardware ---")
try:
    svc = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = svc.least_busy(operational=True, simulator=False)
    print(f"  Backend: {backend.name}")
    qcp_hw = reg.build_perfect()
    qca_hw = reg.build_circuit(res.x)
    sampler = SamplerV2(backend)
    jp = sampler.run([transpile(qcp_hw, backend, optimization_level=3)], shots=1024)
    print(f"  Perfect job: {jp.job_id()} — waiting...")
    cp_hw = dict(jp.result()[0].data.c.get_counts())
    up_hw = reg.uniformity(cp_hw, 1024)
    ja = sampler.run([transpile(qca_hw, backend, optimization_level=3)], shots=1024)
    print(f"  Adaptive job: {ja.job_id()} — waiting...")
    ca_hw = dict(ja.result()[0].data.c.get_counts())
    ua_hw = reg.uniformity(ca_hw, 1024)
    print(f"  Perfect uniformity:  {up_hw:.4f}")
    print(f"  Adaptive uniformity: {ua_hw:.4f}")
    print(f"  AI gain real HW:     {ua_hw-up_hw:+.4f}")
    hw = {"backend": backend.name, "job_perfect": jp.job_id(),
          "job_adaptive": ja.job_id(), "u_perfect": round(up_hw,4),
          "u_adaptive": round(ua_hw,4)}
except Exception as e:
    print(f"  IBM error: {e}"); hw = {"error": str(e)}

fname = save_result(1, "Adaptive Superposition",
                    {"env": env, "temp_sweep": results,
                     "opt_evals": res.nfev, "real_hw": hw})
print(f"\nSaved: {fname}")
print("PRINCIPLE 1 DONE.")
