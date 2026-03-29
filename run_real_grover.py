from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import json, datetime

# Grover's — 2 qubit, target state |11>
qc = QuantumCircuit(2, 2)
# Superposition
qc.h([0, 1])
# Oracle — marks |11>
qc.cz(0, 1)
# Diffusion operator
qc.h([0, 1])
qc.x([0, 1])
qc.cz(0, 1)
qc.x([0, 1])
qc.h([0, 1])
qc.measure([0, 1], [0, 1])

print("Circuit:"); print(qc.draw())

sim = AerSimulator()
sim_counts = sim.run(transpile(qc, sim), shots=1024).result().get_counts()
print(f"AerSimulator: {sim_counts}")

service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.least_busy(operational=True, simulator=False)
print(f"Using: {backend.name}")

qc_t = transpile(qc, backend, optimization_level=1)
sampler = SamplerV2(backend)
job = sampler.run([qc_t], shots=1024)
print(f"Job ID: {job.job_id()}")
print("Waiting...")

real_result = job.result()
real_counts = dict(real_result[0].data.c.get_counts())

target_state = '11'
target_sim = sim_counts.get(target_state, 0)
target_real = real_counts.get(target_state, 0)
total = sum(real_counts.values())
noise_pct = (total - target_real - real_counts.get('00',0)) / total * 100

print(f"\nCOMPARISON")
print(f"Perfect sim : {sim_counts}")
print(f"Real HW     : {real_counts}")
print(f"Target |11> sim: {target_sim/1024*100:.1f}%")
print(f"Target |11> real: {target_real/total*100:.1f}%")
print(f"Noise %: {noise_pct:.2f}%")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    "timestamp": timestamp, "circuit": "Grover Search 2Q target=11",
    "backend": backend.name, "job_id": job.job_id(),
    "shots": 1024, "simulator_result": sim_counts,
    "real_hardware_result": real_counts,
    "target_state": target_state,
    "target_probability_sim": round(target_sim/1024, 4),
    "target_probability_real": round(target_real/total, 4),
    "noise_percentage": round(noise_pct, 4)
}
with open(f"results/grover_{timestamp}.json","w") as f:
    json.dump(output, f, indent=2)
print("Saved. DONE.")
