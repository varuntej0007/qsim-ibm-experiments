from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import json, datetime

# 3-qubit GHZ State
qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.measure([0,1,2], [0,1,2])

print("Circuit:"); print(qc.draw())

# Perfect simulation
sim = AerSimulator()
sim_counts = sim.run(transpile(qc, sim), shots=1024).result().get_counts()
print(f"AerSimulator: {sim_counts}")

# Real hardware
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
print(f"Real hardware: {real_counts}")

noise_states = {k:v for k,v in real_counts.items() if k not in ['000','111']}
noise_pct = sum(noise_states.values())/sum(real_counts.values())*100

print(f"\nCOMPARISON")
print(f"Perfect sim : {sim_counts}")
print(f"Real HW     : {real_counts}")
print(f"Noise states: {noise_states}")
print(f"Noise %     : {noise_pct:.2f}%")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    "timestamp": timestamp, "circuit": "GHZ State 3Q",
    "backend": backend.name, "job_id": job.job_id(),
    "shots": 1024, "simulator_result": sim_counts,
    "real_hardware_result": real_counts,
    "noise_states": noise_states, "noise_percentage": round(noise_pct,4)
}
with open(f"results/ghz_state_{timestamp}.json","w") as f:
    json.dump(output, f, indent=2)
print(f"Saved. DONE.")
