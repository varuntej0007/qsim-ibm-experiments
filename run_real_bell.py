from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import json
import datetime

# ── 1. BUILD THE BELL STATE CIRCUIT ──────────────────────────────
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

print("Circuit built:")
print(qc.draw())

# ── 2. RUN ON YOUR LOCAL AERSIMULATOR (PERFECT — NO NOISE) ───────
print("\nRunning on AerSimulator (perfect, no noise)...")
sim = AerSimulator()
sim_job = sim.run(transpile(qc, sim), shots=1024)
sim_result = sim_job.result()
sim_counts = sim_result.get_counts()
print(f"AerSimulator result: {sim_counts}")

# ── 3. CONNECT TO IBM REAL HARDWARE ──────────────────────────────
print("\nConnecting to IBM Quantum...")
service = QiskitRuntimeService(channel="ibm_quantum_platform")
real_backend = service.least_busy(operational=True, simulator=False)
print(f"Using real quantum computer: {real_backend.name}")
print("Submitting job to real hardware (this may take a few minutes)...")

# Transpile for the real hardware
qc_transpiled = transpile(qc, real_backend, optimization_level=1)

# Run using SamplerV2
sampler = SamplerV2(real_backend)
job = sampler.run([qc_transpiled], shots=1024)

print(f"Job ID: {job.job_id()}")
print("Waiting for real hardware result...")

# Get result (waits until done)
real_result = job.result()

# Extract counts from SamplerV2 result
pub_result = real_result[0]
counts_array = pub_result.data.c.get_counts()
real_counts = dict(counts_array)

print(f"\nReal hardware result: {real_counts}")

# ── 4. COMPARE AND SAVE ───────────────────────────────────────────
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"Perfect simulation : {sim_counts}")
print(f"Real quantum HW    : {real_counts}")

# Calculate noise — states that should be 0 but aren't
noise_states = {k: v for k, v in real_counts.items() if k not in ['00', '11']}
total_shots = sum(real_counts.values())
noise_pct = sum(noise_states.values()) / total_shots * 100 if noise_states else 0

print(f"\nNoise states (should be zero): {noise_states}")
print(f"Total noise percentage: {noise_pct:.2f}%")

# Save full results to JSON
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    "timestamp": timestamp,
    "backend": real_backend.name,
    "job_id": job.job_id(),
    "shots": 1024,
    "circuit": "Bell State 2Q",
    "simulator_result": sim_counts,
    "real_hardware_result": real_counts,
    "noise_states": noise_states,
    "noise_percentage": round(noise_pct, 4)
}

filename = f"results/bell_state_{timestamp}.json"
with open(filename, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nFull results saved to: {filename}")
print("\nDONE. You just ran a circuit on a real quantum computer.")
