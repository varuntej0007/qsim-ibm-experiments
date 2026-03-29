from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform")

# List all available real quantum computers
backends = service.backends(
    filters=lambda b: not b.configuration().simulator
    and b.status().operational
)

print("\nAvailable real quantum computers:")
for b in backends:
    status = b.status()
    print(f"  {b.name} — {b.configuration().n_qubits} qubits — queue: {status.pending_jobs} jobs")

# Find the least busy one
best = service.least_busy(operational=True, simulator=False)
print(f"\nLeast busy right now: {best.name}")
