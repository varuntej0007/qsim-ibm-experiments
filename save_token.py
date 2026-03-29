from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="Your_Token",
    overwrite=True
)

print("Token saved successfully.")
