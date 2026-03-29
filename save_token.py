from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="2WNgkMhW0o_c_gFXQRQ6MVZ5M-5mhchsl_pC8dcb1mRC",
    overwrite=True
)

print("Token saved successfully.")
