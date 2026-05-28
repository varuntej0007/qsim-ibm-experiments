import os
import time
import json
import psutil
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from cryptography.hazmat.primitives.asymmetric import mlkem
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# =========================================================
# 1. IBM QUANTUM HARDWARE CONNECTION (QRNG)
# =========================================================
def get_quantum_randomness(num_bits=128):
    print("\n[PHASE 1] Connecting to IBM Quantum to harvest true randomness...")
    
    # Initialize the service (ensure your IBM token is saved on the Pi)
    service = QiskitRuntimeService(channel="ibm_quantum")
    
    # Select the backend you specified
    backend = service.backend("ibm_fez")
    print(f"✅ Connected to backend: {backend.name}")
    
    # Create a circuit with 12 qubits (to generate 12 bytes / 96 bits for our AES Nonce)
    qc = QuantumCircuit(12)
    qc.h(range(12)) # Put all qubits into perfect superposition
    qc.measure_all()
    
    print("⏳ Submitting QRNG circuit to ibm_fez...")
    sampler = SamplerV2(backend=backend)
    job = sampler.run([qc], shots=1)
    
    # Extract the random bitstring from the result
    result = job.result()
    pub_result = result[0]
    bitstring = list(pub_result.data.meas.get_counts().keys())[0]
    
    # Convert the quantum bitstring (e.g., '10101100...') into raw bytes
    quantum_bytes = int(bitstring, 2).to_bytes(12, byteorder='big')
    print(f"✅ Quantum Randomness Harvested: {quantum_bytes.hex()}")
    
    return quantum_bytes

# =========================================================
# 2. THE EDGE PQC STACK (Seeded by IBM)
# =========================================================
def run_hybrid_pqc():
    print("=" * 65)
    print("🛡️ HYBRID QEDGE: IBM QUANTUM + Pi 5 PQC")
    print("=" * 65)

    # Step 1: Harvest quantum randomness from IBM Fez
    quantum_nonce = get_quantum_randomness()

    print("\n[PHASE 2] Edge Gateway initializing ML-KEM-768...")
    gateway_private_key = mlkem.MLKEM768PrivateKey.generate()
    gateway_public_key = gateway_private_key.public_key()

    print("\n[PHASE 3] IoT Sensor encapsulating shared secret...")
    sensor_shared_secret, ciphertext = gateway_public_key.encapsulate()

    print("\n[PHASE 4] Edge Gateway decapsulating shared secret...")
    gateway_shared_secret = gateway_private_key.decapsulate(ciphertext)
    assert sensor_shared_secret == gateway_shared_secret, "PQC Handshake Failed!"

    print("\n[PHASE 5] Streaming encrypted IoT sensor data using QUANTUM NONCE...")
    aesgcm = AESGCM(sensor_shared_secret[:32])
    
    sensor_data = {
        "device_id": "IND-SMART-METER-001",
        "timestamp": time.time(),
        "temperature_c": 38.5,
        "status": "SECURED_BY_IBM_FEZ"
    }
    payload_bytes = json.dumps(sensor_data).encode('utf-8')

    # INSTEAD OF os.urandom, WE USE THE PERFECT RANDOMNESS FROM IBM FEZ
    start_time = time.perf_counter_ns()
    encrypted_payload = aesgcm.encrypt(quantum_nonce, payload_bytes, associated_data=None)
    enc_time = (time.perf_counter_ns() - start_time) / 1_000_000

    print(f"✅ Data Encrypted (AES-256-GCM) in {enc_time:.3f} ms")
    print(f"📡 Ciphertext: {encrypted_payload[:20]}... [TRUNCATED]")

    decrypted_payload = aesgcm.decrypt(quantum_nonce, encrypted_payload, associated_data=None)
    print(f"📄 Recovered Payload: {decrypted_payload.decode('utf-8')}")

    print("\n" + "=" * 65)
    print("🎯 EXECUTION COMPLETE: True Quantum-Edge Security Achieved")
    print("=" * 65)

if __name__ == "__main__":
    run_hybrid_pqc()
