from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import subprocess, time, json, datetime

def get_pi_temp():
    result = subprocess.run(
        ['cat', '/sys/class/thermal/thermal_zone0/temp'],
        capture_output=True, text=True
    )
    return float(result.stdout.strip()) / 1000

def run_bell_with_noise(error_rate):
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(error_rate, 1), ['h'])
    noise.add_all_qubit_quantum_error(depolarizing_error(error_rate*2, 2), ['cx'])
    sim = AerSimulator(noise_model=noise)
    return sim.run(transpile(qc, sim), shots=1024).result().get_counts()

readings = []
print("Recording 10 temperature readings with simulation runs...")
print("This will take about 2 minutes.\n")

for i in range(10):
    temp = get_pi_temp()
    error_rate = max(0.001, 0.001 + (temp - 50) * 0.0005)
    counts = run_bell_with_noise(error_rate)
    noise_states = {k:v for k,v in counts.items() if k not in ['00','11']}
    noise_pct = sum(noise_states.values())/1024*100

    reading = {
        "reading": i+1,
        "cpu_temp_celsius": temp,
        "error_rate_applied": round(error_rate, 6),
        "bell_state_counts": counts,
        "noise_percentage": round(noise_pct, 4)
    }
    readings.append(reading)
    print(f"Reading {i+1}: Temp={temp}C | Error rate={error_rate:.4f} | Noise={noise_pct:.2f}% | Counts={counts}")
    time.sleep(10)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    "platform": "Raspberry Pi 5 8GB",
    "description": "Environmental noise model — CPU temperature drives decoherence rate",
    "hypothesis": "Higher CPU temperature correlates with higher simulated quantum noise",
    "readings": readings
}
with open(f"results/thermal_noise_{timestamp}.json","w") as f:
    json.dump(output, f, indent=2)

print(f"\nAll readings saved to results/thermal_noise_{timestamp}.json")
print("DONE.")
