import json, os, glob

results_dir = "results"
all_files = glob.glob(f"{results_dir}/*.json")

summary = {
    "platform": "Raspberry Pi 5 8GB",
    "quantum_backend": "IBM Quantum (ibm_torino / least busy)",
    "experiments": []
}

for f in sorted(all_files):
    with open(f) as fp:
        data = json.load(fp)
    summary["experiments"].append({
        "circuit": data.get("circuit"),
        "timestamp": data.get("timestamp"),
        "backend": data.get("backend"),
        "job_id": data.get("job_id"),
        "shots": data.get("shots"),
        "simulator_result": data.get("simulator_result"),
        "real_hardware_result": data.get("real_hardware_result"),
        "noise_percentage": data.get("noise_percentage")
    })
    print(f"Loaded: {data.get('circuit')} — noise: {data.get('noise_percentage')}%")

with open("results/FULL_SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nFULL SUMMARY:")
print(json.dumps(summary, indent=2))
print("\nSaved to results/FULL_SUMMARY.json")
