"""
QSimEdge IoT PQC Handshake Simulator
Simulates a complete device authentication handshake using:
- PQC (Dilithium) — quantum safe
- Classical (RSA) — quantum vulnerable
Measures total round-trip authentication time for IoT context.
"""

import time, json, datetime, platform

def get_temp():
    try: return float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000
    except: return 0.0

print("="*60)
print("QSimEdge IoT PQC Handshake Latency Simulator")
print("Simulating complete device authentication protocol")
print("="*60)
print(f"Platform: {platform.machine()} | Temp: {get_temp()}°C\n")

results = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "platform": platform.machine(), "handshakes": []}

# Simulate IoT device authentication:
# 1. Device generates keypair (startup)
# 2. Server sends challenge
# 3. Device signs challenge
# 4. Server verifies signature
# 5. Authentication complete

iters = 20
challenge = b"IoT_AUTH_CHALLENGE_" + b"X" * 32  # 51 byte challenge

print("--- Dilithium2 IoT Authentication ---")
from dilithium_py.dilithium import Dilithium2

# Phase 1: Key generation (done once at device startup)
t=time.perf_counter()
for _ in range(iters): pk,sk=Dilithium2.keygen()
kg_time = (time.perf_counter()-t)/iters*1000

# Phase 2: Sign challenge (per authentication)
t=time.perf_counter()
for _ in range(iters): sig=Dilithium2.sign(sk,challenge)
sign_time = (time.perf_counter()-t)/iters*1000

# Phase 3: Verify (server side)
t=time.perf_counter()
for _ in range(iters): Dilithium2.verify(pk,challenge,sig)
verify_time = (time.perf_counter()-t)/iters*1000

total_first = kg_time + sign_time + verify_time
total_repeat = sign_time + verify_time

print(f"  Keygen (device startup, once): {kg_time:.2f}ms")
print(f"  Sign challenge (per auth):     {sign_time:.2f}ms")
print(f"  Verify (server side):          {verify_time:.2f}ms")
print(f"  First authentication total:    {total_first:.2f}ms")
print(f"  Repeat authentication total:   {total_repeat:.2f}ms")
print(f"  Auths per second (repeat):     {1000/total_repeat:.1f}")

results["handshakes"].append({
    "algorithm": "Dilithium2", "standard": "FIPS 204", "quantum_safe": True,
    "keygen_ms": round(kg_time,3), "sign_ms": round(sign_time,3),
    "verify_ms": round(verify_time,3),
    "first_auth_total_ms": round(total_first,3),
    "repeat_auth_ms": round(total_repeat,3),
    "auths_per_second": round(1000/total_repeat,2)
})

print("\n--- RSA-2048 IoT Authentication ---")
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256

t=time.perf_counter()
for _ in range(10): key=RSA.generate(2048)
kg_rsa = (time.perf_counter()-t)/10*1000

t=time.perf_counter()
for _ in range(iters):
    h=SHA256.new(challenge); pkcs1_15.new(key).sign(h)
sign_rsa = (time.perf_counter()-t)/iters*1000

t=time.perf_counter()
for _ in range(iters):
    try: pkcs1_15.new(key.publickey()).verify(SHA256.new(challenge),pkcs1_15.new(key).sign(SHA256.new(challenge)))
    except: pass
verify_rsa = (time.perf_counter()-t)/iters*1000

total_first_rsa = kg_rsa + sign_rsa + verify_rsa
total_repeat_rsa = sign_rsa + verify_rsa

print(f"  Keygen (device startup, once): {kg_rsa:.2f}ms")
print(f"  Sign challenge (per auth):     {sign_rsa:.2f}ms")
print(f"  Verify (server side):          {verify_rsa:.2f}ms")
print(f"  First authentication total:    {total_first_rsa:.2f}ms")
print(f"  Repeat authentication total:   {total_repeat_rsa:.2f}ms")
print(f"  Auths per second (repeat):     {1000/total_repeat_rsa:.1f}")

results["handshakes"].append({
    "algorithm": "RSA-2048", "standard": "BROKEN BY SHOR'S", "quantum_safe": False,
    "keygen_ms": round(kg_rsa,3), "sign_ms": round(sign_rsa,3),
    "verify_ms": round(verify_rsa,3),
    "first_auth_total_ms": round(total_first_rsa,3),
    "repeat_auth_ms": round(total_repeat_rsa,3),
    "auths_per_second": round(1000/total_repeat_rsa,2)
})

print(f"\n{'='*60}")
print("IoT AUTHENTICATION COMPARISON")
print(f"{'='*60}")
print(f"{'Metric':<35} {'Dilithium2':>15} {'RSA-2048':>15} {'Winner'}")
print("-"*70)
dil = results["handshakes"][0]
rsa = results["handshakes"][1]
rows = [
    ("Keygen (startup cost, ms)", dil['keygen_ms'], rsa['keygen_ms']),
    ("Sign (per auth, ms)", dil['sign_ms'], rsa['sign_ms']),
    ("Verify (per auth, ms)", dil['verify_ms'], rsa['verify_ms']),
    ("First auth latency (ms)", dil['first_auth_total_ms'], rsa['first_auth_total_ms']),
    ("Repeat auth latency (ms)", dil['repeat_auth_ms'], rsa['repeat_auth_ms']),
    ("Auths per second", dil['auths_per_second'], rsa['auths_per_second']),
]
for metric,d,r in rows:
    if "per second" in metric:
        winner = "Dilithium2" if d>r else "RSA-2048"
    else:
        winner = "Dilithium2" if d<r else "RSA-2048"
    print(f"{metric:<35} {d:>15} {r:>15} {winner}")

print(f"\nQuantum Safety:")
print(f"  Dilithium2: QUANTUM SAFE  (NIST FIPS 204)")
print(f"  RSA-2048:   QUANTUM BROKEN (Shor's algorithm breaks in ~8 hours on fault-tolerant QC)")
print(f"\nConclusion: Dilithium2 is {round(rsa['keygen_ms']/dil['keygen_ms'])}x faster at startup")
print(f"            AND quantum safe. RSA has no future in post-quantum IoT.")

ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname=f"results/iot_handshake_{ts}.json"
with open(fname,"w") as f: json.dump(results,f,indent=2)
print(f"\nSaved: {fname}")
print("DONE.")
