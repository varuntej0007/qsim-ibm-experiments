"""
QSimEdge RSA Attack Demo (Optimized)
Fast, realistic demonstration of RSA security scaling
Uses Pollard Rho instead of brute force
"""

import time, json, datetime, random
from Crypto.Util.number import getPrime
import math

# -------------------------------
# Pollard Rho (fast factoring)
# -------------------------------
def pollard_rho(n, max_iter=100000):
    if n % 2 == 0:
        return 2

    def f(x): return (x*x + 1) % n

    for _ in range(5):  # retry attempts
        x = random.randint(2, n-2)
        y = x
        d = 1

        for _ in range(max_iter):
            x = f(x)
            y = f(f(y))
            d = math.gcd(abs(x - y), n)

            if d == 1:
                continue
            if d == n:
                break
            return d

    return None


# -------------------------------
# Main Demo
# -------------------------------
print("="*60)
print("QSimEdge RSA Attack Demo (Optimized)")
print("="*60)
print("Demonstrating classical attack limits on RSA\n")

results = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "experiments": []
}

# Keep sizes realistic for demo
test_sizes = [32, 48, 64, 96]

for bits in test_sizes:
    print(f"\n--- RSA-{bits} ---")

    # Generate key FAST
    p = getPrime(bits//2)
    q = getPrime(bits//2)
    n = p * q

    print(f"  Modulus bit-length: {n.bit_length()}")

    start = time.perf_counter()

    factor = pollard_rho(n)

    elapsed = (time.perf_counter() - start) * 1000

    if factor:
        print(f"  FACTORED in {elapsed:.2f} ms")
        print(f"  Factor found: {factor}")
        success = True
    else:
        print(f"  Could not factor quickly (>{elapsed:.2f} ms)")
        print("  Demonstrates computational hardness")
        success = False

    results["experiments"].append({
        "bits": bits,
        "factored": success,
        "time_ms": round(elapsed, 2)
    })


# -------------------------------
# PQC Explanation
# -------------------------------
print("\n" + "="*60)
print("Why PQC is Needed")
print("="*60)

print("""
RSA:
- Security = factoring n = p × q
- Broken by Shor's Algorithm (quantum)

PQC (Dilithium, Kyber):
- Based on lattice problems (MLWE)
- No efficient quantum attacks known
- Secure for post-quantum era
""")


# -------------------------------
# Comparison Table
# -------------------------------
print("Scaling Comparison:")
print(f"{'Algorithm':<15} {'Security Basis':<25} {'Quantum Safe?'}")
print("-"*60)

data = [
    ("RSA-2048", "Factoring", "NO"),
    ("RSA-4096", "Factoring", "NO"),
    ("Dilithium2", "Lattice (MLWE)", "YES"),
    ("Kyber", "Lattice (MLWE)", "YES"),
]

for row in data:
    print(f"{row[0]:<15} {row[1]:<25} {row[2]}")


# -------------------------------
# Save results
# -------------------------------
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname = f"rsa_attack_results_{ts}.json"

with open(fname, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results to: {fname}")
print("DONE.")
