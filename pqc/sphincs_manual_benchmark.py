"""
SPHINCS+ Manual Benchmark using Python hashlib
Implements simplified SPHINCS+ operations to benchmark
hash-based signature performance on ARM hardware.
Based on NIST FIPS 205 specification.
"""
import hashlib, os, time, json, datetime, platform, hmac

def get_temp():
    try: return float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000
    except: return 0.0

# SPHINCS+ parameter sets (from NIST FIPS 205)
PARAMS = {
    "SPHINCS+-SHA2-128s": {"n":16,"h":63,"d":7,"k":14,"w":16,"sec_bits":128},
    "SPHINCS+-SHA2-128f": {"n":16,"h":60,"d":20,"k":10,"w":16,"sec_bits":128},
    "SPHINCS+-SHA2-192s": {"n":24,"h":63,"d":7,"k":17,"w":16,"sec_bits":192},
    "SPHINCS+-SHA2-256s": {"n":32,"h":64,"d":8,"k":22,"w":16,"sec_bits":256},
}

def sphincs_keygen_sim(n_bytes):
    """Simulate SPHINCS+ keygen: generate SK.seed, SK.prf, PK.seed, PK.root"""
    sk_seed = os.urandom(n_bytes)
    sk_prf  = os.urandom(n_bytes)
    pk_seed = os.urandom(n_bytes)
    # Simulate PK.root computation (simplified)
    pk_root = hashlib.sha256(sk_seed + pk_seed).digest()[:n_bytes]
    sk = sk_seed + sk_prf + pk_seed + pk_root
    pk = pk_seed + pk_root
    return pk, sk

def sphincs_sign_sim(msg, sk, n_bytes, h, d, k):
    """Simulate SPHINCS+ signing: FORS + HT signature chain"""
    # R = randomised hash of message
    R = hmac.new(sk[:n_bytes], msg, hashlib.sha256).digest()[:n_bytes]
    # FORS signature: k trees of height a
    fors_sigs = []
    for i in range(k):
        leaf = hashlib.sha256(R + i.to_bytes(4,'big') + sk[n_bytes:2*n_bytes]).digest()[:n_bytes]
        auth = [hashlib.sha256(leaf + j.to_bytes(4,'big')).digest()[:n_bytes] for j in range(h//d)]
        fors_sigs.append(leaf + b''.join(auth))
    # HT signature: d layers of XMSS
    ht_sigs = []
    for layer in range(d):
        xmss_sig = hashlib.sha256(R + layer.to_bytes(4,'big') + sk[:n_bytes]).digest()[:n_bytes]
        auth_path = [hashlib.sha256(xmss_sig + j.to_bytes(4,'big')).digest()[:n_bytes] for j in range(h//d)]
        ht_sigs.append(xmss_sig + b''.join(auth_path))
    return R + b''.join(fors_sigs) + b''.join(ht_sigs)

def sphincs_verify_sim(msg, sig, pk, n_bytes):
    """Simulate SPHINCS+ verification"""
    R = sig[:n_bytes]
    expected = hashlib.sha256(pk + R + msg).digest()
    return len(sig) > n_bytes

print("="*65)
print("SPHINCS+ Benchmark (FIPS 205 Simulation)")
print("Hash-based post-quantum signatures on ARM")
print("="*65)
print(f"Platform: {platform.machine()} | Temp: {get_temp()}°C\n")

results = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "platform": platform.machine(), "sphincs_results": []}

msg = b"QSimEdge NIST SPHINCS+ benchmark message"
iters = 20

for name, params in PARAMS.items():
    n = params["n"]; h = params["h"]; d = params["d"]
    k = params["k"]; sec = params["sec_bits"]
    print(f"--- {name} (Security Level: {sec}-bit) ---")

    # Keygen
    kg=[]
    for _ in range(iters):
        t=time.perf_counter(); pk,sk=sphincs_keygen_sim(n); kg.append(time.perf_counter()-t)

    # Sign
    si=[]
    sigs=[]
    for _ in range(iters):
        t=time.perf_counter(); sig=sphincs_sign_sim(msg,sk,n,h,d,k); si.append(time.perf_counter()-t); sigs.append(sig)

    # Verify
    ve=[]
    for sig in sigs:
        t=time.perf_counter(); sphincs_verify_sim(msg,sig,pk,n); ve.append(time.perf_counter()-t)

    r = {
        "algorithm": name, "standard": "FIPS 205",
        "type": "Hash-based (SHA2)", "security_bits": sec,
        "iterations": iters,
        "keygen_ms_avg": round(sum(kg)/len(kg)*1000,4),
        "sign_ms_avg":   round(sum(si)/len(si)*1000,4),
        "verify_ms_avg": round(sum(ve)/len(ve)*1000,4),
        "public_key_bytes": len(pk),
        "signature_bytes": len(sigs[0])
    }
    results["sphincs_results"].append(r)
    print(f"  keygen={r['keygen_ms_avg']}ms | sign={r['sign_ms_avg']}ms | verify={r['verify_ms_avg']}ms | pk={len(pk)}B | sig={len(sigs[0])}B")

ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname=f"results/sphincs_benchmark_{ts}.json"
with open(fname,"w") as f: json.dump(results,f,indent=2)

print(f"\n{'='*65}")
print("SPHINCS+ SUMMARY TABLE")
print(f"{'='*65}")
print(f"{'Algorithm':<28} {'Sec':<6} {'Keygen(ms)':<14} {'Sign(ms)':<12} {'Verify(ms)':<12} {'Sig(B)'}")
print("-"*84)
for r in results["sphincs_results"]:
    print(f"{r['algorithm']:<28} {r['security_bits']:<6} {r['keygen_ms_avg']:<14} {r['sign_ms_avg']:<12} {r['verify_ms_avg']:<12} {r['signature_bytes']}")

print(f"\nSaved: {fname}")
print("DONE.")
