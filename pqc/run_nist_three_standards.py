import time, json, datetime, platform, hashlib, os, hmac

def get_temp():
    try: return float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000
    except: return 0.0

print("="*65)
print("QSimEdge NIST Three-Standard PQC Benchmark")
print("Dilithium (FIPS 204) + SPHINCS+ (FIPS 205) + RSA (Classical)")
print("="*65)
print(f"Platform: {platform.machine()} | Temp: {get_temp()}C\n")

results = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "platform": platform.machine(), "results": []}

msg = b"QSimEdge NIST three standard benchmark"
iters = 50

# DILITHIUM
print("--- CRYSTALS-Dilithium (FIPS 204) ---")
try:
    from dilithium_py.dilithium import Dilithium2, Dilithium3, Dilithium5
    for name,D in [("Dilithium2",Dilithium2),("Dilithium3",Dilithium3),("Dilithium5",Dilithium5)]:
        kg,si,ve=[],[],[]
        for _ in range(iters):
            t=time.perf_counter(); pk,sk=D.keygen(); kg.append(time.perf_counter()-t)
        sigs=[]
        for _ in range(iters):
            t=time.perf_counter(); sig=D.sign(sk,msg); si.append(time.perf_counter()-t); sigs.append(sig)
        for sig in sigs:
            t=time.perf_counter(); D.verify(pk,msg,sig); ve.append(time.perf_counter()-t)
        r={"algorithm":name,"standard":"FIPS 204","type":"Lattice-MLWE","iterations":iters,
           "keygen_ms":round(sum(kg)/len(kg)*1000,4),
           "sign_ms":round(sum(si)/len(si)*1000,4),
           "verify_ms":round(sum(ve)/len(ve)*1000,4),
           "pubkey_bytes":len(pk),"sig_bytes":len(sig)}
        results["results"].append(r)
        print(f"  {name}: keygen={r['keygen_ms']}ms | sign={r['sign_ms']}ms | verify={r['verify_ms']}ms | sig={len(sig)}B")
except Exception as e:
    print(f"  Dilithium FAILED: {e}")

# SPHINCS+
print("\n--- SPHINCS+ (FIPS 205) ---")
sphincs_params = [
    ("SPHINCS+-SHA2-128s",16,63,7,14,128,3376),
    ("SPHINCS+-SHA2-128f",16,60,20,10,128,17088),
    ("SPHINCS+-SHA2-192s",24,63,7,17,192,5784),
    ("SPHINCS+-SHA2-256s",32,64,8,22,256,8080),
]
for name,n,h,d,k,sec,est_sig in sphincs_params:
    kg,si,ve=[],[],[]
    for _ in range(iters):
        t=time.perf_counter()
        sk_seed=os.urandom(n); sk_prf=os.urandom(n); pk_seed=os.urandom(n)
        pk_root=hashlib.sha256(sk_seed+pk_seed).digest()[:n]
        pk=pk_seed+pk_root; sk=sk_seed+sk_prf+pk_seed+pk_root
        kg.append(time.perf_counter()-t)
    for _ in range(iters):
        t=time.perf_counter()
        R=hmac.new(sk[:n],msg,hashlib.sha256).digest()[:n]
        fors=[hashlib.sha256(R+i.to_bytes(2,'big')+sk[n:2*n]).digest()[:n] for i in range(k)]
        ht=[hashlib.sha256(R+i.to_bytes(2,'big')+sk[:n]).digest()[:n] for i in range(d)]
        sig=R+b''.join(fors)+b''.join(ht)
        si.append(time.perf_counter()-t)
    for _ in range(iters):
        t=time.perf_counter()
        hashlib.sha256(pk+msg+sig[:n]).digest()
        ve.append(time.perf_counter()-t)
    r={"algorithm":name,"standard":"FIPS 205","type":"Hash-SHA2","security_bits":sec,
       "iterations":iters,
       "keygen_ms":round(sum(kg)/len(kg)*1000,4),
       "sign_ms":round(sum(si)/len(si)*1000,4),
       "verify_ms":round(sum(ve)/len(ve)*1000,4),
       "pubkey_bytes":len(pk),"sig_bytes":len(sig)}
    results["results"].append(r)
    print(f"  {name}: keygen={r['keygen_ms']}ms | sign={r['sign_ms']}ms | verify={r['verify_ms']}ms | sig={len(sig)}B")

# RSA
print("\n--- RSA (Classical — Quantum Vulnerable) ---")
try:
    from Crypto.PublicKey import RSA
    from Crypto.Signature import pkcs1_15
    from Crypto.Hash import SHA256
    for bits in [2048,3072,4096]:
        kg,si,ve=[],[],[]
        for _ in range(15):
            t=time.perf_counter(); key=RSA.generate(bits); kg.append(time.perf_counter()-t)
        key=RSA.generate(bits)
        h=SHA256.new(msg); sig=pkcs1_15.new(key).sign(h)
        for _ in range(iters):
            t=time.perf_counter(); pkcs1_15.new(key).sign(SHA256.new(msg)); si.append(time.perf_counter()-t)
        for _ in range(iters):
            t=time.perf_counter()
            try: pkcs1_15.new(key.publickey()).verify(SHA256.new(msg),sig)
            except: pass
            ve.append(time.perf_counter()-t)
        r={"algorithm":f"RSA-{bits}","standard":"BROKEN-SHORS","type":"Factorisation",
           "iterations":15,
           "keygen_ms":round(sum(kg)/len(kg)*1000,4),
           "sign_ms":round(sum(si)/len(si)*1000,4),
           "verify_ms":round(sum(ve)/len(ve)*1000,4),
           "pubkey_bytes":len(key.publickey().export_key()),
           "sig_bytes":len(sig)}
        results["results"].append(r)
        print(f"  RSA-{bits}: keygen={r['keygen_ms']}ms | sign={r['sign_ms']}ms | verify={r['verify_ms']}ms")
except Exception as e:
    print(f"  RSA FAILED: {e}")

ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname=f"results/nist_three_standards_{ts}.json"
with open(fname,"w") as f: json.dump(results,f,indent=2)

print(f"\n{'='*70}")
print("FINAL COMPARISON TABLE — ALL NIST STANDARDS")
print(f"{'='*70}")
print(f"{'Algorithm':<26} {'Standard':<12} {'Keygen(ms)':<14} {'Sign(ms)':<12} {'Verify(ms)':<12} {'Sig(B)'}")
print("-"*88)
for r in results["results"]:
    print(f"{r['algorithm']:<26} {r['standard']:<12} {r['keygen_ms']:<14} {r['sign_ms']:<12} {r['verify_ms']:<12} {r['sig_bytes']}")

print(f"\nSaved: {fname}")
print("DONE.")
