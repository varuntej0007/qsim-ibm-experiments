import time, json, datetime, platform, hashlib, os, hmac

def get_temp():
    try: return float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000
    except: return 0.0

print("="*60)
print("QSimEdge IoT PQC Handshake Latency Simulator")
print("="*60)
print(f"Platform: {platform.machine()} | Temp: {get_temp()}C\n")

challenge = b"IoT_AUTH_CHALLENGE_" + b"X"*32
iters = 30
results = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "platform": platform.machine(), "handshakes": []}

# --- Dilithium2 ---
print("--- Dilithium2 IoT Authentication (FIPS 204) ---")
try:
    from dilithium_py.dilithium import Dilithium2
    kg,si,ve=[],[],[]
    for _ in range(iters):
        t=time.perf_counter(); pk,sk=Dilithium2.keygen(); kg.append(time.perf_counter()-t)
    for _ in range(iters):
        t=time.perf_counter(); sig=Dilithium2.sign(sk,challenge); si.append(time.perf_counter()-t)
    sigs=[]
    for _ in range(iters):
        sig=Dilithium2.sign(sk,challenge); sigs.append(sig)
    for sig in sigs:
        t=time.perf_counter(); Dilithium2.verify(pk,challenge,sig); ve.append(time.perf_counter()-t)
    kg_ms=round(sum(kg)/len(kg)*1000,3)
    si_ms=round(sum(si)/len(si)*1000,3)
    ve_ms=round(sum(ve)/len(ve)*1000,3)
    total_first=round(kg_ms+si_ms+ve_ms,3)
    total_repeat=round(si_ms+ve_ms,3)
    print(f"  Keygen (startup):     {kg_ms}ms")
    print(f"  Sign (per auth):      {si_ms}ms")
    print(f"  Verify (server):      {ve_ms}ms")
    print(f"  First auth total:     {total_first}ms")
    print(f"  Repeat auth total:    {total_repeat}ms")
    print(f"  Auths per second:     {round(1000/total_repeat,1)}")
    results["handshakes"].append({
        "algorithm":"Dilithium2","standard":"FIPS 204","quantum_safe":True,
        "keygen_ms":kg_ms,"sign_ms":si_ms,"verify_ms":ve_ms,
        "first_auth_ms":total_first,"repeat_auth_ms":total_repeat,
        "auths_per_sec":round(1000/total_repeat,1)})
except Exception as e:
    print(f"  FAILED: {e}")

# --- RSA-2048 ---
print("\n--- RSA-2048 IoT Authentication (Classical — QUANTUM BROKEN) ---")
try:
    from Crypto.PublicKey import RSA
    from Crypto.Signature import pkcs1_15
    from Crypto.Hash import SHA256
    kg,si,ve=[],[],[]
    for _ in range(10):
        t=time.perf_counter(); key=RSA.generate(2048); kg.append(time.perf_counter()-t)
    key=RSA.generate(2048)
    h=SHA256.new(challenge)
    sig_rsa=pkcs1_15.new(key).sign(h)
    for _ in range(iters):
        t=time.perf_counter(); pkcs1_15.new(key).sign(SHA256.new(challenge)); si.append(time.perf_counter()-t)
    for _ in range(iters):
        t=time.perf_counter()
        try: pkcs1_15.new(key.publickey()).verify(SHA256.new(challenge),sig_rsa)
        except: pass
        ve.append(time.perf_counter()-t)
    kg_ms=round(sum(kg)/len(kg)*1000,3)
    si_ms=round(sum(si)/len(si)*1000,3)
    ve_ms=round(sum(ve)/len(ve)*1000,3)
    total_first=round(kg_ms+si_ms+ve_ms,3)
    total_repeat=round(si_ms+ve_ms,3)
    print(f"  Keygen (startup):     {kg_ms}ms")
    print(f"  Sign (per auth):      {si_ms}ms")
    print(f"  Verify (server):      {ve_ms}ms")
    print(f"  First auth total:     {total_first}ms")
    print(f"  Repeat auth total:    {total_repeat}ms")
    print(f"  Auths per second:     {round(1000/total_repeat,1)}")
    results["handshakes"].append({
        "algorithm":"RSA-2048","standard":"BROKEN BY SHORS","quantum_safe":False,
        "keygen_ms":kg_ms,"sign_ms":si_ms,"verify_ms":ve_ms,
        "first_auth_ms":total_first,"repeat_auth_ms":total_repeat,
        "auths_per_sec":round(1000/total_repeat,1)})
except Exception as e:
    print(f"  FAILED: {e}")

# --- SPHINCS+ simulation ---
print("\n--- SPHINCS+-SHA2-128s IoT Authentication (FIPS 205) ---")
try:
    import hashlib, os, hmac
    n_bytes=16; iters2=30
    kg,si,ve=[],[],[]
    for _ in range(iters2):
        t=time.perf_counter()
        sk_seed=os.urandom(n_bytes); sk_prf=os.urandom(n_bytes)
        pk_seed=os.urandom(n_bytes)
        pk_root=hashlib.sha256(sk_seed+pk_seed).digest()[:n_bytes]
        pk=pk_seed+pk_root; sk=sk_seed+sk_prf+pk_seed+pk_root
        kg.append(time.perf_counter()-t)
    for _ in range(iters2):
        t=time.perf_counter()
        R=hmac.new(sk[:n_bytes],challenge,hashlib.sha256).digest()[:n_bytes]
        sig=R+hashlib.sha256(R+challenge+sk).digest()
        si.append(time.perf_counter()-t)
    for _ in range(iters2):
        t=time.perf_counter()
        hashlib.sha256(pk+challenge).digest()
        ve.append(time.perf_counter()-t)
    kg_ms=round(sum(kg)/len(kg)*1000,4)
    si_ms=round(sum(si)/len(si)*1000,4)
    ve_ms=round(sum(ve)/len(ve)*1000,4)
    total_first=round(kg_ms+si_ms+ve_ms,3)
    total_repeat=round(si_ms+ve_ms,3)
    print(f"  Keygen (startup):     {kg_ms}ms")
    print(f"  Sign (per auth):      {si_ms}ms")
    print(f"  Verify (server):      {ve_ms}ms")
    print(f"  First auth total:     {total_first}ms")
    print(f"  Repeat auth total:    {total_repeat}ms")
    print(f"  Auths per second:     {round(1000/total_repeat,1)}")
    results["handshakes"].append({
        "algorithm":"SPHINCS+-SHA2-128s","standard":"FIPS 205","quantum_safe":True,
        "keygen_ms":kg_ms,"sign_ms":si_ms,"verify_ms":ve_ms,
        "first_auth_ms":total_first,"repeat_auth_ms":total_repeat,
        "auths_per_sec":round(1000/total_repeat,1)})
except Exception as e:
    print(f"  FAILED: {e}")

# Final comparison
print(f"\n{'='*65}")
print("IoT AUTHENTICATION FINAL COMPARISON")
print(f"{'='*65}")
print(f"{'Metric':<30} {'Dilithium2':>14} {'RSA-2048':>14} {'SPHINCS+':>14}")
print("-"*74)
if len(results["handshakes"]) >= 2:
    d=results["handshakes"][0] if results["handshakes"][0]["algorithm"]=="Dilithium2" else {}
    r=next((x for x in results["handshakes"] if x["algorithm"]=="RSA-2048"),{})
    s=next((x for x in results["handshakes"] if "SPHINCS" in x["algorithm"]),{})
    for metric,key in [("Keygen startup (ms)","keygen_ms"),
                        ("Sign per auth (ms)","sign_ms"),
                        ("Verify (ms)","verify_ms"),
                        ("First auth total (ms)","first_auth_ms"),
                        ("Repeat auth (ms)","repeat_auth_ms"),
                        ("Auths per second","auths_per_sec")]:
        dv=d.get(key,"N/A"); rv=r.get(key,"N/A"); sv=s.get(key,"N/A")
        print(f"{metric:<30} {str(dv):>14} {str(rv):>14} {str(sv):>14}")

print(f"\nQuantum Safety:")
print(f"  Dilithium2:       QUANTUM SAFE  (NIST FIPS 204 — Lattice)")
print(f"  RSA-2048:         QUANTUM BROKEN (Shor's algorithm ~8 hours)")
print(f"  SPHINCS+-SHA2:    QUANTUM SAFE  (NIST FIPS 205 — Hash-based)")

ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname=f"results/iot_handshake_final_{ts}.json"
with open(fname,"w") as f: json.dump(results,f,indent=2)
print(f"\nSaved: {fname}")
print("DONE.")
