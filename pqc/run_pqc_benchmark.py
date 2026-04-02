
import time, json, datetime, platform

def get_device_info():
    info = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        info["cpu_temp_celsius"] = float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000
    except:
        info["cpu_temp_celsius"] = "unavailable"
    return info

def benchmark_kyber(variant, iters=50):
    if variant == "Kyber512":
        from kyber_py.kyber import Kyber512 as K
    elif variant == "Kyber768":
        from kyber_py.kyber import Kyber768 as K
    else:
        from kyber_py.kyber import Kyber1024 as K
    print(f"  {variant}...")
    kg,en,de=[],[],[]
    for _ in range(iters):
        t=time.perf_counter(); pk,sk=K.keygen(); kg.append(time.perf_counter()-t)
    cts=[]
    for _ in range(iters):
        t=time.perf_counter(); c,key=K.encaps(pk); en.append(time.perf_counter()-t); cts.append(c)
    for c in cts:
        t=time.perf_counter(); K.decaps(sk,c); de.append(time.perf_counter()-t)
    return {"algorithm":variant,"type":"PQC-KEM-NIST","iterations":iters,
            "keygen_ms_avg":round(sum(kg)/len(kg)*1000,4),
            "encap_ms_avg":round(sum(en)/len(en)*1000,4),
            "decap_ms_avg":round(sum(de)/len(de)*1000,4)}

def benchmark_dilithium(variant, iters=50):
    if variant == "Dilithium2":
        from dilithium_py.dilithium import Dilithium2 as D
    elif variant == "Dilithium3":
        from dilithium_py.dilithium import Dilithium3 as D
    else:
        from dilithium_py.dilithium import Dilithium5 as D
    print(f"  {variant}...")
    msg = b"QSimEdge PQC benchmark"
    kg,si,ve=[],[],[]
    for _ in range(iters):
        t=time.perf_counter(); pk,sk=D.keygen(); kg.append(time.perf_counter()-t)
    sigs=[]
    for _ in range(iters):
        t=time.perf_counter(); sig=D.sign(sk,msg); si.append(time.perf_counter()-t); sigs.append(sig)
    for sig in sigs:
        t=time.perf_counter(); D.verify(pk,msg,sig); ve.append(time.perf_counter()-t)
    return {"algorithm":variant,"type":"PQC-Signature-NIST","iterations":iters,
            "keygen_ms_avg":round(sum(kg)/len(kg)*1000,4),
            "sign_ms_avg":round(sum(si)/len(si)*1000,4),
            "verify_ms_avg":round(sum(ve)/len(ve)*1000,4)}

def benchmark_rsa(bits, iters=20):
    from Crypto.PublicKey import RSA
    from Crypto.Cipher import PKCS1_OAEP
    from Crypto.Signature import pkcs1_15
    from Crypto.Hash import SHA256
    print(f"  RSA-{bits}...")
    msg=b"QSimEdge PQC benchmark"
    kg=[]
    for _ in range(iters):
        t=time.perf_counter(); key=RSA.generate(bits); kg.append(time.perf_counter()-t)
    key=RSA.generate(bits)
    en,de,si,ve=[],[],[],[]
    cp=PKCS1_OAEP.new(key.publickey()); cv=PKCS1_OAEP.new(key)
    cts=[]
    for _ in range(iters):
        t=time.perf_counter(); ct=cp.encrypt(msg); en.append(time.perf_counter()-t); cts.append(ct)
    for ct in cts:
        t=time.perf_counter(); cv.decrypt(ct); de.append(time.perf_counter()-t)
    h=SHA256.new(msg); sig=pkcs1_15.new(key).sign(h)
    for _ in range(iters):
        t=time.perf_counter(); pkcs1_15.new(key).sign(SHA256.new(msg)); si.append(time.perf_counter()-t)
    for _ in range(iters):
        t=time.perf_counter()
        try: pkcs1_15.new(key.publickey()).verify(SHA256.new(msg),sig)
        except: pass
        ve.append(time.perf_counter()-t)
    return {"algorithm":f"RSA-{bits}","type":"Classical-RSA","iterations":iters,
            "keygen_ms_avg":round(sum(kg)/len(kg)*1000,4),
            "encrypt_ms_avg":round(sum(en)/len(en)*1000,4),
            "decrypt_ms_avg":round(sum(de)/len(de)*1000,4),
            "sign_ms_avg":round(sum(si)/len(si)*1000,4),
            "verify_ms_avg":round(sum(ve)/len(ve)*1000,4),
            "public_key_bytes":len(key.publickey().export_key()),
            "ciphertext_bytes":len(cts[0])}

print("="*60)
print("QSimEdge PQC Benchmark — Raspberry Pi 5 8GB")
print("="*60)
device=get_device_info()
print(f"Temp: {device['cpu_temp_celsius']}C\n")

results={"device":"Raspberry Pi 5 8GB",
         "timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "device_info":device,
         "library":"kyber-py + dilithium-py (pure Python NIST PQC)",
         "research_question":"Are NIST PQC algorithms feasible on ARM IoT edge hardware?",
         "kem_results":[],"sig_results":[],"rsa_results":[]}

print("--- Kyber KEM ---")
for v in ["Kyber512","Kyber768","Kyber1024"]:
    try:
        r=benchmark_kyber(v); results["kem_results"].append(r)
        print(f"    {v}: keygen={r['keygen_ms_avg']}ms | encap={r['encap_ms_avg']}ms | decap={r['decap_ms_avg']}ms")
    except Exception as e: print(f"    {v} failed: {e}")

print("\n--- Dilithium Signatures ---")
for v in ["Dilithium2","Dilithium3","Dilithium5"]:
    try:
        r=benchmark_dilithium(v); results["sig_results"].append(r)
        print(f"    {v}: keygen={r['keygen_ms_avg']}ms | sign={r['sign_ms_avg']}ms | verify={r['verify_ms_avg']}ms")
    except Exception as e: print(f"    {v} failed: {e}")

print("\n--- RSA Baseline ---")
for bits in [2048,3072,4096]:
    r=benchmark_rsa(bits); results["rsa_results"].append(r)
    print(f"    RSA-{bits}: keygen={r['keygen_ms_avg']}ms | enc={r['encrypt_ms_avg']}ms | dec={r['decrypt_ms_avg']}ms | sign={r['sign_ms_avg']}ms")

ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fname=f"../results/pqc_benchmark_pi5_{ts}.json"
with open(fname,"w") as f: json.dump(results,f,indent=2)

print(f"\n{'='*60}")
print("FINAL SUMMARY TABLE")
print(f"{'='*60}")
print(f"{'Algorithm':<18} {'Keygen(ms)':<16} {'Enc/Sign(ms)':<16} {'Dec/Verify(ms)'}")
print("-"*68)
for r in results["kem_results"]:
    print(f"{r['algorithm']:<18} {r['keygen_ms_avg']:<16} {r['encap_ms_avg']:<16} {r['decap_ms_avg']}")
for r in results["sig_results"]:
    print(f"{r['algorithm']:<18} {r['keygen_ms_avg']:<16} {r['sign_ms_avg']:<16} {r['verify_ms_avg']}")
for r in results["rsa_results"]:
    print(f"{r['algorithm']:<18} {r['keygen_ms_avg']:<16} {r['encrypt_ms_avg']:<16} {r['decrypt_ms_avg']}")

print(f"\nSaved: {fname}")
print("DONE.")


