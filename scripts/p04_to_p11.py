import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from arq_utils import *
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import time, argparse

sim = AerSimulator()

# ── P4: NOISE INCLUSION ───────────────────────────────────────────────────────
def run_p4():
    print_header(4, "NOISE INCLUSION — QUANTUM STOCHASTIC RESONANCE")
    def grover_prob(noise, n=3):
        qc = QuantumCircuit(n,n); qc.h(range(n))
        qc.h(n-1); qc.ccx(0,1,2); qc.h(n-1)
        qc.h(range(n)); qc.x(range(n))
        qc.h(n-1); qc.ccx(0,1,2); qc.h(n-1)
        qc.x(range(n)); qc.h(range(n)); qc.measure_all()
        nm = NoiseModel()
        if noise > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(noise,1),['h','x'])
        b = AerSimulator(noise_model=nm) if noise > 0 else sim
        counts = b.run(transpile(qc,b), shots=2048).result().get_counts()
        return counts.get('111',0)/2048

    levels = [0.0,0.002,0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.08,0.10,0.15,0.20]
    print(f"\n{'Noise':>10} {'P(|111>)':>12} {'Bar'}")
    print("-"*55)
    probs = []
    for nl in levels:
        p = grover_prob(nl); probs.append(p)
        print(f"{nl:>10.3f} {p:>12.4f} {'█'*int(p*25)}")
    peak = levels[np.argmax(probs)]
    print(f"\nResonance peak at noise={peak:.3f}, prob={max(probs):.4f}")
    ac = adiabatic_condition(0.3, 10.0)
    print(f"Adiabatic condition satisfied: {ac['satisfied']} (margin {ac['margin']}x)")
    save_result(4,"Noise Inclusion",{"levels":levels,"probs":probs,"peak":peak,"adiabatic":ac})
    print("PRINCIPLE 4 DONE.")

# ── P5: SOFT MEASUREMENT ─────────────────────────────────────────────────────
def run_p5():
    print_header(5, "SOFT MEASUREMENT — WEAK MEASUREMENT VIA ANCILLA")
    def weak_meas(coupling, shots=4096):
        qc = QuantumCircuit(3,3)
        qc.h(0); qc.h(1); qc.cx(0,1)
        qc.ry(coupling,2); qc.cx(0,2); qc.cx(1,2)
        qc.measure(2,2); qc.measure(0,0); qc.measure(1,1)
        counts = sim.run(transpile(qc,sim), shots=shots).result().get_counts()
        total = sum(counts.values())
        main = {}
        for bs,cnt in counts.items():
            mb = bs[:2]
            main[mb] = main.get(mb,0)+cnt
        exp = total/4
        dev = sum(abs(cnt-exp) for cnt in main.values())
        coherence = max(0.0, 1.0-dev/(2*total))
        anc = {}
        for bs,cnt in counts.items():
            anc[bs[-1]] = anc.get(bs[-1],0)+cnt
        pnorm = [v/total for v in anc.values()]
        info = max(0, -sum(p*np.log2(p+1e-10) for p in pnorm)+1.0)
        return coherence, info

    couplings = np.linspace(0, np.pi/2, 13)
    print(f"\n{'Coupling':>10} {'Coherence':>12} {'Info Bits':>12} {'Type'}")
    print("-"*50)
    data = []
    for c in couplings:
        coh,info = weak_meas(c)
        t = "NO MEAS" if c<0.1 else ("VERY SOFT" if c<0.5 else ("SOFT" if c<1.0 else ("MODERATE" if c<1.3 else "HARD")))
        print(f"{c:>10.4f} {coh:>12.4f} {info:>12.4f} {t}")
        data.append({"coupling":round(c,4),"coherence":round(coh,4),"info":round(info,4)})

    print("\n--- AI Feedback Loop ---")
    cc = 0.1
    for step in range(10):
        coh,info = weak_meas(cc)
        if coh > 0.9: cc = min(cc+0.05, np.pi/2)
        elif coh < 0.8: cc = max(cc-0.03, 0.05)
        print(f"  Step {step+1:2d}: coupling={cc:.3f} coherence={coh:.4f} info={info:.4f} bits")

    save_result(5,"Soft Measurement",{"sweep":data})
    print("PRINCIPLE 5 DONE.")

# ── P6: AI-STABILISED GATES ──────────────────────────────────────────────────
def run_p6():
    print_header(6, "AI-STABILISED VARIATIONAL GATES WITH TEMPERATURE FEEDBACK")
    def build_circuit(params, temp):
        tf = max(0.75, 1.0-(temp-40)*0.012)
        qc = QuantumCircuit(3,3)
        for i in range(3): qc.ry(params[i]*tf,i); qc.rz(params[3+i]*tf,i)
        qc.cx(0,1); qc.cx(1,2)
        for i in range(3): qc.ry(params[6+i]*tf,i)
        qc.measure_all()
        return qc

    def cost(params):
        t = get_temp(); r = compute_thermal_noise_rate(t)
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(r,1),['ry','rz'])
        nm.add_all_qubit_quantum_error(depolarizing_error(r*6,2),['cx'])
        s = AerSimulator(noise_model=nm)
        qc = build_circuit(params, t)
        counts = s.run(transpile(qc,s), shots=512).result().get_counts()
        target = (counts.get('000',0)+counts.get('111',0))/512
        return 1-target

    x0 = np.random.uniform(0,np.pi,9)
    print(f"\n  Temp: {get_temp()}°C | Optimising 9 gate parameters...")
    t0 = time.perf_counter()
    res = minimize(cost, x0, method='COBYLA', options={'maxiter':80,'rhobeg':0.4})
    print(f"  Done: {res.nfev} evals, {time.perf_counter()-t0:.1f}s")
    print(f"  Target state probability: {1-res.fun:.4f}")

    print(f"\n  Temperature sensitivity:")
    print(f"{'Temp':>6} {'T-factor':>10} {'Target prob':>14}")
    print("-"*35)
    for t in [30,40,45,50,55,65,75]:
        tf = max(0.75,1.0-(t-40)*0.012)
        r = compute_thermal_noise_rate(t)
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(r,1),['ry','rz'])
        nm.add_all_qubit_quantum_error(depolarizing_error(r*6,2),['cx'])
        s = AerSimulator(noise_model=nm)
        qc = build_circuit(res.x, t)
        counts = s.run(transpile(qc,s), shots=512).result().get_counts()
        prob = (counts.get('000',0)+counts.get('111',0))/512
        print(f"{t:>5}°C {tf:>10.4f} {prob:>14.4f}")

    save_result(6,"AI Stabilised Gates",{"opt_params":res.x.tolist(),"evals":res.nfev,"prob":float(1-res.fun)})
    print("PRINCIPLE 6 DONE.")

# ── P7: ADIABATIC MORPHING ────────────────────────────────────────────────────
def run_p7():
    print_header(7, "ADIABATIC HAMILTONIAN MORPHING — CONTINUOUS EVOLUTION")
    N=4; STEPS=15

    def adiabatic_step(s):
        qc = QuantumCircuit(N,N); qc.h(range(N))
        if s<0.99:
            for q in range(N): qc.rx((1-s)*np.pi,q)
        if s>0.01:
            for q1,q2 in [(0,1),(1,2),(2,3),(0,2),(1,3)]:
                qc.cx(q1,q2); qc.rz(s*np.pi/3,q2); qc.cx(q1,q2)
        qc.measure_all()
        return qc

    def gs_prob(counts, shots):
        def energy(bs):
            spins=[1 if c=='1' else -1 for c in bs]
            return sum(spins[i]*spins[j] for i,j in [(0,1),(1,2),(2,3),(0,2),(1,3)])
        energies={bs:energy(bs) for bs in counts}
        gse=min(energies.values())
        gs=sum(cnt for bs,cnt in counts.items() if energies[bs]==gse)
        return gs/shots, gse

    ac = adiabatic_condition(0.5, STEPS)
    print(f"\n  Adiabatic condition: satisfied={ac['satisfied']} margin={ac['margin']}x")
    print(f"\n{'Step':>5} {'s':>5} {'GS Prob':>9} {'Energy':>10} {'Adiabatic':>12} {'Bar'}")
    print("-"*60)
    results=[]
    prev=None
    for step in range(STEPS+1):
        s=step/STEPS
        qc=adiabatic_step(s)
        counts=sim.run(transpile(qc,sim), shots=2048).result().get_counts()
        gsp,gse=gs_prob(counts,2048)
        adia=True if prev is None else (gsp>=prev*0.85)
        prev=gsp
        bar='█'*int(gsp*18)
        print(f"{step:>5} {s:>5.2f} {gsp:>9.4f} {gse:>10.1f} {'✓' if adia else '✗ jump':>12} {bar}")
        results.append({"step":step,"s":s,"gs_prob":round(gsp,4),"gs_energy":gse})

    save_result(7,"Adiabatic Morphing",{"steps":results,"adiabatic":ac})
    print("PRINCIPLE 7 DONE.")

# ── P8: DECOHERENCE PREDICTOR ─────────────────────────────────────────────────
def run_p8():
    print_header(8, "AI DECOHERENCE PREDICTOR — ML ON REAL NOISE DATA")
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor

    X = np.array([
        [4,2,1,133,0],[6,3,1,133,1],[12,2,1,133,2],
        [4,2,2,156,0],[6,3,2,156,1],[12,2,2,156,2],
        [4,2,2,156,0],[6,3,2,156,1],[107,4,2,156,3],
        [596,4,2,156,2],[9,5,2,156,5],[12,2,2,156,6],
    ])
    y = np.array([11.62,5.08,16.31,3.03,5.86,3.80,
                  2.34,4.20,6.12,52.0,9.70,6.84])

    poly = PolynomialFeatures(degree=2); Xp = poly.fit_transform(X)
    ridge = Ridge(alpha=2.0); ridge.fit(Xp,y)
    gb = GradientBoostingRegressor(n_estimators=80,max_depth=3,random_state=42); gb.fit(X,y)

    print(f"\n  Ridge R²: {ridge.score(Xp,y):.4f}")
    print(f"  GBM R²:   {gb.score(X,y):.4f}")

    circuits = [
        ([9,4,2,156,5],"BV 4Q ibm_kingston"),
        ([160,6,2,156,3],"QAOA 6-node ibm_fez"),
        ([20,5,2,156,1],"GHZ 5Q ibm_fez"),
        ([596,4,2,156,2],"Grover 4Q ibm_fez (actual 52%)"),
        ([200,5,3,156,3],"Future Heron r3 QAOA"),
        ([1000,6,2,156,2],"Deep circuit stress test"),
    ]
    print(f"\n{'Circuit':<45} {'Ridge':>8} {'GBM':>8} {'Risk'}")
    print("-"*72)
    preds=[]
    for feat,name in circuits:
        rp=max(0,ridge.predict(poly.transform([feat]))[0])
        gp=max(0,gb.predict([feat])[0])
        risk="EXTREME" if gp>40 else ("HIGH" if gp>15 else ("MED" if gp>7 else "LOW"))
        print(f"{name:<45} {rp:>7.2f}% {gp:>7.2f}% {risk}")
        preds.append({"name":name,"ridge":round(rp,2),"gbm":round(gp,2),"risk":risk})

    print(f"\n  Depth lookup (4Q Heron r2):")
    print(f"{'Depth':>8} {'Predicted Noise':>18} {'Usable?'}")
    print("-"*35)
    for d in [4,9,20,50,100,200,400,596,800,1000]:
        pred=max(0,gb.predict([[d,4,2,156,3]])[0])
        usable="YES" if pred<20 else ("MARGINAL" if pred<40 else "NO")
        print(f"{d:>8} {pred:>17.2f}% {usable}")

    save_result(8,"Decoherence Predictor",{"ridge_r2":round(ridge.score(Xp,y),4),"gbm_r2":round(gb.score(X,y),4),"predictions":preds})
    print("PRINCIPLE 8 DONE.")

# ── P9: QUANTUM PARALLELISM ───────────────────────────────────────────────────
def run_p9():
    print_header(9, "QUANTUM PARALLELISM + AI ALGORITHM SELECTOR")
    def grover(n_bits, target_idx):
        target = format(target_idx, f'0{n_bits}b')
        iters = max(1,int(np.pi/4*np.sqrt(2**n_bits)))
        anc = n_bits
        qc = QuantumCircuit(n_bits+1,n_bits)
        qc.x(anc); qc.h(range(n_bits+1))
        def oracle():
            for i,b in enumerate(reversed(target)):
                if b=='0': qc.x(i)
            qc.h(anc); qc.mcx(list(range(n_bits)),anc); qc.h(anc)
            for i,b in enumerate(reversed(target)):
                if b=='0': qc.x(i)
        def diffuser():
            qc.h(range(n_bits)); qc.x(range(n_bits))
            qc.h(n_bits-1); qc.mcx(list(range(n_bits-1)),n_bits-1); qc.h(n_bits-1)
            qc.x(range(n_bits)); qc.h(range(n_bits))
        for _ in range(iters): oracle(); diffuser()
        qc.measure(range(n_bits),range(n_bits))
        t0=time.perf_counter()
        counts=sim.run(transpile(qc,sim),shots=1024).result().get_counts()
        t=time.perf_counter()-t0
        top=max(counts,key=counts.get)
        return iters,t,top==target,counts.get(target,0)/1024

    def ai_selector(n_bits, qubit_budget=20):
        space=2**n_bits; q_iters=max(1,int(np.pi/4*np.sqrt(space)))
        c_avg=space//2; depth_est=q_iters*(n_bits+2)*3
        noise_est=min(0.9,0.02*depth_est/100)
        eff=c_avg/q_iters*(1-noise_est)
        if n_bits>qubit_budget: return "CLASSICAL","exceeds qubit budget"
        if noise_est>0.15: return "CLASSICAL",f"noise {noise_est:.0%} too high"
        if eff<2: return "CLASSICAL","speedup <2x after noise"
        return "QUANTUM",f"effective speedup {eff:.1f}x"

    print(f"\n{'Bits':>5} {'Space':>8} {'Classical':>12} {'Grover Q':>10} {'Speedup':>8} {'P(target)':>10} {'AI'}")
    print("-"*70)
    results=[]
    for n in [2,3,4,5,6]:
        iters,t,success,prob = grover(n, 2**(n-1)+1)
        c_avg=2**(n-1); speedup=c_avg/iters
        decision,reason = ai_selector(n)
        print(f"{n:>5} {2**n:>8,} {c_avg:>12,} {iters:>10} {speedup:>8.1f}x {prob:>9.1%} {decision}")
        results.append({"n":n,"speedup":round(speedup,1),"prob":round(prob,4),"ai":decision})

    print(f"\n  AI Decision Boundary (larger problems):")
    print(f"{'Bits':>5} {'Eff Speedup':>12} {'AI Decision':>14} {'Reason'}")
    print("-"*60)
    for n in [2,5,8,12,16,20,25,30]:
        d,r = ai_selector(n)
        q_iters=max(1,int(np.pi/4*np.sqrt(2**n)))
        c_avg=2**(n-1)
        print(f"{n:>5} {c_avg/q_iters:>12.1f} {d:>14} {r}")

    save_result(9,"Quantum Parallelism",{"scaling":results})
    print("PRINCIPLE 9 DONE.")

# ── P10: NO-CLONING QKD ───────────────────────────────────────────────────────
def run_p10():
    print_header(10, "NO-CLONING PROTECTION — BB84 QKD")
    def bb84(n_bits=80, eve=False, eve_rate=1.0):
        np.random.seed(int(time.time())%10000)
        ab=np.random.randint(0,2,n_bits); abase=np.random.randint(0,2,n_bits)
        bbase=np.random.randint(0,2,n_bits)
        if eve: ebase=np.random.randint(0,2,n_bits)
        results=[]
        for i in range(n_bits):
            qc=QuantumCircuit(1,1)
            if ab[i]==1: qc.x(0)
            if abase[i]==1: qc.h(0)
            if eve and np.random.random()<eve_rate:
                qce=qc.copy()
                if ebase[i]==1: qce.h(0)
                qce.measure(0,0)
                ce=sim.run(transpile(qce,sim),shots=1).result().get_counts()
                eb=int(list(ce.keys())[0])
                qc=QuantumCircuit(1,1)
                if eb==1: qc.x(0)
                if eve and ebase[i]==1: qc.h(0)
            if bbase[i]==1: qc.h(0)
            qc.measure(0,0)
            cb=sim.run(transpile(qc,sim),shots=1).result().get_counts()
            results.append(int(list(cb.keys())[0]))
        sifted=[(ab[i],results[i]) for i in range(n_bits) if abase[i]==bbase[i]]
        if not sifted: return 0,0
        errors=sum(a!=b for a,b in sifted)
        return errors/len(sifted), len(sifted)

    scenarios=[
        ("Clean (no Eve)",False,0.0),
        ("Eve 30%",True,0.3),
        ("Eve 60%",True,0.6),
        ("Eve 100%",True,1.0),
    ]
    print(f"\n{'Scenario':<25} {'Avg QBER':>10} {'Detected?':>12} {'Key bits':>10}")
    print("-"*60)
    all_res={}
    for name,eve,rate in scenarios:
        qbers=[bb84(80,eve,rate)[0] for _ in range(10)]
        avg=np.mean(qbers)
        detected=avg>0.11
        print(f"{name:<25} {avg:>10.4f} {'YES — ABORT' if detected else 'NO — SECURE':>12} ~40")
        all_res[name]={"avg_qber":round(avg,4),"detected":bool(detected)}

    print("\n--- No-Cloning Theorem Demo ---")
    for desc,prep in [
        ("|0> — basis (CNOT clones OK)", lambda qc: None),
        ("|1> — basis (CNOT clones OK)", lambda qc: qc.x(0)),
        ("|+> — superposition (FAILS)",  lambda qc: qc.h(0)),
    ]:
        qc=QuantumCircuit(2,2); prep(qc); qc.cx(0,1); qc.measure_all()
        c=sim.run(transpile(qc,sim),shots=1024).result().get_counts()
        perfect=(c.get('00',0)+c.get('11',0))/1024
        print(f"  {desc}")
        print(f"    Clone success: {perfect:.3f}  {'OK' if perfect>0.95 else '<- CLONING FAILED'}")

    save_result(10,"No-Cloning QKD",{"scenarios":all_res})
    print("PRINCIPLE 10 DONE.")

# ── P11: ALGORITHM FUSION ─────────────────────────────────────────────────────
def run_p11():
    print_header(11, "ALGORITHM FUSION — QAOA + GROVER HYBRID ON REAL IBM HARDWARE")
    N=4
    edges=[(0,1),(1,2),(2,3),(3,0),(0,2)]
    def cut_val(bs): return sum(1 for i,j in edges if bs[i]!=bs[j])
    optimal=[bs for bs in [format(i,f'0{N}b') for i in range(2**N)] if cut_val(bs)==4]

    def fusion_circuit(gamma,beta,target='1010'):
        qc=QuantumCircuit(N,N); qc.h(range(N))
        for q1,q2 in edges:
            qc.cx(q1,q2); qc.rz(2*gamma,q2); qc.cx(q1,q2)
        for q in range(N): qc.rx(2*beta,q)
        for i,b in enumerate(reversed(target)):
            if b=='0': qc.x(i)
        qc.h(N-1); qc.mcx(list(range(N-1)),N-1); qc.h(N-1)
        for i,b in enumerate(reversed(target)):
            if b=='0': qc.x(i)
        qc.h(range(N)); qc.x(range(N))
        qc.h(N-1); qc.mcx(list(range(N-1)),N-1); qc.h(N-1)
        qc.x(range(N)); qc.h(range(N))
        qc.measure_all(); return qc

    def pure_qaoa(gamma,beta):
        qc=QuantumCircuit(N,N); qc.h(range(N))
        for q1,q2 in edges:
            qc.cx(q1,q2); qc.rz(2*gamma,q2); qc.cx(q1,q2)
        for q in range(N): qc.rx(2*beta,q)
        qc.measure_all(); return qc

    def cost(params):
        gamma,beta=params; qc=fusion_circuit(gamma,beta)
        counts=sim.run(transpile(qc,sim),shots=512).result().get_counts()
        return -(sum(cnt*cut_val(bs) for bs,cnt in counts.items())/512)

    print("\n  Optimising QAOA parameters...")
    res=minimize(cost,[np.pi/4,np.pi/4],method='COBYLA',options={'maxiter':120,'rhobeg':0.5})
    g,b=res.x
    print(f"  γ={g:.4f} β={b:.4f} | {res.nfev} evals | cut={-res.fun:.4f}")

    counts_f=sim.run(transpile(fusion_circuit(g,b),sim),shots=2048).result().get_counts()
    counts_p=sim.run(transpile(pure_qaoa(g,b),sim),shots=2048).result().get_counts()
    opt_f=sum(counts_f.get(bs,0) for bs in optimal)/2048
    opt_p=sum(counts_p.get(bs,0) for bs in optimal)/2048
    print(f"\n  Simulator — optimal state prob:")
    print(f"    Pure QAOA:    {opt_p:.4f}")
    print(f"    QAOA+Grover:  {opt_f:.4f}")
    print(f"    Fusion gain:  {opt_f-opt_p:+.4f}")

    print("\n  Real IBM Hardware...")
    try:
        svc=QiskitRuntimeService(channel="ibm_quantum_platform")
        backend=svc.least_busy(operational=True,simulator=False)
        print(f"  Backend: {backend.name}")
        sampler=SamplerV2(backend)
        qcp_t=transpile(pure_qaoa(g,b),backend,optimization_level=3)
        qcf_t=transpile(fusion_circuit(g,b),backend,optimization_level=3)
        print(f"  Pure QAOA depth: {qcp_t.depth()} | Fusion depth: {qcf_t.depth()}")
        jp=sampler.run([qcp_t],shots=1024)
        print(f"  Pure QAOA job: {jp.job_id()} — waiting...")
        cp_hw=dict(jp.result()[0].data.c.get_counts())
        jf=sampler.run([qcf_t],shots=1024)
        print(f"  Fusion job:    {jf.job_id()} — waiting...")
        cf_hw=dict(jf.result()[0].data.c.get_counts())
        op_hw=sum(cp_hw.get(bs,0) for bs in optimal)/1024
        of_hw=sum(cf_hw.get(bs,0) for bs in optimal)/1024
        print(f"\n  Real HW optimal prob — QAOA: {op_hw:.4f}  Fusion: {of_hw:.4f}")
        hw={"backend":backend.name,"job_pure":jp.job_id(),"job_fusion":jf.job_id(),
            "opt_pure_hw":round(op_hw,4),"opt_fusion_hw":round(of_hw,4)}
    except Exception as e:
        print(f"  IBM error: {e}"); hw={"error":str(e)}

    save_result(11,"Algorithm Fusion",{"gamma":round(g,4),"beta":round(b,4),
        "opt_sim_pure":round(opt_p,4),"opt_sim_fusion":round(opt_f,4),"hw":hw})
    print("PRINCIPLE 11 DONE.")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--principle", type=int, default=0)
    args = parser.parse_args()
    runners = {4:run_p4,5:run_p5,6:run_p6,7:run_p7,
               8:run_p8,9:run_p9,10:run_p10,11:run_p11}
    if args.principle == 0:
        for fn in runners.values(): fn()
    elif args.principle in runners:
        runners[args.principle]()
    else:
        print(f"Use --principle 4 through 11")
