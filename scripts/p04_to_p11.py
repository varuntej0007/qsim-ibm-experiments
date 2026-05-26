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


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: safe counts reader — handles measure_all bit reversal
# ─────────────────────────────────────────────────────────────────────────────
def get_count(counts, bitstring):
    """Get count for a bitstring, also checking reversed version."""
    direct = counts.get(bitstring, 0)
    rev    = counts.get(bitstring[::-1], 0)
    return direct + rev

def dominant_state(counts):
    """Return the most frequent bitstring."""
    return max(counts, key=counts.get)


# ═════════════════════════════════════════════════════════════════════════════
# P4: NOISE INCLUSION — STOCHASTIC RESONANCE
# ═════════════════════════════════════════════════════════════════════════════
def run_p4():
    print_header(4, "NOISE INCLUSION — QUANTUM STOCHASTIC RESONANCE")

    def grover_3q_target_prob(noise_level):
        """
        3-qubit Grover targeting |101⟩.
        Uses explicit measure (not measure_all) to avoid bit reversal.
        """
        target = '101'
        n = 3
        qc = QuantumCircuit(n, n)
        qc.h(range(n))

        # Oracle for |101⟩ — flip bits where target has '0'
        qc.x(1)           # flip qubit 1 (target bit 1 is '0')
        qc.h(2)
        qc.ccx(0, 1, 2)   # CCX acts as CZ when middle qubit in |−⟩
        qc.h(2)
        qc.x(1)

        # Diffusion operator
        qc.h(range(n))
        qc.x(range(n))
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
        qc.x(range(n))
        qc.h(range(n))

        qc.measure(range(n), range(n))   # explicit — no reversal

        nm = NoiseModel()
        if noise_level > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(noise_level, 1), ['h', 'x'])
            nm.add_all_qubit_quantum_error(depolarizing_error(noise_level * 8, 3), ['ccx'])
        backend = AerSimulator(noise_model=nm) if noise_level > 0 else sim
        counts = backend.run(transpile(qc, backend), shots=2048).result().get_counts()
        return counts.get(target, 0) / 2048

    levels = [0.0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02,
              0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

    print(f"\n{'Noise':>10} {'P(|101>)':>12} {'Bar':>30} {'Note'}")
    print("-"*70)

    probs = []
    for nl in levels:
        p = grover_3q_target_prob(nl)
        probs.append(p)
        bar = "█" * int(p * 25)
        note = "← IDEAL" if nl == 0.0 else ""
        print(f"{nl:>10.3f} {p:>12.4f} {bar:<30} {note}")

    peak_idx = np.argmax(probs)
    peak_noise = levels[peak_idx]
    peak_prob  = probs[peak_idx]
    print(f"\n  Ideal (zero noise) probability:  {probs[0]:.4f}")
    print(f"  Resonance peak at noise={peak_noise:.3f}: {peak_prob:.4f}")
    if peak_prob > probs[0] + 0.01:
        print(f"  STOCHASTIC RESONANCE CONFIRMED: small noise improves performance")
    else:
        print(f"  Peak = ideal (circuit too shallow for resonance; increase iterations)")

    ac = adiabatic_condition(0.3, 10.0)
    print(f"\n  Adiabatic condition check (ΔE=0.3, τ=10):")
    print(f"    Satisfied: {ac['satisfied']} | Margin: {ac['margin']}×")

    save_result(4, "Noise Inclusion", {
        "noise_levels": levels, "probabilities": probs,
        "peak_noise": peak_noise, "peak_prob": peak_prob,
        "adiabatic": ac
    })
    print("PRINCIPLE 4 DONE.")


# ═════════════════════════════════════════════════════════════════════════════
# P5: SOFT MEASUREMENT — WEAK MEASUREMENT
# ═════════════════════════════════════════════════════════════════════════════
def run_p5():
    print_header(5, "SOFT MEASUREMENT — WEAK MEASUREMENT VIA ANCILLA")

    def weak_meas(coupling, shots=4096):
        """
        2 main qubits + 1 ancilla.
        Ancilla couples weakly to main system via Ry(coupling) + CNOT chain.
        Measure all three explicitly.
        """
        qc = QuantumCircuit(3, 3)
        # Prepare main system in Bell-like superposition
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)

        # Weak coupling to ancilla
        qc.ry(coupling, 2)
        qc.cx(0, 2)

        # Measure all three explicitly (no reversal)
        qc.measure([0, 1, 2], [0, 1, 2])

        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        total  = sum(counts.values())

        # Main system marginal (bits 0 and 1)
        main = {}
        for bs, cnt in counts.items():
            mb = bs[:2]
            main[mb] = main.get(mb, 0) + cnt

        exp = total / 4
        dev = sum(abs(cnt - exp) for cnt in main.values())
        coherence = max(0.0, 1.0 - dev / (2 * total))

        # Information extracted from ancilla (bit 2)
        anc = {}
        for bs, cnt in counts.items():
            ab = bs[2]
            anc[ab] = anc.get(ab, 0) + cnt
        pnorm = [v / total for v in anc.values()]
        info = max(0.0, -sum(p * np.log2(p + 1e-12) for p in pnorm) + 1.0)

        return coherence, info

    couplings = np.linspace(0, np.pi / 2, 13)
    print(f"\n{'Coupling':>10} {'Coherence':>12} {'Info Bits':>12} {'Type'}")
    print("-"*52)
    data = []
    for c in couplings:
        coh, info = weak_meas(c)
        if   c < 0.1:  t = "NO MEAS"
        elif c < 0.5:  t = "VERY SOFT"
        elif c < 1.0:  t = "SOFT"
        elif c < 1.3:  t = "MODERATE"
        else:          t = "HARD"
        print(f"{c:>10.4f} {coh:>12.4f} {info:>12.4f} {t}")
        data.append({"coupling": round(c, 4), "coherence": round(coh, 4),
                     "info": round(info, 4)})

    print("\n--- AI Feedback Loop (target coherence 0.85) ---")
    cc = 0.1
    for step in range(10):
        coh, info = weak_meas(cc)
        if   coh > 0.90: cc = min(cc + 0.05, np.pi / 2)
        elif coh < 0.80: cc = max(cc - 0.03, 0.05)
        print(f"  Step {step+1:2d}: coupling={cc:.3f} coherence={coh:.4f} info={info:.4f} bits")

    save_result(5, "Soft Measurement", {"sweep": data})
    print("PRINCIPLE 5 DONE.")


# ═════════════════════════════════════════════════════════════════════════════
# P6: AI-STABILISED VARIATIONAL GATES
# ═════════════════════════════════════════════════════════════════════════════
def run_p6():
    print_header(6, "AI-STABILISED VARIATIONAL GATES WITH TEMPERATURE FEEDBACK")

    def build_circuit(params, temp):
        """
        3-qubit parametric circuit with temperature-compensated gates.
        Target: GHZ-like state |000⟩ + |111⟩.
        Uses explicit measurement to avoid bit reversal.
        """
        tf = max(0.75, 1.0 - (temp - 40) * 0.012)
        qc = QuantumCircuit(3, 3)
        # Layer 1: single-qubit rotations
        for i in range(3):
            qc.ry(params[i]     * tf, i)
            qc.rz(params[3 + i] * tf, i)
        # Entangling layer
        qc.cx(0, 1)
        qc.cx(1, 2)
        # Layer 2: final rotations
        for i in range(3):
            qc.ry(params[6 + i] * tf, i)
        qc.measure(range(3), range(3))   # explicit
        return qc

    def cost(params):
        t = get_temp()
        r = compute_thermal_noise_rate(t)
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(r,     1), ['ry', 'rz'])
        nm.add_all_qubit_quantum_error(depolarizing_error(r * 6, 2), ['cx'])
        s  = AerSimulator(noise_model=nm)
        qc = build_circuit(params, t)
        counts = s.run(transpile(qc, s), shots=512).result().get_counts()
        # GHZ target: |000⟩ and |111⟩
        target = counts.get('000', 0) + counts.get('111', 0)
        return 1.0 - target / 512

    np.random.seed(42)
    x0 = np.random.uniform(0, np.pi, 9)
    print(f"\n  Temp: {get_temp()}°C | Optimising 9 gate parameters...")
    t0 = time.perf_counter()
    res = minimize(cost, x0, method='COBYLA', options={'maxiter': 120, 'rhobeg': 0.5})
    elapsed = time.perf_counter() - t0
    print(f"  Done: {res.nfev} evals, {elapsed:.1f}s")
    print(f"  Target state probability: {1 - res.fun:.4f}")

    print(f"\n  Temperature sensitivity (AI-optimised parameters):")
    print(f"{'Temp':>6} {'T-factor':>10} {'Target prob':>14} {'Noise rate':>12}")
    print("-"*46)
    for t in [30, 40, 45, 50, 55, 65, 75]:
        tf   = max(0.75, 1.0 - (t - 40) * 0.012)
        rate = compute_thermal_noise_rate(t)
        nm   = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(rate,     1), ['ry', 'rz'])
        nm.add_all_qubit_quantum_error(depolarizing_error(rate * 6, 2), ['cx'])
        s    = AerSimulator(noise_model=nm)
        qc   = build_circuit(res.x, t)
        counts = s.run(transpile(qc, s), shots=512).result().get_counts()
        prob = (counts.get('000', 0) + counts.get('111', 0)) / 512
        print(f"{t:>5}°C {tf:>10.4f} {prob:>14.4f} {rate:>12.6f}")

    save_result(6, "AI Stabilised Gates", {
        "opt_params": res.x.tolist(),
        "evals": res.nfev,
        "final_prob": float(1 - res.fun),
        "time_s": round(elapsed, 2)
    })
    print("PRINCIPLE 6 DONE.")


# ═════════════════════════════════════════════════════════════════════════════
# P7: ADIABATIC HAMILTONIAN MORPHING
# ═════════════════════════════════════════════════════════════════════════════
def run_p7():
    print_header(7, "ADIABATIC HAMILTONIAN MORPHING — CONTINUOUS EVOLUTION")

    N = 4
    STEPS = 15
    EDGES = [(0,1),(1,2),(2,3),(0,2),(1,3)]

    def adiabatic_step(s):
        qc = QuantumCircuit(N, N)
        qc.h(range(N))
        if s < 0.99:
            for q in range(N): qc.rx((1 - s) * np.pi, q)
        if s > 0.01:
            for q1, q2 in EDGES:
                qc.cx(q1, q2)
                qc.rz(s * np.pi / 3, q2)
                qc.cx(q1, q2)
        qc.measure(range(N), range(N))   # explicit
        return qc

    def ising_energy(bs):
        spins = [1 if c == '1' else -1 for c in bs]
        return sum(spins[i] * spins[j] for i, j in EDGES)

    def gs_prob(counts, shots):
        energies  = {bs: ising_energy(bs) for bs in counts}
        gse       = min(energies.values())
        gs_count  = sum(cnt for bs, cnt in counts.items() if energies[bs] == gse)
        return gs_count / shots, gse

    ac = adiabatic_condition(0.5, STEPS)
    print(f"\n  Adiabatic condition: satisfied={ac['satisfied']} | margin={ac['margin']}×")
    print(f"\n{'Step':>5} {'s':>5} {'GS Prob':>9} {'Energy':>10} {'Adiabatic':>12} {'Bar'}")
    print("-"*62)

    results = []
    prev = None
    for step in range(STEPS + 1):
        s = step / STEPS
        qc = adiabatic_step(s)
        counts = sim.run(transpile(qc, sim), shots=2048).result().get_counts()
        gsp, gse = gs_prob(counts, 2048)
        adia = True if prev is None else (gsp >= prev * 0.85)
        prev = gsp
        bar  = "█" * int(gsp * 18)
        print(f"{step:>5} {s:>5.2f} {gsp:>9.4f} {gse:>10.1f} "
              f"{'✓' if adia else '✗ jump':>12} {bar}")
        results.append({"step": step, "s": s, "gs_prob": round(gsp, 4),
                        "gs_energy": gse, "adiabatic": adia})

    final = results[-1]["gs_prob"]
    initial = results[0]["gs_prob"]
    print(f"\n  Initial GS prob: {initial:.4f}  →  Final GS prob: {final:.4f}")
    print(f"  Transfer: {'SUCCESSFUL' if final > 0.15 else 'PARTIAL (increase STEPS)'}")

    save_result(7, "Adiabatic Morphing", {"steps": results, "adiabatic": ac})
    print("PRINCIPLE 7 DONE.")


# ═════════════════════════════════════════════════════════════════════════════
# P8: AI DECOHERENCE PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
def run_p8():
    print_header(8, "AI DECOHERENCE PREDICTOR — ML ON REAL NOISE DATA")
    try:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        print("  ERROR: sklearn not installed.")
        print("  Run: pip install scikit-learn")
        print("  Or on Jetson: pip3 install scikit-learn")
        return

    # Real experimental data: [depth, n_qubits, heron_gen, backend_qubits, algo]
    # algo: 0=Bell 1=GHZ 2=Grover 3=QAOA 4=VQE 5=BV 6=ZNE
    X = np.array([
        [4,   2, 1, 133, 0],   # Bell ibm_torino
        [6,   3, 1, 133, 1],   # GHZ ibm_torino
        [12,  2, 1, 133, 2],   # Grover 2Q ibm_torino
        [4,   2, 2, 156, 0],   # Bell ibm_marrakesh
        [6,   3, 2, 156, 1],   # GHZ ibm_marrakesh
        [12,  2, 2, 156, 2],   # Grover 2Q ibm_marrakesh
        [4,   2, 2, 156, 0],   # Bell ibm_kingston
        [6,   3, 2, 156, 1],   # GHZ ibm_kingston
        [107, 4, 2, 156, 3],   # QAOA p=2 ibm_kingston
        [596, 4, 2, 156, 2],   # Grover 4Q ibm_fez
        [9,   5, 2, 156, 5],   # BV ibm_kingston
        [12,  2, 2, 156, 6],   # ZNE ibm_fez
    ])
    y = np.array([11.62, 5.08, 16.31, 3.03, 5.86, 3.80,
                  2.34,  4.20, 6.12,  52.0, 9.70, 6.84])

    poly  = PolynomialFeatures(degree=2, include_bias=True)
    Xp    = poly.fit_transform(X)
    ridge = Ridge(alpha=2.0); ridge.fit(Xp, y)
    gb    = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                      learning_rate=0.1, random_state=42)
    gb.fit(X, y)

    print(f"\n  Training samples: {len(X)} real IBM Quantum experiments")
    print(f"  Ridge R²: {ridge.score(Xp, y):.4f}")
    print(f"  GBM R²:   {gb.score(X, y):.4f}")

    test_circuits = [
        ([9,   4, 2, 156, 5], "BV 4Q ibm_kingston         (actual: 9.70%)"),
        ([160, 6, 2, 156, 3], "QAOA 6-node ibm_fez"),
        ([20,  5, 2, 156, 1], "GHZ 5Q ibm_fez"),
        ([596, 4, 2, 156, 2], "Grover 4Q ibm_fez          (actual: 52.0%)"),
        ([9,   5, 2, 156, 5], "BV 5Q Jetson ibm_fez       (actual: 8.6%)"),
        ([200, 5, 3, 156, 3], "Future Heron r3 QAOA"),
        ([1000,6, 2, 156, 2], "Deep circuit stress test"),
    ]

    print(f"\n{'Circuit':<50} {'Ridge':>8} {'GBM':>8} {'Risk'}")
    print("-"*76)
    preds = []
    for feat, name in test_circuits:
        rp = max(0, ridge.predict(poly.transform([feat]))[0])
        gp = max(0, gb.predict([feat])[0])
        risk = "EXTREME" if gp > 40 else ("HIGH" if gp > 15 else ("MED" if gp > 7 else "LOW"))
        print(f"{name:<50} {rp:>7.2f}% {gp:>7.2f}% {risk}")
        preds.append({"name": name, "ridge": round(rp, 2),
                      "gbm": round(gp, 2), "risk": risk})

    print(f"\n  Depth-noise lookup (4Q Heron r2 circuits):")
    print(f"{'Depth':>8} {'Predicted Noise':>18} {'Usable?'}")
    print("-"*35)
    for d in [4, 9, 20, 50, 100, 200, 400, 596, 800, 1000]:
        pred   = max(0, gb.predict([[d, 4, 2, 156, 3]])[0])
        usable = "YES" if pred < 20 else ("MARGINAL" if pred < 40 else "NO")
        print(f"{d:>8} {pred:>17.2f}% {usable}")

    save_result(8, "Decoherence Predictor", {
        "ridge_r2":    round(ridge.score(Xp, y), 4),
        "gbm_r2":      round(gb.score(X, y), 4),
        "predictions": preds
    })
    print("PRINCIPLE 8 DONE.")


# ═════════════════════════════════════════════════════════════════════════════
# P9: QUANTUM PARALLELISM + AI ALGORITHM SELECTOR
# ═════════════════════════════════════════════════════════════════════════════
def run_p9():
    print_header(9, "QUANTUM PARALLELISM + AI ALGORITHM SELECTOR")

    def grover_search(n_bits, target_idx):
        target = format(target_idx, f'0{n_bits}b')
        iters  = max(1, int(np.pi / 4 * np.sqrt(2 ** n_bits)))
        anc    = n_bits

        qc = QuantumCircuit(n_bits + 1, n_bits)
        qc.x(anc); qc.h(range(n_bits + 1))

        def oracle():
            for i, b in enumerate(reversed(target)):
                if b == '0': qc.x(i)
            qc.h(anc)
            qc.mcx(list(range(n_bits)), anc)
            qc.h(anc)
            for i, b in enumerate(reversed(target)):
                if b == '0': qc.x(i)

        def diffuser():
            qc.h(range(n_bits))
            qc.x(range(n_bits))
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
            qc.x(range(n_bits))
            qc.h(range(n_bits))

        for _ in range(iters): oracle(); diffuser()
        qc.measure(range(n_bits), range(n_bits))  # explicit

        t0 = time.perf_counter()
        counts = sim.run(transpile(qc, sim), shots=1024).result().get_counts()
        t = time.perf_counter() - t0
        success_prob = counts.get(target, 0) / 1024
        return iters, t, success_prob

    def ai_selector(n_bits, qubit_budget=20):
        space      = 2 ** n_bits
        q_iters    = max(1, int(np.pi / 4 * np.sqrt(space)))
        c_avg      = space // 2
        depth_est  = q_iters * (n_bits + 2) * 3
        noise_est  = min(0.9, 0.02 * depth_est / 100)
        eff        = (c_avg / q_iters) * (1 - noise_est)
        if n_bits > qubit_budget:
            return "CLASSICAL", "exceeds qubit budget"
        if noise_est > 0.15:
            return "CLASSICAL", f"noise {noise_est:.0%} too high"
        if eff < 2.0:
            return "CLASSICAL", "speedup <2x after noise"
        return "QUANTUM", f"effective speedup {eff:.1f}x"

    print(f"\n{'Bits':>5} {'Space':>8} {'Classical':>12} {'Grover Q':>10} "
          f"{'Speedup':>8} {'P(target)':>10} {'AI Decision'}")
    print("-"*70)
    results = []
    for n in [2, 3, 4, 5, 6]:
        target_idx = 2 ** (n - 1) + 1
        iters, t, prob = grover_search(n, target_idx)
        c_avg   = 2 ** (n - 1)
        speedup = c_avg / iters
        decision, reason = ai_selector(n)
        print(f"{n:>5} {2**n:>8,} {c_avg:>12,} {iters:>10} "
              f"{speedup:>8.1f}x {prob:>9.1%} {decision}")
        results.append({"n": n, "speedup": round(speedup, 1),
                        "prob": round(prob, 4), "ai": decision})

    print(f"\n  AI Decision Boundary (larger problem sizes):")
    print(f"{'Bits':>5} {'Q Iters':>10} {'Eff Speedup':>13} {'AI Decision':>14} {'Reason'}")
    print("-"*72)
    for n in [2, 5, 8, 12, 16, 20, 25, 30]:
        q  = max(1, int(np.pi / 4 * np.sqrt(2 ** n)))
        ca = 2 ** (n - 1)
        de, re = ai_selector(n)
        print(f"{n:>5} {q:>10,} {ca/q:>13.1f} {de:>14} {re}")

    save_result(9, "Quantum Parallelism", {"scaling": results})
    print("PRINCIPLE 9 DONE.")


# ═════════════════════════════════════════════════════════════════════════════
# P10: NO-CLONING QKD
# ═════════════════════════════════════════════════════════════════════════════
def run_p10():
    print_header(10, "NO-CLONING PROTECTION — BB84 QKD")

    def bb84(n_bits=80, eve=False, eve_rate=1.0):
        np.random.seed(int(time.time()) % 10000)
        ab    = np.random.randint(0, 2, n_bits)
        abase = np.random.randint(0, 2, n_bits)
        bbase = np.random.randint(0, 2, n_bits)
        ebase = np.random.randint(0, 2, n_bits) if eve else None
        results = []

        for i in range(n_bits):
            qc = QuantumCircuit(1, 1)
            if ab[i]    == 1: qc.x(0)
            if abase[i] == 1: qc.h(0)

            if eve and np.random.random() < eve_rate:
                qce = qc.copy()
                if ebase[i] == 1: qce.h(0)
                qce.measure(0, 0)
                ce  = sim.run(transpile(qce, sim), shots=1).result().get_counts()
                eb  = int(list(ce.keys())[0])
                qc  = QuantumCircuit(1, 1)
                if eb       == 1: qc.x(0)
                if ebase[i] == 1: qc.h(0)

            if bbase[i] == 1: qc.h(0)
            qc.measure(0, 0)
            cb = sim.run(transpile(qc, sim), shots=1).result().get_counts()
            results.append(int(list(cb.keys())[0]))

        sifted = [(int(ab[i]), results[i]) for i in range(n_bits)
                  if abase[i] == bbase[i]]
        if not sifted: return 0.0, 0
        errors = sum(a != b for a, b in sifted)
        return errors / len(sifted), len(sifted)

    scenarios = [
        ("Clean (no Eve)",  False, 0.0),
        ("Eve 30%",         True,  0.3),
        ("Eve 60%",         True,  0.6),
        ("Eve 100%",        True,  1.0),
    ]
    print(f"\n{'Scenario':<25} {'Avg QBER':>10} {'Detected?':>14} {'Key bits':>10}")
    print("-"*62)
    all_res = {}
    for name, eve, rate in scenarios:
        qbers = [bb84(80, eve, rate)[0] for _ in range(10)]
        avg   = np.mean(qbers)
        det   = avg > 0.11
        print(f"{name:<25} {avg:>10.4f} {'YES — ABORT' if det else 'NO — SECURE':>14} ~40")
        all_res[name] = {"avg_qber": round(avg, 4), "detected": bool(det)}

    print("\n--- No-Cloning Theorem Demo ---")
    print("  Testing: CNOT clones basis states but fails for superposition\n")

    # States to test — using explicit measure(0,0) and measure(1,1)
    for desc, prep_fn in [
        ("|0⟩  basis state — CNOT should clone",  lambda qc: None),
        ("|1⟩  basis state — CNOT should clone",  lambda qc: qc.x(0)),
        ("|+⟩  superposition — CNOT CANNOT clone", lambda qc: qc.h(0)),
        ("|i⟩  Y-eigenstate — CNOT CANNOT clone",  lambda qc: (qc.h(0), qc.s(0))),
    ]:
        qc = QuantumCircuit(2, 2)
        prep_fn(qc)
        qc.cx(0, 1)
        qc.measure(0, 0)   # explicit — no reversal
        qc.measure(1, 1)
        counts  = sim.run(transpile(qc, sim), shots=1024).result().get_counts()
        perfect = (counts.get('00', 0) + counts.get('11', 0)) / 1024
        cloned  = perfect > 0.95
        print(f"  {desc}")
        print(f"    Perfect clone rate: {perfect:.3f}  "
              f"→  {'CLONED ✓' if cloned else 'FAILED — No-Cloning Theorem verified ✓'}\n")

    save_result(10, "No-Cloning QKD", {"scenarios": all_res})
    print("PRINCIPLE 10 DONE.")


# ═════════════════════════════════════════════════════════════════════════════
# P11: ALGORITHM FUSION — QAOA + GROVER ON REAL IBM HARDWARE
# ═════════════════════════════════════════════════════════════════════════════
def run_p11():
    print_header(11, "ALGORITHM FUSION — QAOA + GROVER HYBRID ON REAL IBM HARDWARE")

    N     = 4
    EDGES = [(0,1),(1,2),(2,3),(3,0),(0,2)]

    def cut_val(bs):
        """
        Max-Cut value for bitstring bs.
        Uses explicit measure so no reversal needed.
        """
        b = bs.replace(' ', '')
        if len(b) < N: return 0
        return sum(1 for i, j in EDGES if b[i] != b[j])

    optimal_states = [bs for bs in [format(i, f'0{N}b') for i in range(2**N)]
                      if cut_val(bs) == 4]
    print(f"\n  Max-Cut problem: 4-node graph, {len(EDGES)} edges")
    print(f"  Optimal states (cut=4): {optimal_states}")

    def fusion_circuit(gamma, beta, target='1010'):
        qc = QuantumCircuit(N, N)
        qc.h(range(N))
        # QAOA cost layer
        for q1, q2 in EDGES:
            qc.cx(q1, q2); qc.rz(2 * gamma, q2); qc.cx(q1, q2)
        # QAOA mixer layer
        for q in range(N): qc.rx(2 * beta, q)
        # Grover oracle marking target
        for i, bit in enumerate(target):
            if bit == '0': qc.x(i)
        qc.h(N-1)
        qc.mcx(list(range(N-1)), N-1)
        qc.h(N-1)
        for i, bit in enumerate(target):
            if bit == '0': qc.x(i)
        # Diffusion
        qc.h(range(N)); qc.x(range(N))
        qc.h(N-1); qc.mcx(list(range(N-1)), N-1); qc.h(N-1)
        qc.x(range(N)); qc.h(range(N))
        qc.measure(range(N), range(N))   # explicit
        return qc

    def pure_qaoa(gamma, beta):
        qc = QuantumCircuit(N, N)
        qc.h(range(N))
        for q1, q2 in EDGES:
            qc.cx(q1, q2); qc.rz(2 * gamma, q2); qc.cx(q1, q2)
        for q in range(N): qc.rx(2 * beta, q)
        qc.measure(range(N), range(N))   # explicit
        return qc

    def cost_fn(params):
        gamma, beta = params
        qc = fusion_circuit(gamma, beta)
        counts = sim.run(transpile(qc, sim), shots=512).result().get_counts()
        exp_cut = sum(cnt * cut_val(bs) for bs, cnt in counts.items()) / 512
        return -exp_cut

    print("\n  Optimising QAOA+Grover parameters...")
    x0  = np.array([np.pi/4, np.pi/4])
    res = minimize(cost_fn, x0, method='COBYLA',
                   options={'maxiter': 120, 'rhobeg': 0.5})
    g, b_ang = res.x
    print(f"  γ={g:.4f} β={b_ang:.4f} | {res.nfev} evals | expected cut={-res.fun:.4f}")

    # Simulator comparison
    counts_fusion = sim.run(transpile(fusion_circuit(g, b_ang),  sim), shots=2048).result().get_counts()
    counts_pure   = sim.run(transpile(pure_qaoa(g, b_ang),       sim), shots=2048).result().get_counts()

    opt_fusion = sum(counts_fusion.get(bs, 0) for bs in optimal_states) / 2048
    opt_pure   = sum(counts_pure.get(bs,   0) for bs in optimal_states) / 2048

    print(f"\n  Simulator — Optimal state probability:")
    print(f"    Pure QAOA:    {opt_pure:.4f}")
    print(f"    QAOA+Grover:  {opt_fusion:.4f}")
    print(f"    Fusion gain:  {opt_fusion - opt_pure:+.4f}")

    print(f"\n  Top 8 states (Fusion circuit — simulator):")
    print(f"  {'State':<8} {'Count':>8} {'Pct':>8} {'Cuts':>6} {'Optimal?'}")
    print("  " + "-"*38)
    for bs, cnt in sorted(counts_fusion.items(), key=lambda x: -x[1])[:8]:
        cv = cut_val(bs)
        print(f"  {bs:<8} {cnt:>8} {cnt/2048*100:>7.1f}% {cv:>6}  "
              f"{'★ YES' if cv == 4 else ''}")

    # Real IBM Hardware
    print("\n  Real IBM Hardware...")
    hw = {}
    try:
        svc     = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = svc.least_busy(operational=True, simulator=False)
        print(f"  Backend: {backend.name} ({backend.num_qubits}Q)")

        qcp_t = transpile(pure_qaoa(g, b_ang),          backend, optimization_level=3)
        qcf_t = transpile(fusion_circuit(g, b_ang),     backend, optimization_level=3)
        print(f"  Pure QAOA depth: {qcp_t.depth()} | Fusion depth: {qcf_t.depth()}")

        sampler = SamplerV2(backend)
        jp = sampler.run([qcp_t], shots=1024)
        print(f"  Pure QAOA job: {jp.job_id()} — waiting...")
        cp_hw = dict(jp.result()[0].data.c.get_counts())

        jf = sampler.run([qcf_t], shots=1024)
        print(f"  Fusion job:    {jf.job_id()} — waiting...")
        cf_hw = dict(jf.result()[0].data.c.get_counts())

        op_hw = sum(cp_hw.get(bs, 0) for bs in optimal_states) / 1024
        of_hw = sum(cf_hw.get(bs, 0) for bs in optimal_states) / 1024

        print(f"\n  Real HW optimal prob — QAOA: {op_hw:.4f}  Fusion: {of_hw:.4f}")
        print(f"  Fusion gain on real HW: {of_hw - op_hw:+.4f}")

        hw = {
            "backend":         backend.name,
            "job_pure":        jp.job_id(),
            "job_fusion":      jf.job_id(),
            "opt_pure_hw":     round(op_hw, 4),
            "opt_fusion_hw":   round(of_hw, 4),
            "pure_depth":      qcp_t.depth(),
            "fusion_depth":    qcf_t.depth()
        }
    except Exception as e:
        print(f"  IBM error: {e}")
        hw = {"error": str(e)}

    save_result(11, "Algorithm Fusion", {
        "gamma": round(g, 4), "beta": round(b_ang, 4),
        "opt_sim_pure":   round(opt_pure,   4),
        "opt_sim_fusion": round(opt_fusion, 4),
        "real_hw": hw
    })
    print("PRINCIPLE 11 DONE.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARQ-AIC Principles 4-11")
    parser.add_argument("--principle", type=int, default=0,
                        help="Run specific principle 4-11 (0 = all)")
    args = parser.parse_args()

    runners = {
        4: run_p4,
        5: run_p5,
        6: run_p6,
        7: run_p7,
        8: run_p8,
        9: run_p9,
        10: run_p10,
        11: run_p11,
    }

    if args.principle == 0:
        for fn in runners.values(): fn()
    elif args.principle in runners:
        runners[args.principle]()
    else:
        print(f"Principle {args.principle} not in this file. Use 4-11.")
