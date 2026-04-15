import numpy as np, json, datetime, platform, time, os

def get_platform():
    node = platform.node()
    if "jetson" in node.lower() or "quantum" in node.lower():
        return "NVIDIA Jetson Orin Nano"
    return "Raspberry Pi 5"

def get_temp():
    for zone in range(6):
        try:
            t = float(open(f"/sys/class/thermal/thermal_zone{zone}/temp").read()) / 1000
            if 20 < t < 120: return round(t, 2)
        except: pass
    return 45.0

def get_cpu_load():
    try: return round(os.getloadavg()[0], 2)
    except: return 0.0

def compute_thermal_noise_rate(temp, base=0.001, alpha=0.0005, T0=50.0):
    return round(max(base, base + alpha * (temp - T0)), 6)

def thermal_decoherence_time(temp):
    Ea = 0.05; kT_ref = 323.15; kT = temp + 273.15
    return round(np.exp(-Ea * (1/kT - 1/kT_ref) * kT_ref), 4)

def adiabatic_condition(delta_E, evolution_time):
    dH_dt = 1.0 / evolution_time
    sat = dH_dt < delta_E**2
    return {"satisfied": sat, "lhs": round(dH_dt,6),
            "rhs": round(delta_E**2,6), "margin": round(delta_E**2/dH_dt,2)}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        return super().default(obj)

def save_result(pnum, pname, data, rdir="../results"):
    result = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": get_platform(), "temp": get_temp(), "load": get_cpu_load(),
        "principle": pnum, "name": pname, "data": data
    }
    os.makedirs(rdir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{rdir}/P{pnum:02d}_{pname.replace(' ','_')}_{ts}.json"
    with open(fname, "w") as f: json.dump(result, f, indent=2, cls=NumpyEncoder)
    return fname

def print_header(n, title):
    print("="*65)
    print(f"ARQ-AIC PRINCIPLE {n}: {title}")
    print(f"Platform: {get_platform()} | Temp: {get_temp()}°C | Load: {get_cpu_load()}")
    print("="*65)
