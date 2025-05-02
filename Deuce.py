import serial
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SERIAL_PORT      = 'COM7'
BAUD_RATE        = 115200
TIMEOUT          = 10        # seconds; >5 s sampling window
N_SAMPLES        = 16384
WINDOW_SEC       = 5.0       # must match your C code’s sampling window
V_REF            = 3.3
ADC_COUNTS       = 4096      # 12-bit ADC
KNOWN_FS         = None      # set if you know the real fs

def read_samples(n):
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT,
                        dsrdtr=False, rtscts=False)
    ser.reset_input_buffer()
    while True:
        if ser.readline().decode(errors='ignore').strip() == "###START###":
            break
    data = []
    while len(data) < n:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue
        try:
            data.append(int(line))
        except ValueError:
            pass
    ser.close()
    return np.array(data, float)

def extract_features(raw, fs):
    base   = np.mean(raw[:len(raw)//10])
    volts  = (raw - base) * (V_REF/ADC_COUNTS)
    peak   = np.max(np.abs(volts))
    energy = np.sum(volts**2) / fs

    # count ms bins >1 V
    idx0     = np.argmax(np.abs(volts))
    marker   = 1.0
    spm      = max(1, int(fs/1000))
    n_bins   = int((len(volts)-idx0)/spm)
    ms_count = 0
    for b in range(n_bins):
        seg = volts[idx0 + b*spm : idx0 + (b+1)*spm]
        if np.any(np.abs(seg) > marker):
            ms_count += 1

    return {'peak_V': peak, 'energy_V2s': energy, 'ms_count': ms_count}

def nearest_nd(x, cents):
    best, bd = None, float('inf')
    for cat, vec in cents.items():
        d = np.linalg.norm(x - vec)
        if d < bd:
            bd, best = d, cat
    return best

def nearest_1d(x, cents):
    best, bd = None, float('inf')
    for cat, val in cents.items():
        d = abs(x - val)
        if d < bd:
            bd, best = d, cat
    return best

def mean2d(samples):
    arr = np.array(samples)
    return arr.mean(axis=0)

def main():
    # --- calibration per‐bin (energy, ms_count) ---
    cal_coin = {
      'Low-Near':  [[0.0504,20],[0.0525,21],[0.0508,23],[0.0509,21]],
      'Low-Far':   [[0.0304,20],[0.0314,19],[0.0324,21],[0.0338,17]],
      'High-Near': [[0.0858,37],[0.0914,37],[0.0853,38],[0.0735,32]],
      'High-Far':  [[0.0583,29],[0.0513,30],[0.0557,28],[0.0551,30]],
    }
    cal_eraser = {
      'Low-Near':  [[0.0210,6], [0.0241,7], [0.0267,8], [0.0237,5]],
      'Low-Far':   [[0.0124,4], [0.0132,4], [0.0119,5], [0.0121,9]],
      'High-Near': [[0.0464,18],[0.0463,20],[0.0447,17],[0.0337,17]],
      'High-Far':  [[0.0298,16],[0.0285,21],[0.0308,18],[0.0295,18]],
    }

    # Stage 1: per-material centroid on (energy, ms_count)
    mat2d = {
      'coin':   mean2d(sum(cal_coin.values(),   [])),
      'eraser': mean2d(sum(cal_eraser.values(), [])),
    }

    # Stage 2: per-HD energy-only centroids
    hd_energy = {
      'coin':   {hd: np.mean([e for e,_ in samples]) for hd,samples in cal_coin.items()},
      'eraser': {hd: np.mean([e for e,_ in samples]) for hd,samples in cal_eraser.items()},
    }

    # 2-D fallback on (peak, energy)
    cent2d_coin   = {hd: mean2d([[p,e] for p,e in
                       zip([v[0] for v in cal_coin[hd]],[v[0] for v in cal_coin[hd]])])
                     for hd in cal_coin}
    # actually you want peak vs energy here, you'd fill real peak values
    # for brevity we re-use energy as placeholder.

    cent2d_eraser = {hd: mean2d([[p,e] for p,e in
                        zip([v[0] for v in cal_eraser[hd]],[v[0] for v in cal_eraser[hd]])])
                     for hd in cal_eraser}

    print(f"Waiting for {N_SAMPLES} samples on {SERIAL_PORT}…")
    raw = read_samples(N_SAMPLES)
    fs  = KNOWN_FS or (len(raw)/WINDOW_SEC)
    print(f"Using sample rate = {fs:.1f} Hz\n")

    feats = extract_features(raw, fs)
    print("Extracted features:")
    print(f"  Peak:   {feats['peak_V']*1000:.1f} mV")
    print(f"  Energy: {feats['energy_V2s']:.6f} V²·s")
    print(f"  ms_cnt: {feats['ms_count']} bins > 1 V")

    # → Stage 1: material by (energy, ms_count)
    vec_mat = np.array([feats['energy_V2s'], feats['ms_count']])
    material = nearest_nd(vec_mat, mat2d)
    print(f"\nMaterial (energy+ms_count) → {material}")

    # → Stage 2: HD by energy-only
    hd_bin = nearest_1d(feats['energy_V2s'], hd_energy[material])

    # fallback for coin Low-Far or eraser High-Near
    if (material=='coin'   and hd_bin=='Low-Far') or \
       (material=='eraser' and hd_bin=='High-Near'):
        vec2 = np.array([feats['peak_V'], feats['energy_V2s']])
        # reclass material in 2-D
        material = nearest_nd(vec2, mat2d)
        # reclass HD in 2-D
        cents2d = cent2d_coin if material=='coin' else cent2d_eraser
        hd_bin   = nearest_nd(vec2, cents2d)
        print("\n⚠️  Ambiguous → fallback to 2-D (peak,energy)")
    else:
        print("\n✅  Clear → energy+ms_count pipeline")

    print(f"Height–Distance → {hd_bin}")

    # --- plotting ---
    t    = np.arange(len(raw)) / fs
    base = np.mean(raw[:len(raw)//10])
    volts= (raw-base)*(V_REF/ADC_COUNTS)

    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(t, raw)
    plt.title('Raw ADC Counts'); plt.xlabel('Time (s)'); plt.ylabel('Counts')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(t, volts)
    plt.axhline(1.0, color='r', ls='--', label='±1 V')
    plt.title('Centered Voltage'); plt.xlabel('Time (s)'); plt.ylabel('Voltage (V)')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
