"""
Generate a realistic NACA 4-digit airfoil aerodynamic dataset.
Based on thin airfoil theory + empirical corrections.
Saves to naca_airfoil_dataset.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def compute_aero(m, p, t, alpha, Re):
    """
    Compute realistic Cl and Cd for a NACA 4-digit airfoil.
    m = max camber (0-9%), p = camber position (1-9), t = thickness (6-24%)
    alpha = angle of attack (degrees), Re = Reynolds number
    """
    alpha_rad = np.radians(alpha)

    # Lift coefficient (thin airfoil theory + camber correction)
    alpha_zero = -2 * m * (1 - 2*p/10) * np.pi / 10  # zero-lift angle
    Cl_linear = 2 * np.pi * (alpha_rad - alpha_zero)

    # Stall model — Cl drops after stall angle
    stall_angle = np.radians(12 + t * 30)
    if abs(alpha_rad) > stall_angle:
        stall_factor = 1 - 3 * (abs(alpha_rad) - stall_angle)**2
        stall_factor = max(0.2, stall_factor)
        Cl = Cl_linear * stall_factor
    else:
        Cl = Cl_linear

    Cl = np.clip(Cl, -1.8, 2.0)

    # Drag coefficient (profile + induced + thickness drag)
    Cd_min   = 0.005 + 0.008 * t + 1.2e-6 / (Re / 1e6)
    Cd_lift  = (Cl ** 2) / (np.pi * 8.0 * 0.85)   # induced drag
    Cd_stall = max(0, 0.02 * (abs(alpha_rad) - stall_angle)) if abs(alpha_rad) > stall_angle else 0
    Cd       = Cd_min + Cd_lift + Cd_stall
    Cd       = np.clip(Cd, 0.003, 0.25)

    # Moment coefficient about quarter-chord
    Cm = -np.pi * m / 10 * (1 - 2*p/10) - 0.05 * alpha_rad
    Cm = np.clip(Cm, -0.25, 0.05)

    return round(Cl, 5), round(Cd, 5), round(Cm, 5)

rows = []
airfoil_names = []

# Generate across all common NACA 4-digit families
for m in range(0, 10):           # 0–9% camber
    for p in range(1, 10):       # camber position 1–9
        for t_pct in [6,8,10,12,15,18,21,24]:  # thickness %
            t = t_pct / 100
            for alpha in np.arange(-6, 18, 2):   # AoA: -6 to 16 deg
                for Re in [5e5, 1e6, 3e6]:         # Reynolds numbers
                    Cl, Cd, Cm = compute_aero(m/100, p, t, alpha, Re)

                    # NACA name string e.g. NACA2412
                    if m == 0:
                        name = f"NACA00{t_pct:02d}"
                    else:
                        name = f"NACA{m}{p}{t_pct:02d}"

                    rows.append({
                        "airfoil"        : name,
                        "max_camber_pct" : m,
                        "camber_pos"     : p,
                        "thickness_pct"  : t_pct,
                        "angle_of_attack": alpha,
                        "reynolds_number": int(Re),
                        "Cl"             : Cl,
                        "Cd"             : Cd,
                        "Cm"             : Cm,
                        "LD_ratio"       : round(Cl/Cd, 2) if Cd > 0 else 0
                    })

df = pd.DataFrame(rows)

# Remove physically unrealistic rows
df = df[df["Cd"] > 0]
df = df[df["LD_ratio"].abs() < 200]

print(f"Dataset shape   : {df.shape}")
print(f"Airfoil families: {df['airfoil'].nunique()}")
print(f"Cl range        : [{df['Cl'].min():.3f}, {df['Cl'].max():.3f}]")
print(f"Cd range        : [{df['Cd'].min():.4f}, {df['Cd'].max():.4f}]")
print(f"\nSample rows:")
print(df[df["airfoil"] == "NACA2412"].head(6).to_string(index=False))

df.to_csv("naca_airfoil_dataset.csv", index=False)
print(f"\nSaved → naca_airfoil_dataset.csv ({len(df):,} rows)")
