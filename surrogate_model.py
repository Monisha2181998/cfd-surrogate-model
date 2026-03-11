"""
==============================================================
 CFD Surrogate Model — NACA Airfoil Aerodynamic Predictor
 
 Task: Predict lift (Cl) and drag (Cd) coefficients from
       airfoil shape + flight condition parameters,
       replacing expensive CFD simulations.

 Inputs  (5 features):
   - max_camber_pct    : Maximum camber (0–9%)
   - camber_pos        : Camber position (1–9)
   - thickness_pct     : Airfoil thickness (6–24%)
   - angle_of_attack   : AoA in degrees (-6 to 16°)
   - reynolds_number   : Re number (500k–3M)

 Outputs (2 targets):
   - Cl : Lift coefficient
   - Cd : Drag coefficient

 Author: Monisha Ravi Kumar
 Application: TU Dresden / ScaDS.AI / DLR-SP — AAI Research Group
==============================================================

SETUP:
    pip install torch pandas numpy matplotlib seaborn scikit-learn

RUN:
    python surrogate_model.py

Expected time on CPU: ~3-5 minutes
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────
EPOCHS      = 300
BATCH_SIZE  = 256
LR          = 1e-3
HIDDEN      = [128, 256, 256, 128, 64]
SEED        = 42
DATA_PATH   = "naca_airfoil_dataset.csv"
OUTPUT_DIR  = "surrogate_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*58}")
print(f"  CFD Surrogate Model — NACA Airfoil Aerodynamic Predictor")
print(f"  Device  : {device}")
print(f"  Epochs  : {EPOCHS}")
print(f"{'='*58}\n")

# ── 1. Load & Prepare Data ────────────────────────────────────
print("[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Total samples   : {len(df):,}")
print(f"  Airfoil families: {df['airfoil'].nunique()}")

FEATURES = ["max_camber_pct", "camber_pos", "thickness_pct",
            "angle_of_attack", "reynolds_number"]
TARGETS  = ["Cl", "Cd"]

X = df[FEATURES].values.astype(np.float32)
y = df[TARGETS].values.astype(np.float32)

# Split: 70% train, 15% val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
X_train, X_val,  y_train, y_val  = train_test_split(X_temp, y_temp, test_size=0.176, random_state=SEED)

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}\n")

# Normalize inputs
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train)
y_val_s   = scaler_y.transform(y_val)

# ── 2. Dataset & DataLoader ───────────────────────────────────
class AirfoilDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(AirfoilDataset(X_train_s, y_train_s), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(AirfoilDataset(X_val_s,   y_val_s),   batch_size=BATCH_SIZE)

# ── 3. Surrogate Model Architecture ──────────────────────────
class SurrogateNet(nn.Module):
    """
    Deep fully-connected network with residual-style skip connections.
    
    Why this architecture?
    - Aerodynamic relationships are highly nonlinear (especially near stall)
    - Deep networks capture complex feature interactions
    - BatchNorm stabilizes training with mixed-scale inputs (Re vs AoA)
    - Dropout prevents overfitting on the structured dataset
    """
    def __init__(self, in_dim=5, out_dim=2, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = [128, 256, 256, 128, 64]

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.1)
            ]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model     = SurrogateNet(in_dim=5, out_dim=2, hidden=HIDDEN).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

params = sum(p.numel() for p in model.parameters())
print(f"[Model] Parameters: {params:,}")
print(f"[Model] Architecture: 5 → {' → '.join(map(str,HIDDEN))} → 2\n")

# ── 4. Training Loop ──────────────────────────────────────────
print("[4/6] Training...\n")
history = {"train_loss": [], "val_loss": [], "val_r2_Cl": [], "val_r2_Cd": []}
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_losses = []
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        pred = model(X_b)
        loss = criterion(pred, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())
    scheduler.step()

    # Validate
    model.eval()
    val_preds, val_true = [], []
    val_losses = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred = model(X_b)
            val_losses.append(criterion(pred, y_b).item())
            val_preds.append(pred.cpu().numpy())
            val_true.append(y_b.cpu().numpy())

    val_preds = scaler_y.inverse_transform(np.vstack(val_preds))
    val_true  = scaler_y.inverse_transform(np.vstack(val_true))

    r2_Cl = r2_score(val_true[:, 0], val_preds[:, 0])
    r2_Cd = r2_score(val_true[:, 1], val_preds[:, 1])
    avg_train = np.mean(train_losses)
    avg_val   = np.mean(val_losses)

    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["val_r2_Cl"].append(r2_Cl)
    history["val_r2_Cd"].append(r2_Cd)

    # Save best model
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"R² Cl: {r2_Cl:.4f} | R² Cd: {r2_Cd:.4f}")

print("\n[Training] Complete! Loading best model...\n")
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt"))

# ── 5. Final Test Evaluation ──────────────────────────────────
print("[5/6] Final Evaluation on Test Set...")
model.eval()
X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_s = model(X_test_t).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_s)
y_true = y_test

# Per-target metrics
metrics = {}
for i, target in enumerate(TARGETS):
    r2   = r2_score(y_true[:, i], y_pred[:, i])
    mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) /
                          (np.abs(y_true[:, i]) + 1e-8))) * 100
    metrics[target] = {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}
    print(f"  {target}: R²={r2:.4f} | MAE={mae:.4f} | "
          f"RMSE={rmse:.4f} | MAPE={mape:.2f}%")

# Save results
with open(f"{OUTPUT_DIR}/results.txt", "w") as f:
    f.write("CFD SURROGATE MODEL — RESULTS\n")
    f.write("="*45 + "\n\n")
    f.write(f"Architecture : 5 → {' → '.join(map(str,HIDDEN))} → 2\n")
    f.write(f"Parameters   : {params:,}\n")
    f.write(f"Epochs       : {EPOCHS}\n")
    f.write(f"Dataset      : {len(df):,} samples, {df['airfoil'].nunique()} airfoils\n\n")
    f.write("Test Set Metrics:\n")
    for t, m in metrics.items():
        f.write(f"  {t}: R²={m['R2']:.4f} | MAE={m['MAE']:.5f} | "
                f"RMSE={m['RMSE']:.5f} | MAPE={m['MAPE']:.2f}%\n")

# ── 6. Visualizations ─────────────────────────────────────────
print("\n[6/6] Generating plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("CFD Surrogate Model — NACA Airfoil Aerodynamic Predictor\n"
             "Neural Network vs CFD Simulation", fontsize=14, fontweight="bold")

# Plot 1: Predicted vs Actual — Cl
axes[0,0].scatter(y_true[:,0], y_pred[:,0], alpha=0.3, s=8, color="steelblue")
lims = [y_true[:,0].min(), y_true[:,0].max()]
axes[0,0].plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
axes[0,0].set_xlabel("CFD Cl (True)"); axes[0,0].set_ylabel("Surrogate Cl (Predicted)")
axes[0,0].set_title(f"Lift Coefficient Cl  (R²={metrics['Cl']['R2']:.4f})", fontweight="bold")
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# Plot 2: Predicted vs Actual — Cd
axes[0,1].scatter(y_true[:,1], y_pred[:,1], alpha=0.3, s=8, color="darkorange")
lims = [y_true[:,1].min(), y_true[:,1].max()]
axes[0,1].plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
axes[0,1].set_xlabel("CFD Cd (True)"); axes[0,1].set_ylabel("Surrogate Cd (Predicted)")
axes[0,1].set_title(f"Drag Coefficient Cd  (R²={metrics['Cd']['R2']:.4f})", fontweight="bold")
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# Plot 3: Polar curve for NACA 2412 at Re=1M
naca2412 = df[(df["airfoil"] == "NACA2412") & (df["reynolds_number"] == 1000000)].copy()
if len(naca2412) > 0:
    X_n = scaler_X.transform(naca2412[FEATURES].values.astype(np.float32))
    with torch.no_grad():
        y_n = scaler_y.inverse_transform(
            model(torch.tensor(X_n, dtype=torch.float32).to(device)).cpu().numpy())
    axes[0,2].plot(naca2412["Cl"].values, naca2412["Cd"].values,
                   "b-o", markersize=5, label="CFD (true)", linewidth=2)
    axes[0,2].plot(y_n[:,0], y_n[:,1],
                   "r--s", markersize=5, label="Surrogate", linewidth=2)
    axes[0,2].set_xlabel("Cl"); axes[0,2].set_ylabel("Cd")
    axes[0,2].set_title("Polar Curve — NACA 2412 @ Re=1M", fontweight="bold")
    axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

# Plot 4: Training loss curves
axes[1,0].semilogy(history["train_loss"], label="Train Loss", color="blue")
axes[1,0].semilogy(history["val_loss"],   label="Val Loss",   color="red")
axes[1,0].set_title("Training Loss (log scale)", fontweight="bold")
axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("MSE Loss")
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# Plot 5: R² over training
axes[1,1].plot(history["val_r2_Cl"], label="R² Cl", color="steelblue", linewidth=2)
axes[1,1].plot(history["val_r2_Cd"], label="R² Cd", color="darkorange", linewidth=2)
axes[1,1].axhline(y=0.99, color="green", linestyle="--", alpha=0.5, label="R²=0.99")
axes[1,1].set_title("R² Score During Training", fontweight="bold")
axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("R² Score")
axes[1,1].set_ylim(0, 1.02); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

# Plot 6: AoA sweep for 3 airfoils
axes[1,2].set_title("Cl vs Angle of Attack — 3 Airfoils", fontweight="bold")
colors = ["steelblue", "darkorange", "green"]
for color, airfoil_name in zip(colors, ["NACA0012", "NACA2412", "NACA4412"]):
    subset = df[(df["airfoil"] == airfoil_name) & (df["reynolds_number"] == 1000000)]
    if len(subset) == 0:
        continue
    X_s = scaler_X.transform(subset[FEATURES].values.astype(np.float32))
    with torch.no_grad():
        y_s = scaler_y.inverse_transform(
            model(torch.tensor(X_s, dtype=torch.float32).to(device)).cpu().numpy())
    axes[1,2].plot(subset["angle_of_attack"].values, subset["Cl"].values,
                   "-", color=color, linewidth=2, label=f"{airfoil_name} (CFD)")
    axes[1,2].plot(subset["angle_of_attack"].values, y_s[:,0],
                   "--", color=color, linewidth=1.5, alpha=0.7, label=f"{airfoil_name} (Surrogate)")

axes[1,2].set_xlabel("Angle of Attack (°)"); axes[1,2].set_ylabel("Cl")
axes[1,2].legend(fontsize=7); axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/surrogate_results.png", dpi=150, bbox_inches="tight")
print(f"  Saved → {OUTPUT_DIR}/surrogate_results.png\n")

# ── 7. Demo Inference ─────────────────────────────────────────
print("Demo — Predicting aerodynamics for new airfoils:\n")
demo_cases = [
    {"airfoil": "NACA 2412", "m": 2, "p": 4, "t": 12, "aoa": 5,  "Re": 1e6},
    {"airfoil": "NACA 0012", "m": 0, "p": 1, "t": 12, "aoa": 0,  "Re": 1e6},
    {"airfoil": "NACA 4415", "m": 4, "p": 4, "t": 15, "aoa": 8,  "Re": 3e6},
    {"airfoil": "NACA 6409", "m": 6, "p": 4, "t": 9,  "aoa": 10, "Re": 5e5},
]

for case in demo_cases:
    x_in = np.array([[case["m"], case["p"], case["t"],
                       case["aoa"], case["Re"]]], dtype=np.float32)
    x_s  = scaler_X.transform(x_in)
    with torch.no_grad():
        y_out = scaler_y.inverse_transform(
            model(torch.tensor(x_s).to(device)).cpu().numpy())
    Cl, Cd = y_out[0]
    LD = Cl / Cd if Cd > 0 else 0
    print(f"  {case['airfoil']} | AoA={case['aoa']:3d}° | Re={case['Re']/1e6:.1f}M "
          f"→  Cl={Cl:.4f}  Cd={Cd:.5f}  L/D={LD:.1f}")

print(f"\n{'='*58}")
print(f"  DONE!  R² Cl={metrics['Cl']['R2']:.4f}  R² Cd={metrics['Cd']['R2']:.4f}")
print(f"  Results → {OUTPUT_DIR}/results.txt")
print(f"  Plots   → {OUTPUT_DIR}/surrogate_results.png")
print(f"{'='*58}\n")
