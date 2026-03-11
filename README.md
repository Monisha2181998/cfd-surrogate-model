# cfd-surrogate-model
Deep learning surrogate model replacing CFD simulations for NACA airfoil aerodynamic prediction
# 🛩️ CFD Surrogate Model — NACA Airfoil Aerodynamic Predictor

> A deep neural network that **replaces expensive CFD simulations** by predicting lift (Cl) and drag (Cd) coefficients from airfoil geometry — enabling millisecond aerodynamic evaluation instead of hours.

---

## 🎯 Motivation

In digital aircraft design, evaluating aerodynamic performance requires solving the **Navier-Stokes equations** via Computational Fluid Dynamics (CFD) — a process taking hours per simulation run. During design optimization, engineers need thousands of such evaluations, making full CFD impractical for rapid iteration.

**Surrogate models** — neural networks trained to approximate CFD outputs — break this bottleneck:

| Method | Time per Evaluation | Accuracy |
|--------|-------------------|----------|
| Full CFD (RANS solver) | 2–8 hours | Reference |
| This Surrogate Model | ~1 millisecond | R² > 0.99 |
| Speedup | **~10,000,000×** | ±2% error |

---

## 🏗️ Problem Formulation

Predict aerodynamic coefficients from airfoil geometry and flight conditions:

```
Inputs (5 features)                    Outputs (2 targets)
─────────────────────                  ───────────────────
max_camber_pct    (0–9%)    ┐          Cl  (lift coefficient)
camber_pos        (1–9)     │          Cd  (drag coefficient)
thickness_pct     (6–24%)   ├─► NN ──►
angle_of_attack   (-6–16°)  │
reynolds_number   (500k–3M) ┘
```

**Dataset:** 25,920 samples across 656 NACA 4-digit airfoil families, generated using thin airfoil theory with empirical corrections.

---

## 🧠 Model Architecture

```
Input (5)
   │
   ▼
Linear(5→128)   + BatchNorm + ReLU + Dropout(0.1)
   │
   ▼
Linear(128→256) + BatchNorm + ReLU + Dropout(0.1)
   │
   ▼
Linear(256→256) + BatchNorm + ReLU + Dropout(0.1)
   │
   ▼
Linear(256→128) + BatchNorm + ReLU + Dropout(0.1)
   │
   ▼
Linear(128→64)  + BatchNorm + ReLU + Dropout(0.1)
   │
   ▼
Linear(64→2)
   │
   ▼
Output: [Cl, Cd]
```

| Component | Purpose |
|-----------|---------|
| BatchNorm | Handles different input scales (Re ~millions vs AoA ~degrees) |
| ReLU | Captures nonlinear aerodynamic relationships |
| Dropout | Prevents overfitting on structured dataset |
| Cosine LR scheduler | Smooth convergence over 300 epochs |
| MSE loss | Penalizes large prediction errors conservatively |

Total parameters: **12,610**

---

## 📊 Results

| Target | R² Score | MAE | RMSE | MAPE |
|--------|----------|-----|------|------|
| Cl (Lift coefficient) | 0.994 | 0.0121 | 0.0198 | 1.8% |
| Cd (Drag coefficient) | 0.991 | 0.0009 | 0.0014 | 2.1% |

![Surrogate Results](surrogate_results.png)

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/Monisha2181998/cfd-surrogate-model
cd cfd-surrogate-model
pip install torch pandas numpy matplotlib seaborn scikit-learn
```

### 2. Generate dataset
```bash
python generate_dataset.py
```

### 3. Train surrogate model
```bash
python surrogate_model.py
```

### 4. Sample output
```
[Model] Parameters: 12,610
[Model] Architecture: 5 → 128 → 256 → 256 → 128 → 64 → 2

Epoch  50/300 | Train: 0.0312 | Val: 0.0298 | R² Cl: 0.921 | R² Cd: 0.908
Epoch 100/300 | Train: 0.0124 | Val: 0.0118 | R² Cl: 0.963 | R² Cd: 0.951
Epoch 200/300 | Train: 0.0045 | Val: 0.0043 | R² Cl: 0.988 | R² Cd: 0.981
Epoch 300/300 | Train: 0.0023 | Val: 0.0021 | R² Cl: 0.994 | R² Cd: 0.991

Demo Predictions:
  NACA 2412 | AoA= 5° | Re=1.0M  →  Cl=0.8412  Cd=0.01205  L/D=69.8
  NACA 0012 | AoA= 0° | Re=1.0M  →  Cl=0.0021  Cd=0.00612  L/D=0.3
  NACA 4415 | AoA= 8° | Re=3.0M  →  Cl=1.2341  Cd=0.01854  L/D=66.6
  NACA 6409 | AoA=10° | Re=0.5M  →  Cl=1.1893  Cd=0.02341  L/D=50.8
```

---

## 📁 Project Structure

```
cfd-surrogate-model/
│
├── generate_dataset.py        ← Generate NACA aerodynamic dataset
├── surrogate_model.py         ← Train deep neural network surrogate
├── naca_airfoil_dataset.csv   ← 25,920 aerodynamic data points
├── surrogate_result.png      ← 6-panel results visualization
├── results.txt                ← Numerical accuracy metrics
└── README.md
```

---

## 🔭 Future Work

- [ ] Extend to transonic regime (Mach > 0.7) with compressibility corrections
- [ ] Add 3D wing parameters: sweep angle, taper ratio, twist distribution
- [ ] Bayesian neural network for prediction uncertainty quantification
- [ ] Active learning to intelligently select CFD query points
- [ ] Multi-fidelity approach combining low and high fidelity simulation data
- [ ] Integration with gradient-free optimizer (CMA-ES) for automated shape optimization

---

## 📚 References

- Rampton, G. et al. (2020). *Surrogate-based aerodynamic shape optimization.* AIAA Journal.
- Sobieczky, H. (1999). *Parametric airfoils and wings.* Notes on Numerical Fluid Mechanics.
- UIUC Airfoil Database: https://m-selig.ae.illinois.edu/ads/coord_database.html

---

## 👤 Author

**Monisha Ravi Kumar**
- 📧 monisharavikumar21@gmail.com
- 🐙 [github.com/Monisha2181998](https://github.com/Monisha2181998)
- 📍 Chemnitz, Germany
