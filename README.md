# Integration of Component Sizing and DRL-Based Energy Management Strategy for Fuel Cell Hybrid Electric Vehicles (FCHEVs)####

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![Optimization](https://img.shields.io/badge/Optimization-PSO%20%2B%20SAC-success.svg)](https://github.com/AliBaghani1/Integration-of-Component-sizing-and-DRL-based-energy-management-strategy-for-FCEVs)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)](#)

---

## 📌 Authors & Contributors

* **Ali Baghani** — [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8423--970X-green.svg)](https://orcid.org/0000-0002-8423-970X)
* **Mohammad Hosein Kazemi** — [![ORCID](https://img.shields.io/badge/ORCID-0009--0009--5610--0346-green.svg)](https://orcid.org/0009-0009-5610-0346)

---

## 📖 Executive Summary

This repository presents an end-to-end framework for the **simultaneous component sizing and Deep Reinforcement Learning (DRL)-based Energy Management Strategy (EMS)** for Fuel Cell Hybrid Electric Vehicles (FCHEVs), modeled after the **Toyota Mirai** powertrain architecture.

The research integrates:
1. **High-Fidelity Physics & Degradation Modeling**: Validated longitudinal vehicle dynamics, electrical equivalent circuit battery modeling with electro-thermal-aging dynamics, and a Proton Exchange Membrane Fuel Cell (PEMFC) stack model considering multi-regime durability degradation (high load, low load, load transients, and start-stop cycles).
2. **Soft Actor-Critic (SAC) Deep Reinforcement Learning EMS**: An entropy-regularized continuous control EMS that optimizes instant power split between the fuel cell stack and high-voltage battery to minimize total operating costs while safeguarding electrochemical health.
3. **Bi-Level Co-Optimization (PSO + SAC)**: A nested optimization loop utilizing Particle Swarm Optimization (PSO) in the outer loop for powertrain component sizing ($Q_{\text{bat}}$, $C_{\text{rate,max}}$, and $\alpha_{\text{power}}$) and SAC in the inner loop for real-time optimal energy control, achieving minimum **200,000 km Total Cost of Ownership (TCO)**.
4. **Experimental Validation**: Rigorous calibration against real-world chassis dynamometer test data (`61712010.csv`) from the **Argonne National Laboratory (ANL) Advanced Powertrain Research Facility (APRF)**.

```
+===================================================================================================+
|                                      BI-LEVEL CO-OPTIMIZATION                                     |
+===================================================================================================+
|                                                                                                   |
|  [ OUTER LOOP: Particle Swarm Optimization (PSO) ]                                                |
|  Candidate Sizing Vector: [ Q_bat (kWh), C_rate_max, alpha_power ]                               |
|        │                                                   ▲                                      |
|        │ (Component Sizes)                                 │ (200k km TCO Cost)                   |
|        ▼                                                   │                                      |
|  [ INNER LOOP: Soft Actor-Critic (SAC) DRL Agent ] ────────┘                                      |
|  Train Continuous Policy π_θ(a|s) over Driving Cycles (e.g. UDDS)                                 |
|  State: s = [ P_req, SOC ] ──► Action: a (P_fc) ──► Environment (Mirai FCHEV Model)              |
|                                                                                                   |
|  ┌───────────────────────────────── Vehicle Environment ──────────────────────────────────┐       |
|  │ Longitudinal Dynamics ──► Motor Efficiency Map ──► Power Split (P_fc, P_bat)           │       |
|  │ PEMFC Degradation (ΔV_d) + Battery Aging (ΔSOH) + H2 Consumption + SOC Deviation       │       |
|  └────────────────────────────────────────────────────────────────────────────────────────┘       |
+===================================================================================================+
```

---

## 🗂 Repository Structure

```text
Integration-of-Component-sizing-and-DRL-based-energy-management-strategy-for-FCEVs/
├── 01_FCHEV model validation/
│   ├── 61712010.csv                     # Experimental chassis dynamometer test dataset (ANL APRF)
│   └── vehicle model validation.ipynb   # Complete vehicle dynamics, component & baseline validation
├── 02_SAC-based EMS/
│   └── Energy managment stratgy via SAC.ipynb  # Soft Actor-Critic DRL algorithm for optimal power split
├── 03_Integrated optimization of component size and EMS/
│   └── Co-optimization of companent sizing and EMS via PSO & SAC.ipynb  # Nested PSO-SAC co-optimization
└── README.md                            # Comprehensive project documentation
```

---

## ⚙️ Powertrain Architecture & Modeling Details

### 1. Vehicle Longitudinal Dynamics
The total tractive effort $F_{\text{tr}}$ required to propel the vehicle is formulated via 1D longitudinal vehicle dynamics:

$$F_{\text{tr}} = F_{\text{roll}} + F_{\text{aero}} + F_{\text{grade}} + F_{\text{inertia}}$$

* **Rolling Resistance**: $F_{\text{roll}} = m_{\text{veh}} \cdot g \cdot f_r \cdot \cos(\theta)$
* **Aerodynamic Drag**: $F_{\text{aero}} = \frac{1}{2} \rho_{\text{air}} C_d A_f v^2$
* **Grading Force**: $F_{\text{grade}} = m_{\text{veh}} \cdot g \cdot \sin(\theta)$
* **Inertial Resistance**: $F_{\text{inertia}} = m_{\text{veh}} \cdot a + 4 \frac{J_{\text{wheel}}}{R_{\text{wheel}}} \dot{\omega}_{\text{wheel}}$

The total vehicle mass dynamically scales with component sizing:

$$m_{\text{veh}} = 1793.45 + \frac{P_{\text{max,fc}}}{1.6} + \frac{Q_{\text{bat}} \cdot 1000}{24.5} \quad [\text{kg}]$$

### 2. Drivetrain & Electric Machine (EM)
The wheel torque and angular velocity are transferred to the electric motor through the final drive / fixed-gear reducer ($\text{GR} = 9.09, \eta_{\text{gb}} = 0.98$):

$$\omega_{\text{EM}} = \omega_{\text{wheel}} \cdot \text{GR}$$

$$T_{\text{gb}} = \begin{cases} \frac{T_{\text{wheel}}}{\eta_{\text{gb}} \cdot \text{GR}}, & T_{\text{wheel}} \ge 0 \quad (\text{Traction}) \\ \frac{T_{\text{wheel}} \cdot \eta_{\text{gb}}}{\text{GR}}, & T_{\text{wheel}} < 0 \quad (\text{Regeneration}) \end{cases}$$

$$T_{\text{EM}} = J_{\text{EM}} \dot{\omega}_{\text{EM}} + T_{\text{gb}}$$

The electric machine efficiency $\eta_{\text{EM}}(\omega_{\text{EM}}, T_{\text{EM}})$ is evaluated via 2D bivariate spline interpolation extracted from experimental dynamometer efficiency maps. The total required electrical power includes auxiliary systems load ($P_{\text{aux}} = 440\text{ W}$):

$$P_{\text{req}} = \begin{cases} \left( \frac{P_{\text{EM}}}{\eta_{\text{EM}}} + P_{\text{aux}} \right) \times 10^{-3}, & P_{\text{EM}} \ge 0 \\ (P_{\text{EM}} \cdot \eta_{\text{EM}} + P_{\text{aux}}) \times 10^{-3}, & P_{\text{EM}} < 0 \end{cases} \quad [\text{kW}]$$

---

### 3. Proton Exchange Membrane Fuel Cell (PEMFC) Model
* **Hydrogen Consumption**: Calculated from the fuel cell thermodynamic power $P_{\text{th}} = \left(\frac{\eta_{\text{stack}}}{\eta_{\text{sys}}}\right) P_{\text{fc}}$:
  
  $$\dot{m}_{\text{H}_2} = 8.5 \times 10^{-5} P_{\text{th}}^2 + 9.1 \times 10^{-3} P_{\text{th}} + 0.0064 \quad [\text{g/s}]$$

* **Multi-Factor PEMFC Durability & Degradation Model**: Stack voltage loss rate $\Delta V_d$ accounts for four primary operational degradation modes:
  1. **High Load ($P_{\text{fc}}/P_{\text{max,fc}} \ge 0.8$)**: $\Delta V_{\text{high}} = \frac{\Delta t}{3600} \cdot 10.0 \times 10^{-6}\text{ V}$
  2. **Low Load ($P_{\text{fc}}/P_{\text{max,fc}} \le 0.2$)**: $\Delta V_{\text{low}} = \frac{\Delta t}{3600} \cdot 8.662 \times 10^{-6}\text{ V}$
  3. **Load Transients / Dynamics ($\Delta P_{\text{fc}}$)**: $\Delta V_{\text{trans}} = 4.185 \times 10^{-8} \cdot \frac{|\Delta P_{\text{fc}}|}{P_{\text{max,fc}}}\text{ V}$
  4. **Start-Stop Cycles ($P_{\text{fc,last}} = 0 \to P_{\text{fc}} > 0$)**: $\Delta V_{\text{ss}} = 13.79 \times 10^{-8}\text{ V}$

  $$\Delta V_d = \Delta V_{\text{high}} + \Delta V_{\text{low}} + \Delta V_{\text{trans}} + \Delta V_{\text{ss}}$$

  $$\text{SOH}_{\text{fc}}(t + \Delta t) = \text{SOH}_{\text{fc}}(t) - \frac{\Delta V_d}{0.07}$$

---

### 4. Battery Electro-Thermal & Degradation Model
* **Equivalent Circuit Model (ECM)**: State of Charge (SOC) tracking using Coulomb Counting with nonlinear open-circuit voltage $V_{\text{oc}}(\text{SOC})$ and internal resistance maps ($R_{\text{dis}}(\text{SOC}), R_{\text{chg}}(\text{SOC})$):
  
  $$I_{\text{dis}} = \frac{V_{\text{oc}} - \sqrt{V_{\text{oc}}^2 - 4 R_{\text{dis}} P_{\text{bat,dis}}}}{2 R_{\text{dis}}}, \quad I_{\text{chg}} = \frac{-V_{\text{oc}} + \sqrt{V_{\text{oc}}^2 + 4 R_{\text{chg}} P_{\text{bat,chg}}}}{2 R_{\text{chg}}}$$

  $$\Delta \text{SOC} = \frac{I_{\text{chg}} \cdot \Delta t}{Q_{\text{nom}}} - \frac{I_{\text{dis}}^{1.1} \cdot \Delta t}{Q_{\text{nom}}}$$

* **Semi-Empirical Capacity Loss & SOH**: Incorporates C-rate severity factor $M(C_{\text{rate}})$ and Arrhenius thermal dynamics:
  
  $$Q_{\text{loss}} = M(C_{\text{rate}}) \cdot \exp\left( \frac{-31700 + 370.3 \cdot C_{\text{rate}}}{8.31 \cdot T_{\text{bat}}} \right)$$

  $$\Delta \text{SOH}_{\text{bat}} = -\left( \frac{|I_{\text{bat}}| \cdot \Delta t}{A_{h,\text{tot}} \cdot 3600 \cdot 2} \right) \times 0.01$$

---

## 🧠 Stage 1: Validation Against ANL Test Data

The baseline vehicle model and rule-based EMS are validated against empirical dynamometer test data for the 2016 Toyota Mirai (`61712010.csv`).

```
                              VALIDATION RESULTS SUMMARY
┌───────────────────────────────┬───────────────────────────────┬────────────────────────┐
│ Parameter                     │ Experimental / Reference      │ Simulation Model       │
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ Final Battery SOC             │ ~57.5%                        │ Accurately Tracked     │
│ H2 Consumption Rate           │ Dynamometer Flowmeter Logged  │ High-Precision Match   │
│ Dynamic Power Split           │ Production ECU Logic Replic.  │ Dynamic Equilibrium    │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
```

Run validation notebook:
```bash
jupyter notebook "01_FCHEV model validation/vehicle model validation.ipynb"
```

---

## 🚀 Stage 2: Soft Actor-Critic (SAC) Deep Reinforcement Learning EMS

Soft Actor-Critic (SAC) is an off-policy actor-critic Deep RL algorithm based on the **maximum entropy reinforcement learning** framework, providing optimal power-split decisions while ensuring robust exploration.

```
                              SAC DRL AGENT ARCHITECTURE
                                
                             State s = [ P_req , SOC ]
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
          ┌───────────────────┐                   ┌───────────────────┐
          │   Actor Network   │                   │ Critic Networks   │
          │   (Gaussian π_θ)  │                   │ (Twin Q_ψ1, Q_ψ2) │
          └─────────┬─────────┘                   └─────────┬─────────┘
                    │ Action a ∈ [-1, 1]                    │
                    ▼                                       ▼
          P_fc = P_max,fc * (a + 1)/2             Q-Value Estimates (Min-Q)
                    │
                    ▼
          P_bat = P_req - P_fc (Power Split)
```

### 1. Markov Decision Process (MDP) Definition
* **State Space** $\mathcal{S} \in \mathbb{R}^2$: $s_t = [P_{\text{req}}(t), \text{SOC}(t)]$
* **Action Space** $\mathcal{A} \in [-1, 1]$: Mapped linearly to fuel cell stack power $P_{\text{fc}} \in [0, P_{\text{max,fc}}]$:
  
  $$P_{\text{fc}} = P_{\text{max,fc}} \left(\frac{a + 1}{2}\right), \quad P_{\text{bat}} = P_{\text{req}} - P_{\text{fc}}$$

### 2. Multi-Objective Cost & Reward Function
The instantaneous real operating cost is:

$$\text{Cost}_{\text{real}} = C_{\text{H}_2} + C_{\text{fc}} + C_{\text{bat}}$$

* **Hydrogen Cost**: $C_{\text{H}_2} = 0.03294 \cdot (\dot{m}_{\text{H}_2} + m_{\text{equ}})$ (accounting for equivalent electrical energy penalty $m_{\text{equ}}$)
* **Fuel Cell Degradation Cost**: $C_{\text{fc}} = P_{\text{max,fc}} \cdot 93 \cdot \left(\frac{\Delta V_d}{0.07}\right)$
* **Battery Degradation Cost**: $C_{\text{bat}} = Q_{\text{bat}} \cdot 139 \cdot |\Delta\text{SOH}_{\text{bat}}|$
* **SOC Regulation Cost**: $C_{\text{SOC}} = 1.5 \cdot (0.575 - \text{SOC})^2$

$$\mathcal{R}(s_t, a_t) = -\left( \text{Cost}_{\text{real}} + C_{\text{SOC}} \right)$$

### 3. Training & Hyperparameters
* **Actor Architecture**: 3-layer MLP (`State` $\to 256 \to 256 \to 256 \to (\mu, \log \sigma)$) with Gaussian reparameterization.
* **Critic Architecture**: Twin Q-Networks (`(State + Action)` $\to 256 \to 256 \to 256 \to Q$).
* **Replay Buffer**: $10^4$ transitions, Batch Size: $256$.
* **Discount Factor ($\gamma$)**: $0.9$, Soft Target Update ($\tau$): $0.01$.
* **Learning Rates**: $\alpha_{\text{actor}} = 5\times 10^{-4}, \alpha_{\text{critic}} = 5\times 10^{-4}$.
* **Entropy Temperature ($\alpha$)**: Automatically tuned via gradient descent against target entropy $\bar{\mathcal{H}} = -\dim(\mathcal{A}) = -1$.

Run SAC EMS notebook:
```bash
jupyter notebook "02_SAC-based EMS/Energy managment stratgy via SAC.ipynb"
```

---

## 🔄 Stage 3: Bi-Level Co-Optimization (PSO + SAC)

Powertrain component sizing and energy management are inherently coupled. A bi-level optimization scheme is employed:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              BI-LEVEL CO-OPTIMIZATION LOOP                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   [ OUTER LEVEL - Particle Swarm Optimization (PSO) ]                                  │
│   Sizing Decision Vector:  X = [ Q_bat , C_rate_max , alpha_power ]                    │
│   Bounds:                                                                              │
│     • Q_bat       ∈ [1.0, 5.0]  kWh                                                    │
│     • C_rate_max  ∈ [2.0, 14.0]                                                        │
│     • alpha_power ∈ [1.0, 1.4]                                                         │
│                                                                                        │
│   Powertrain Dimensioning:                                                             │
│     • P_bat,max = C_rate_max * Q_bat                                                   │
│     • P_fc,max  = (96 * alpha_power) - P_bat,max                                       │
│                                                                                        │
│             │                                                     ▲                    │
│             ▼                                                     │                    │
│   [ INNER LEVEL - Soft Actor-Critic (SAC) DRL Training ]          │                    │
│   • Instantiate Environment with (P_fc,max, Q_bat)                │                    │
│   • Train DRL Agent over Driving Cycles                           │                    │
│   • Evaluate Best Realized Operating Cost: Cost_real,best         │                    │
│             │                                                     │                    │
│             └─────────────────────────────────────────────────────┘                    │
│                                                                                        │
│   Global Objective Function (Total Cost of Ownership for 200,000 km):                   │
│                                                                                        │
│       TCO = ( 200,000 / D_cycle ) * Cost_real,best + (93 * P_fc,max) + (139 * Q_bat)   │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Run co-optimization notebook:
```bash
jupyter notebook "03_Integrated optimization of component size and EMS/Co-optimization of companent sizing and EMS via PSO & SAC.ipynb"
```

---

## 📊 Key Results & Comparative Advantages

| Metric / Aspect | Conventional Rule-Based EMS | SAC-Based EMS (Fixed Size) | Integrated PSO + SAC Co-Optimization |
| :--- | :--- | :--- | :--- |
| **Component Sizing** | Fixed (Baseline) | Fixed (Baseline) | **Optimally Sized** ($Q_{\text{bat}}^*, P_{\text{fc}}^*$) |
| **Hydrogen Economy** | Baseline | **+8–12% Improvement** | **Maximum System Efficiency** |
| **Stack & Battery Health** | Passive Monitoring | Active Degradation Penalization | **Minimized Lifetime Aging Costs** |
| **200,000 km Lifecycle TCO** | High | Moderately Reduced | **Lowest Global Cost (Optimal Sizing & EMS)** |
| **Action Domain** | Rule Heuristics | Continuous ($a \in [-1, 1]$) | Continuous Adaptive |

---

## 💻 Installation & Dependencies

### 1. Prerequisites
* Python 3.8+ (Recommended: Python 3.9 or 3.10)
* CUDA-compatible GPU (recommended for PyTorch neural network training)

### 2. Environment Setup
Clone the repository and install required packages:

```bash
# Clone repository
git clone https://github.com/AliBaghani1/Integration-of-Component-sizing-and-DRL-based-energy-management-strategy-for-FCEVs.git
cd Integration-of-Component-sizing-and-DRL-based-energy-management-strategy-for-FCEVs

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy matplotlib pyswarms jupyter ipykernel
```

---

## 🛠️ Step-by-Step Execution Guide

1. **Model Validation**:
   - Open `01_FCHEV model validation/vehicle model validation.ipynb`.
   - Update file path to `61712010.csv`.
   - Run all cells to verify simulated power, hydrogen consumption, and SOC curves against experimental ANL dynamometer traces.

2. **SAC EMS Training**:
   - Open `02_SAC-based EMS/Energy managment stratgy via SAC.ipynb`.
   - Configure output save directory (`dir = '...'`) and driving cycle path (e.g. `Standard_UDDS.mat`).
   - Execute training loop (`MAX_EPISODES = 150`).
   - Monitor learning curves, episode scores, fuel economy, SOH degradation, and saved policy weights (`final_actor.h5`, `final_critic1.h5`).

3. **Bi-Level Co-Optimization**:
   - Open `03_Integrated optimization of component size and EMS/Co-optimization of companent sizing and EMS via PSO & SAC.ipynb`.
   - Set particle swarm size (`n_particles`), iterations (`iters`), and search boundaries (`bounds`).
   - Launch outer-loop PSO optimizer to discover the Pareto-optimal powertrain sizing and EMS weights.

---

## 📚 References & Data Sources

1. **Argonne National Laboratory (ANL)**: Advanced Powertrain Research Facility (APRF) Downloadable Dynamometer Database (D3) — *2016 Toyota Mirai Test Data*.
2. **Soft Actor-Critic (SAC)**: Haarnoja, T., et al. *"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"*, ICML 2018.
3. **PSO Algorithm**: Kennedy, J. and Eberhart, R., *"Particle Swarm Optimization"*, IEEE ICNN 1995 (`pyswarms` toolkit).
4. **PEMFC & Battery Degradation**: Multi-factor electrochemical degradation and thermal modeling formulations adapted from leading literature in hydrogen fuel cell hybrid vehicle durability.

---

## 📄 Citation

If you find this research or code useful for your work, please cite the authors:

```bibtex
@article{baghani2024integration,
  title={Integration of Component Sizing and DRL-Based Energy Management Strategy for Fuel Cell Hybrid Electric Vehicles},
  author={Baghani, Ali and Kazemi, Mohammad Hosein},
  year={2024},
  publisher={GitHub},
  journal={GitHub Repository},
  howpublished={\url{https://github.com/AliBaghani1/Integration-of-Component-sizing-and-DRL-based-energy-management-strategy-for-FCEVs}}
}
```

---

## 📬 Contact & Support

For academic collaboration, inquiries, or bug reports:
* **Ali Baghani** — [ORCID Profile](https://orcid.org/0000-0002-8423-970X)
* **Mohammad Hosein Kazemi** — [ORCID Profile](https://orcid.org/0009-0009-5610-0346)
