# AutoPulse ML: Automotive OBD-II Engine & Powertrain Fault Classification

**Course:** M.Tech (Data Science and Engineering) — Machine Learning (BITS Pilani WILP)  
**Live Streamlit App:** 'https://auto-obd2-fault-classifier-bits.streamlit.app/'

---

## a. Problem Statement
Modern connected vehicles rely on On-Board Diagnostics II (OBD-II) and CAN-bus electronic control units (ECUs) to monitor internal combustion and hybrid powertrain health in real time. Unexpected powertrain faults—such as fuel-rail pressure loss, lubrication degradation, cooling system failure, air-fuel ratio misfires, and cylinder knock—cause costly breakdowns and severe engine damage. This project implements, evaluates, and deploys five supervised machine learning classification models to predict whether a vehicle's operating state is **Normal Operation (`0`)** or an **Active Powertrain / Engine Fault (`1`)** using 14 real-time OBD-II telemetry features.

---

## b. Dataset Description
* **Dataset Name:** Automotive OBD-II Engine & Powertrain Fault Diagnostics Dataset (modeled on the public Kaggle/UCI *Automotive Engine Health & OBD-II Telemetry* benchmark).
* **Instance Size:** **1,600 instances** (Exceeds minimum requirement of 500 instances).
  * Training Set (80% Stratified): 1,280 instances
  * Test Set (`test_data.csv`, 20% Stratified): 320 instances (180 Normal `0`, 140 Fault `1`)
* **Feature Size:** **14 continuous OBD-II sensor features** + 1 binary target `engine_fault_status` (Exceeds minimum requirement of 12 features):
  1. `engine_rpm` – Crankshaft rotational speed (650 to 6,200 rpm)
  2. `lub_oil_pressure_bar` – Main gallery lubricating oil pressure (0.80 to 6.50 bar)
  3. `fuel_rail_pressure_bar` – Common-rail fuel injection pressure (35.0 to 220.0 bar)
  4. `coolant_pressure_bar` – Engine coolant loop pressure (0.60 to 3.20 bar)
  5. `lub_oil_temp_c` – Engine lubricating oil sump temperature (65.0 to 135.0 °C)
  6. `coolant_temp_c` – Engine cylinder head coolant temperature (60.0 to 125.0 °C)
  7. `maf_airflow_gps` – Mass Air Flow sensor intake rate (3.5 to 85.0 g/s)
  8. `throttle_pos_pct` – Electronic throttle valve position (4.0 to 98.0 %)
  9. `intake_manifold_kpa` – Intake Manifold Absolute Pressure (22.0 to 145.0 kPa)
  10. `exhaust_gas_temp_c` – Pre-catalyst Exhaust Gas Temperature (220.0 to 820.0 °C)
  11. `short_term_fuel_trim_pct` – Short-Term Fuel Trim STFT correction (-22.0 to +25.0 %)
  12. `long_term_fuel_trim_pct` – Long-Term Fuel Trim LTFT adaptive offset (-18.0 to +20.0 %)
  13. `knock_sensor_mv` – Piezoelectric cylinder knock sensor amplitude (20.0 to 480.0 mV)
  14. `battery_alternator_v` – Alternator / ECU charging bus voltage (11.20 to 15.40 V)

---

## c. GitHub Repository Link
* **Repository URL:** `https://github.com/vinayakahavaldar0-sketch/automotive-obd2-ml-diagnostics-bits`

---

## d. Models Used & Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9250 | 0.9702 | 0.9085 | 0.9214 | 0.9149 | 0.8479 |
| **Decision Tree** | 0.8719 | 0.8679 | 0.8779 | 0.8214 | 0.8487 | 0.7390 |
| **kNN** | 0.9625 | 0.9881 | 0.9706 | 0.9429 | 0.9565 | 0.9239 |
| **Naive Bayes** | 0.8438 | 0.9309 | 0.8082 | 0.8429 | 0.8252 | 0.6845 |
| **Support Vector Machine** | **0.9750** | **0.9912** | **0.9783** | **0.9643** | **0.9712** | **0.9492** |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved a strong linear baseline (Accuracy = 92.50%, AUC = 0.9702, MCC = 0.8479). Captures additive thermal and pressure trends well, but its linear hyperplane misses nonlinear multi-sensor interactions (such as high RPM + lean fuel trim + knock voltage), resulting in 24 total errors on 320 test instances. |
| **Decision Tree** | Recorded Accuracy = 87.19%, AUC = 0.8679, and MCC = 0.7390 (`max_depth=7`). Axis-parallel rectangular splits struggle to approximate smooth thermodynamic ratios across air-fuel flow and manifold pressure, leading to 25 missed faults. |
| **kNN** | Ranked 2nd across all six metrics (Accuracy = 96.25%, AUC = 0.9881, F1 = 0.9565, MCC = 0.9239) using `k=7` and inverse-distance weighting on standardized features. Local neighborhood clusters effectively isolate normal vs. faulty operating regimes with only 12 test errors. |
| **Naive Bayes** | Gaussian Naive Bayes yielded the lowest Accuracy (84.38%) and MCC (0.6845), though keeping an AUC of 0.9309. Its conditional independence assumption is violated because engine RPM, MAF airflow, throttle position, manifold pressure, and EGT are physically coupled. |
| **Support Vector Machine** | Ranked 1st across every metric (Accuracy = 97.50%, AUC = 0.9912, Precision = 97.83%, Recall = 96.43%, F1 = 0.9712, MCC = 0.9492). The RBF kernel maps coupled OBD-II telemetry features into a higher-dimensional Hilbert space where multi-mode faults become linearly separable with only 8 total errors out of 320 test vehicles. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF Kernel)** is the overall winner, achieving the highest AUC (0.9912), F1 (0.9712), and MCC (0.9492), detecting 96.43% of true powertrain faults while maintaining 97.83% precision. |
