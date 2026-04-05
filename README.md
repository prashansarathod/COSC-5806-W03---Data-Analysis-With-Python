# COSC-5806-W03---Data-Analysis-With-Python


# Multivariate Time-Series Health Index and Early Anomaly Detection Framework for Wind Turbine Predictive Maintenance Using SCADA Data

## Project Overview

Wind turbines operate under varying environmental and mechanical conditions. Continuous monitoring of turbine operational parameters is essential to prevent unexpected failures and reduce maintenance costs. Supervisory Control and Data Acquisition (SCADA) systems collect operational measurements such as wind speed, power output, rotor speed, and component temperatures.

This project develops a **data-driven predictive maintenance framework** for wind turbines using SCADA data. The framework applies statistical analysis and machine learning techniques to monitor turbine behavior and detect abnormal operational patterns that may indicate potential faults.

The project uses **multivariate time-series analysis, regression modeling, Principal Component Analysis (PCA), and anomaly detection using Isolation Forest** to construct a turbine health monitoring system.

---

# Dataset

This project uses a real-world wind turbine dataset:

**Vestas V52 – 10 Minute SCADA Dataset (2006–2020)**
Source: Mendeley Data

Dataset Link
[https://data.mendeley.com/datasets/tm988rs48k/2](https://data.mendeley.com/datasets/tm988rs48k/2)

### Dataset Characteristics

Sampling Interval: **10 minutes**

Monitoring Period: **January 2006 – March 2020**

The dataset contains several operational variables including:

| Variable     | Description                    |
| ------------ | ------------------------------ |
| Timestamps   | Measurement timestamp          |
| WindSpeed    | Wind speed measured at turbine |
| Power        | Electrical power output        |
| RotorRPM     | Rotor rotational speed         |
| GenRPM       | Generator rotational speed     |
| GearOilTemp  | Gearbox oil temperature        |
| GearBearTemp | Gearbox bearing temperature    |
| GenBearTemp  | Generator bearing temperature  |
| NacelTemp    | Temperature inside nacelle     |
| Pitch        | Blade pitch angle              |
| StdDevPower  | Power output variability       |

These parameters provide important information about turbine operational behavior and mechanical condition.

---

# Project Workflow

The project follows a structured predictive maintenance pipeline.

## Step 1: Data Loading

The SCADA dataset is loaded using **Pandas**.

```python
import pandas as pd

df = pd.read_csv("data/VestasV52_10_min_raw_SCADA_DkIT 30_Jan2006-12_Mar2020.csv")
```

---

# Step 2: Data Preprocessing

Data preprocessing ensures the dataset accurately reflects turbine operational behavior.

Main preprocessing steps include:

* Convert timestamps to datetime format
* Remove invalid sensor values
* Replace placeholder values (999) with missing values
* Filter unrealistic operational states
* Sort dataset chronologically

Example:

```python
df['Timestamps'] = pd.to_datetime(df['Timestamps'], format='mixed')
df = df.sort_values('Timestamps')
df.set_index('Timestamps', inplace=True)

df = df[(df['Power'] > 0) & (df['WindSpeed'] > 0)]
```

---

# Step 3: Exploratory Data Analysis (EDA)

EDA was performed to understand turbine operational behavior.

The following visualizations were generated:

* Wind Speed vs Power (Power Curve)
* Temperature trends
* Sensor distributions
* Correlation matrix between turbine variables

Example visualization:

```python
plt.scatter(df['WindSpeed'], df['Power'])
plt.xlabel("Wind Speed")
plt.ylabel("Power Output")
plt.title("Wind Turbine Power Curve")
plt.show()
```

This analysis confirmed that the dataset reflects realistic turbine operational conditions.

---

# Step 4: Power Curve Regression Modeling

A **polynomial regression model** was used to model the nonlinear relationship between wind speed and turbine power output.

The model predicts expected power output:

Power = f(WindSpeed)

Example implementation:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = df[['WindSpeed']]
y = df['Power']

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

df['PredictedPower'] = model.predict(X_poly)
```

---

# Step 5: Residual Analysis

Residuals measure deviation between predicted and actual power output.

Residual formula:

Residual = Actual Power − Predicted Power

```python
df['Residual'] = df['Power'] - df['PredictedPower']
```

Large residual values may indicate abnormal turbine behavior.

---

# Step 6: Feature Engineering

Additional features were created to capture turbine operational conditions.

Temperature difference features:

```python
df['GearTempDiff'] = df['GearOilTemp'] - df['EnvirTemp']
df['BearTempDiff'] = df['GearBearTemp'] - df['EnvirTemp']
df['GenBearTempDiff'] = df['GenBearTemp'] - df['EnvirTemp']
```

Rolling statistical features:

```python
df['PowerRollingMean'] = df['Power'].rolling(12).mean()
df['PowerRollingStd'] = df['Power'].rolling(12).std()
```

These features help capture mechanical stress and operational variability.

---

# Step 7: Turbine Health Index Using PCA

Principal Component Analysis (PCA) was applied to combine multiple turbine variables into a **single health indicator**.

Example implementation:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

features = [
'Power','RotorRPM','GearOilTemp','GearBearTemp',
'GenBearTemp','GearTempDiff','BearTempDiff',
'GenBearTempDiff','PowerRollingMean','PowerRollingStd','Residual'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

pca = PCA(n_components=1)
df['HealthIndex'] = pca.fit_transform(X_scaled)
```

This health index summarizes turbine operational condition.

---

# Step 8: Anomaly Detection Using Isolation Forest

Isolation Forest was used to detect abnormal operational patterns.

```python
from sklearn.ensemble import IsolationForest

features = [
'Residual','GearTempDiff','BearTempDiff',
'GenBearTempDiff','PowerRollingStd','HealthIndex'
]

model = IsolationForest(contamination=0.01, random_state=42)

df['Anomaly'] = model.fit_predict(df[features])
df['Anomaly'] = df['Anomaly'].map({1:0, -1:1})
```

Approximately **1% of observations were detected as anomalies**.

---

# Step 9: Early Warning Detection

To detect sustained abnormal behavior, a rolling anomaly window was implemented.

```python
df['AnomalyRolling'] = df['Anomaly'].rolling(12).sum()
df['EarlyWarning'] = df['AnomalyRolling'] >= 6
```

This identifies periods where multiple anomalies occur within **2 hours of turbine operation**.

---

# Results

Key results of the analysis:

* Dataset duration: **2006 – 2020**
* Sampling interval: **10 minutes**
* Total records analyzed: **~653,000**
* Anomalies detected: **~1% of observations**
* Early warning signals detected: **3829 periods**

The framework successfully identifies abnormal turbine behavior using multivariate SCADA data.

---

# Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

---

# Repository Structure

```
project/
│
├── data/
│   └── SCADA dataset
│
├── notebook/
│   └── project.ipynb
│
├── report/
│   └── weekly reports
│
└── README.md
```

---

# Future Work

Potential improvements to this framework include:

* Deep learning models for time-series forecasting
* Remaining useful life (RUL) prediction
* Real-time anomaly detection systems
* Deployment as a monitoring dashboard

---
