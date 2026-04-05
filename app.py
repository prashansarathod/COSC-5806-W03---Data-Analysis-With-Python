import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


st.set_page_config(
    page_title="Wind Turbine Predictive Maintenance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- GLOBAL DARK THEME ----------------------
st.markdown("""
<style>
    /* Main app */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    header {
        background: #0b1220 !important;
        color: #f8fafc !important;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95rem;
        color: #f8fafc !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6,
    p, div, label, span, li {
        color: #f8fafc !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0f172a !important;
        border: 1px dashed #475569 !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p {
        color: #e2e8f0 !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploader"] button {
        background: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: #334155 !important;
        border-color: #64748b !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
    }

    [data-testid="metric-container"] label,
    [data-testid="metric-container"] div {
        color: #f8fafc !important;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
        padding: 4px !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #334155 !important;
        border-color: #64748b !important;
    }

    /* Slider */
    .stSlider label {
        color: #f8fafc !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: #102a4c !important;
        color: #f8fafc !important;
        border: 1px solid #2563eb !important;
        border-radius: 12px !important;
    }

    /* Code block fix in sidebar */
    pre, code {
        background: #111827 !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }

    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}
</style>
""", unsafe_allow_html=True)

# Fix uploader instruction text specifically
st.sidebar.markdown(
    """
    <style>
    [data-testid='stFileUploaderDropzoneInstructions'] {
        color: #e2e8f0 !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------- HELPER UI CARDS ----------------------
def sidebar_card(title, content_html):
    st.sidebar.markdown(
        f"""
        <div style="
            background: linear-gradient(180deg, #132f57 0%, #12345f 100%);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 14px 16px;
            margin: 10px 0 18px 0;
            color: #f8fafc;">
            <div style="font-size: 1rem; font-weight: 700; margin-bottom: 8px;">{title}</div>
            <div style="font-size: 0.95rem; line-height: 1.6;">{content_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def expected_columns_card(cols):
    html_items = "".join(
        [f"<div style='margin:4px 0;'>• {c}</div>" for c in cols]
    )
    st.sidebar.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 14px 14px;
            margin-top: 6px;
            margin-bottom: 18px;">
            <div style="font-size: 1rem; font-weight: 700; margin-bottom: 10px; color:#f8fafc;">
                Expected columns
            </div>
            <div style="font-size: 0.95rem; color:#e2e8f0; line-height:1.5;">
                {html_items}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------- TITLE ----------------------
st.title("Wind Turbine Predictive Maintenance Dashboard")
st.write("SCADA-based health monitoring, anomaly detection, and early warning system.")


# ---------------------- DATA FUNCTIONS ----------------------
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df


def preprocess_data(df):
    df = df.copy()

    if "Timestamps" not in df.columns:
        st.error("Column 'Timestamps' not found in dataset.")
        return None

    df["Timestamps"] = pd.to_datetime(df["Timestamps"], format="mixed", errors="coerce")

    # Replace placeholder invalid values
    df.replace(999, np.nan, inplace=True)

    # Drop fully empty columns
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)

    required_cols = [
        "WindSpeed", "Power", "RotorRPM", "EnvirTemp",
        "GearOilTemp", "GearBearTemp", "GenBearTemp"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None

    # Remove rows with missing key operating values
    df = df.dropna(subset=["WindSpeed", "Power", "RotorRPM"])

    # Keep realistic generating turbine rows
    df = df[(df["Power"] > 0) & (df["WindSpeed"] > 0)]
    df = df[df["WindSpeed"] < 30]

    # Sort by time
    df = df.sort_values("Timestamps")
    df.set_index("Timestamps", inplace=True)

    # Fill remaining numeric gaps
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

    # Drop any remaining missing values
    df.dropna(inplace=True)

    return df


def build_model_pipeline(df, contamination_value):
    df = df.copy()

    # ---------------- POWER CURVE REGRESSION ----------------
    X = df[["WindSpeed"]]
    y = df["Power"]

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)

    reg_model = LinearRegression()
    reg_model.fit(X_poly, y)

    df["PredictedPower"] = reg_model.predict(X_poly)
    df["Residual"] = df["Power"] - df["PredictedPower"]

    # ---------------- FEATURE ENGINEERING ----------------
    df["GearTempDiff"] = df["GearOilTemp"] - df["EnvirTemp"]
    df["BearTempDiff"] = df["GearBearTemp"] - df["EnvirTemp"]
    df["GenBearTempDiff"] = df["GenBearTemp"] - df["EnvirTemp"]

    df["PowerRollingMean"] = df["Power"].rolling(12).mean()
    df["PowerRollingStd"] = df["Power"].rolling(12).std()

    df.dropna(inplace=True)

    # ---------------- PCA HEALTH INDEX ----------------
    health_features = [
        "Power",
        "RotorRPM",
        "GearOilTemp",
        "GearBearTemp",
        "GenBearTemp",
        "GearTempDiff",
        "BearTempDiff",
        "GenBearTempDiff",
        "PowerRollingMean",
        "PowerRollingStd",
        "Residual"
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[health_features])

    pca = PCA(n_components=1)
    df["HealthIndex"] = pca.fit_transform(X_scaled)
    df["HealthIndexSmooth"] = df["HealthIndex"].rolling(12).mean()

    # ---------------- ANOMALY DETECTION ----------------
    anomaly_features = [
        "Residual",
        "GearTempDiff",
        "BearTempDiff",
        "GenBearTempDiff",
        "PowerRollingStd",
        "HealthIndex"
    ]

    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination_value,
        random_state=42
    )

    df["Anomaly"] = iso.fit_predict(df[anomaly_features])
    df["Anomaly"] = df["Anomaly"].map({1: 0, -1: 1})

    # ---------------- EARLY WARNING ----------------
    df["AnomalyRolling"] = df["Anomaly"].rolling(12).sum()
    df["EarlyWarning"] = df["AnomalyRolling"] >= 6

    df.dropna(inplace=True)

    return df, reg_model, poly


def style_axes(ax):
    ax.set_facecolor("#0f172a")
    ax.figure.patch.set_facecolor("#0b1220")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(alpha=0.15, color="white")


# ---------------------- SIDEBAR ----------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload SCADA dataset (.csv or .xlsx)",
    type=["csv", "xlsx"]
)

contamination_value = st.sidebar.slider(
    "Anomaly Sensitivity (Contamination)",
    min_value=0.01,
    max_value=0.10,
    value=0.07,
    step=0.01
)

sidebar_card(
    "Sensitivity guide",
    "Higher value = more anomalies detected (more sensitive)."
)

expected_columns = [
    "Timestamps",
    "WindSpeed",
    "Power",
    "RotorRPM",
    "EnvirTemp",
    "GearOilTemp",
    "GearBearTemp",
    "GenBearTemp"
]
expected_columns_card(expected_columns)

st.sidebar.markdown("### Selected Model Settings")
st.sidebar.write(f"Contamination: **{contamination_value:.2f}**")


# ---------------------- MAIN ----------------------
if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
    df = preprocess_data(raw_df)

    if df is not None and not df.empty:
        result_df, reg_model, poly = build_model_pipeline(df, contamination_value)

        # ---------------- OVERVIEW ----------------
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(result_df):,}")
        c2.metric("Anomaly Rate", f"{result_df['Anomaly'].mean() * 100:.2f}%")
        c3.metric("Early Warning Events", f"{int(result_df['EarlyWarning'].sum()):,}")

        st.write("Preview of processed data:")
        st.dataframe(result_df.head(10), use_container_width=True)

        # ---------------- POWER CURVE ----------------
        st.subheader("1. Wind Turbine Power Curve")
        fig1, ax1 = plt.subplots(figsize=(11, 5))
        style_axes(ax1)

        ax1.scatter(
            result_df["WindSpeed"],
            result_df["Power"],
            alpha=0.15,
            label="Actual"
        )

        ws = np.linspace(
            result_df["WindSpeed"].min(),
            result_df["WindSpeed"].max(),
            200
        ).reshape(-1, 1)

        ws_poly = poly.transform(ws)
        pred_curve = reg_model.predict(ws_poly)

        ax1.plot(ws, pred_curve, color="red", linewidth=2.5, label="Polynomial Fit")
        ax1.set_xlabel("Wind Speed")
        ax1.set_ylabel("Power Output")
        ax1.set_title("Power Curve with Polynomial Regression")
        ax1.legend(facecolor="#111827", edgecolor="#475569", labelcolor="white")
        st.pyplot(fig1, use_container_width=True)

        # ---------------- RESIDUALS ----------------
        st.subheader("2. Residual Analysis")
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        style_axes(ax2)

        ax2.scatter(result_df["WindSpeed"], result_df["Residual"], alpha=0.2)
        ax2.set_xlabel("Wind Speed")
        ax2.set_ylabel("Residual (Actual - Predicted)")
        ax2.set_title("Power Curve Residuals")
        st.pyplot(fig2, use_container_width=True)

        # ---------------- HEALTH INDEX ----------------
        st.subheader("3. Turbine Health Index")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        style_axes(ax3)

        ax3.plot(result_df.index, result_df["HealthIndexSmooth"], linewidth=1.5)
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Health Index")
        ax3.set_title("PCA-Based Turbine Health Index")
        st.pyplot(fig3, use_container_width=True)

        # ---------------- ANOMALIES ----------------
        st.subheader("4. Detected Anomalies")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        style_axes(ax4)

        ax4.plot(result_df.index, result_df["HealthIndex"], label="Health Index", linewidth=1.2)
        anomalies = result_df[result_df["Anomaly"] == 1]

        ax4.scatter(
            anomalies.index,
            anomalies["HealthIndex"],
            color="red",
            s=12,
            label="Anomaly"
        )
        ax4.set_title("Isolation Forest Anomaly Detection")
        ax4.legend(facecolor="#111827", edgecolor="#475569", labelcolor="white")
        st.pyplot(fig4, use_container_width=True)

        # ---------------- EARLY WARNINGS ----------------
        st.subheader("5. Early Warning Detection")
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        style_axes(ax5)

        ax5.plot(result_df.index, result_df["HealthIndex"], label="Health Index", linewidth=1.2)
        warnings = result_df[result_df["EarlyWarning"] == True]

        ax5.scatter(
            warnings.index,
            warnings["HealthIndex"],
            color="orange",
            s=20,
            label="Early Warning"
        )
        ax5.set_title("Early Warning Detection")
        ax5.legend(facecolor="#111827", edgecolor="#475569", labelcolor="white")
        st.pyplot(fig5, use_container_width=True)

        # ---------------- TABLES ----------------
        st.subheader("Detected Anomalies Table")
        st.dataframe(
            result_df[result_df["Anomaly"] == 1][
                ["WindSpeed", "Power", "Residual", "HealthIndex", "AnomalyRolling", "EarlyWarning"]
            ].head(50),
            use_container_width=True
        )

        st.subheader("Early Warning Table")
        st.dataframe(
            result_df[result_df["EarlyWarning"] == True][
                ["WindSpeed", "Power", "Residual", "HealthIndex", "AnomalyRolling"]
            ].head(50),
            use_container_width=True
        )

        # ---------------- DOWNLOAD ----------------
        csv = result_df.to_csv().encode("utf-8")
        st.download_button(
            label="Download Processed Results CSV",
            data=csv,
            file_name="turbine_analysis_results.csv",
            mime="text/csv"
        )

    else:
        st.warning("Dataset is empty after preprocessing.")
else:
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #12315a 0%, #163b6d 100%);
            color: #f8fafc;
            padding: 16px 18px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.10);
            font-weight: 500;
            margin-top: 10px;">
            Upload your SCADA CSV or Excel file from the sidebar to start.
        </div>
        """,
        unsafe_allow_html=True
    )