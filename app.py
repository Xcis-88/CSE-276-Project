import streamlit as st
import pandas as pd
import numpy as np
import os

# Scikit-learn imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# SHAP and Matplotlib imports for Explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Urban Heat Mitigation and Cooling Strategy Decision Tool",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Sleek Crypto Dashboard UI Aesthetic
st.markdown("""
<style>
/* Global Dark Theme Background */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: #0b0f19 !important;
    background-image: radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                      radial-gradient(circle at 85% 85%, rgba(0, 255, 135, 0.04) 0%, transparent 40%) !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Hide Streamlit Header, Footer, and Chrome */
header[data-testid="stHeader"], footer, #MainMenu {
    display: none !important;
    visibility: hidden !important;
}

/* Edge-to-Edge Clean Padding */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 95% !important;
}

/* Neon Gradient Headers */
h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #00ff87 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.1rem !important;
}

h2 {
    color: #f8fafc !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    margin-top: 1rem !important;
}

h3, h4 {
    color: #38bdf8 !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}

/* Subheader Styling */
[data-testid="stHeadingWithActionElements"] h2, .sub-dev {
    color: #00f2fe !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 0.8rem !important;
}

/* Modern Dashboard Metric Cards */
[data-testid="stMetric"] {
    background: #131b2e !important;
    border: 1px solid rgba(0, 242, 254, 0.18) !important;
    border-radius: 16px !important;
    padding: 18px 22px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(0, 255, 135, 0.45) !important;
    box-shadow: 0 10px 30px rgba(0, 242, 254, 0.2) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    letter-spacing: -0.02em !important;
}

[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* Neon Green / Cyan Accents for Positive Trends */
[data-testid="stMetricDelta"] svg {
    fill: #00ff87 !important;
}

/* Modern Pill-Style Navigation Tabs */
div[data-baseweb="tab-list"] {
    background: #0f172a !important;
    padding: 6px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    gap: 8px !important;
    margin-bottom: 1.8rem !important;
}

button[data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 24px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: #f8fafc !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(0, 255, 135, 0.15) 100%) !important;
    color: #00f2fe !important;
    border: 1px solid rgba(0, 242, 254, 0.45) !important;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.25) !important;
}

/* Primary Action Button - Cyber Glow */
button[kind="primary"] {
    background: linear-gradient(135deg, #00f2fe 0%, #00ff87 100%) !important;
    color: #080b14 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(0, 242, 254, 0.35) !important;
    transition: all 0.25s ease !important;
}

button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(0, 255, 135, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* Inputs, Selectboxes & File Uploader */
[data-baseweb="select"], [data-baseweb="input"] {
    background-color: #131b2e !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

[data-testid="stFileUploader"] {
    background: #131b2e !important;
    border: 1px dashed rgba(0, 242, 254, 0.35) !important;
    border-radius: 16px !important;
    padding: 18px !important;
}

/* Card Containers & Expanders */
.streamlit-expanderHeader {
    background: #131b2e !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #e2e8f0 !important;
}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {
    background: #131b2e !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    overflow: hidden !important;
}

/* Custom Styled Alerts */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid rgba(0, 242, 254, 0.2) !important;
    background-color: rgba(19, 27, 46, 0.9) !important;
    backdrop-filter: blur(10px) !important;
}

/* Slider Custom Accent */
.stSlider div[data-baseweb="slider"] {
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# Application Header & Subheader
st.title("Urban Heat Mitigation and Cooling Strategy Decision Tool")
st.subheader("Developed by Kulrajveer Singh & Shubh Singh Bhadauriya")

st.markdown("""
A professional decision-support system to analyze urban microclimates, evaluate land surface 
temperatures (LST), and simulate cooling interventions using machine learning and Explainable AI.
""")

st.divider()

# Create Tabs for Clean Layout
tab_setup, tab_prediction, tab_explanation = st.tabs([
    "Data Setup",
    "Live Prediction",
    "Model Explanation"
])

# Initialize Session State Variables
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "model" not in st.session_state:
    st.session_state.model = None
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = {}
if "baseline_mean_lst" not in st.session_state:
    st.session_state.baseline_mean_lst = 35.0
if "explainer" not in st.session_state:
    st.session_state.explainer = None

# ==============================================================================
# TAB 1: DATA SETUP
# ==============================================================================
with tab_setup:
    st.header("Data Ingestion & Model Training")
    st.markdown("Import an urban thermal dataset and configure features to train the Random Forest predictive model.")

    col_upload1, col_upload2 = st.columns([2, 1])

    with col_upload1:
        uploaded_file = st.file_uploader(
            label="Upload CSV Dataset",
            type=["csv"],
            help="Upload a CSV dataset containing NDVI, Impervious Surface, and LST features."
        )

    # Local files scan
    local_csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]

    with col_upload2:
        selected_local_file = None
        if local_csv_files:
            selected_local_file = st.selectbox(
                label="Or select a local CSV file:",
                options=["-- Select File --"] + local_csv_files,
                index=0
            )

    df_loaded = None
    data_source_label = None

    if uploaded_file is not None:
        try:
            df_loaded = pd.read_csv(uploaded_file)
            data_source_label = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    elif selected_local_file and selected_local_file != "-- Select File --":
        try:
            df_loaded = pd.read_csv(selected_local_file)
            data_source_label = selected_local_file
        except Exception as e:
            st.error(f"Error reading local file `{selected_local_file}`: {e}")

    if df_loaded is not None:
        st.session_state.dataset = df_loaded

    df = st.session_state.dataset

    if df is not None:
        st.success(f"Loaded `{data_source_label or 'Dataset'}`: **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.")

        # Data Overview Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", f"{df.shape[0]:,}")
        m2.metric("Total Attributes", df.shape[1])
        m3.metric("Missing Values", int(df.isna().sum().sum()))

        # Verification: First 5 Rows Preview
        st.subheader("Dataset Verification (First 5 Rows)")
        st.dataframe(df.head(5), use_container_width=True)

        with st.expander("View Full Column Profiles & Types"):
            col_profile = pd.DataFrame({
                "Attribute": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Valid Entries": df.notnull().sum().values,
                "Null Values": df.isnull().sum().values
            })
            st.dataframe(col_profile, use_container_width=True)

        st.markdown("---")
        st.subheader("Feature & Target Configuration")

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) < 3:
            st.error("The dataset must include at least three numeric fields (NDVI, Impervious Surface, and LST).")
        else:
            def find_matching_column(candidates, search_terms):
                for candidate in candidates:
                    lowered = candidate.lower()
                    if any(term in lowered for term in search_terms):
                        return candidate
                return candidates[0] if candidates else None

            auto_ndvi = find_matching_column(numeric_columns, ["ndvi", "vegetation", "green"])
            auto_concrete = find_matching_column([c for c in numeric_columns if c != auto_ndvi], ["impervious", "concrete", "built", "ndbi", "surface"])
            auto_lst = find_matching_column([c for c in numeric_columns if c not in [auto_ndvi, auto_concrete]], ["lst", "temp", "temperature", "heat"])

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                col_ndvi = st.selectbox(
                    "Vegetation Feature (NDVI):",
                    options=numeric_columns,
                    index=numeric_columns.index(auto_ndvi) if auto_ndvi in numeric_columns else 0
                )
            with fc2:
                available_for_concrete = [c for c in numeric_columns if c != col_ndvi]
                col_concrete = st.selectbox(
                    "Concrete Feature (Impervious Surface):",
                    options=available_for_concrete,
                    index=available_for_concrete.index(auto_concrete) if auto_concrete in available_for_concrete else 0
                )
            with fc3:
                available_for_lst = [c for c in numeric_columns if c not in [col_ndvi, col_concrete]]
                col_lst = st.selectbox(
                    "Target Feature (Land Surface Temp - LST):",
                    options=available_for_lst,
                    index=available_for_lst.index(auto_lst) if auto_lst in available_for_lst else 0
                )

            st.session_state.feature_cols = {
                "ndvi": col_ndvi,
                "concrete": col_concrete,
                "lst": col_lst
            }

            # Train Model Button & Logic
            if st.button("Train Predictive Model", type="primary"):
                if not SKLEARN_AVAILABLE:
                    st.error("`scikit-learn` is required to train the model. Install via: pip install scikit-learn")
                else:
                    train_data = df[[col_ndvi, col_concrete, col_lst]].dropna()
                    if len(train_data) < 5:
                        st.error("Insufficient samples after removing missing values.")
                    else:
                        X = train_data[[col_ndvi, col_concrete]]
                        y = train_data[col_lst]
                        st.session_state.baseline_mean_lst = float(y.mean())

                        test_portion = 0.2 if len(train_data) >= 15 else 0.1
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_portion, random_state=42
                        )

                        rf_engine = RandomForestRegressor(
                            n_estimators=100,
                            max_depth=6,
                            min_samples_split=2,
                            random_state=42
                        )
                        rf_engine.fit(X_train, y_train)

                        # Evaluate
                        y_test_pred = rf_engine.predict(X_test)
                        r2 = r2_score(y_test, y_test_pred) if len(y_test) > 1 else 1.0
                        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                        mae = mean_absolute_error(y_test, y_test_pred)

                        st.session_state.model = rf_engine
                        st.session_state.model_metrics = {
                            "r2": r2,
                            "rmse": rmse,
                            "mae": mae,
                            "importances": rf_engine.feature_importances_
                        }

                        if SHAP_AVAILABLE:
                            try:
                                st.session_state.explainer = shap.TreeExplainer(rf_engine)
                            except Exception as ex:
                                st.session_state.explainer = None

                        st.success("Random Forest Regressor trained and ready for live simulation.")

        # Display Trained Metrics if Available
        if st.session_state.model_metrics is not None:
            metrics = st.session_state.model_metrics
            st.markdown("---")
            st.subheader("Model Performance Summary")
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Algorithm", "Random Forest Regressor")
            pm2.metric("R² Score", f"{metrics['r2']:.3f}")
            pm3.metric("RMSE", f"{metrics['rmse']:.2f} °C")
            pm4.metric("MAE", f"{metrics['mae']:.2f} °C")
    else:
        st.info("Upload a CSV dataset above or select a local dataset to initialize.")

# ==============================================================================
# TAB 2: LIVE PREDICTION
# ==============================================================================
with tab_prediction:
    st.header("Live Surface Temperature Prediction")
    st.markdown("Adjust proposed urban surface interventions below to generate immediate microclimate estimates:")

    # Sliders directly inside Tab 2
    slider_col1, slider_col2 = st.columns(2)

    with slider_col1:
        proposed_ndvi = st.slider(
            label="Proposed Vegetation (NDVI)",
            min_value=-1.0,
            max_value=1.0,
            value=0.45,
            step=0.01,
            key="slider_ndvi",
            help="Normalized Difference Vegetation Index: values typically range from -1.0 to 1.0 (dense canopy is > 0.4)."
        )

    with slider_col2:
        proposed_concrete = st.slider(
            label="Proposed Concrete (Impervious Surface)",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
            format="%.2f",
            key="slider_concrete",
            help="Fraction of concrete/impervious surface from 0.0 (0%) to 1.0 (100%)."
        )

    st.markdown("---")

    # Check if model is available
    if st.session_state.model is not None and st.session_state.feature_cols:
        model = st.session_state.model
        f_cols = st.session_state.feature_cols
        baseline_lst = st.session_state.baseline_mean_lst

        # Input vector for prediction
        input_vector = pd.DataFrame(
            [[proposed_ndvi, proposed_concrete]],
            columns=[f_cols["ndvi"], f_cols["concrete"]]
        )

        live_predicted_lst = float(model.predict(input_vector)[0])
        cooling_differential = baseline_lst - live_predicted_lst

        st.subheader("Live Microclimate Output")
        out1, out2, out3, out4 = st.columns(4)
        out1.metric(
            label="Predicted Surface Temp (LST)",
            value=f"{live_predicted_lst:.2f} °C",
            delta=f"{-cooling_differential:+.2f} °C vs Mean",
            delta_color="inverse"
        )
        out2.metric(
            label="Net Cooling Achieved",
            value=f"{cooling_differential:+.2f} °C",
            delta="Positive indicates cooling",
            delta_color="normal"
        )
        out3.metric(
            label="Regional Mean LST",
            value=f"{baseline_lst:.2f} °C"
        )
        out4.metric(
            label="Vegetation to Concrete Ratio",
            value=f"{(proposed_ndvi / (proposed_concrete + 1e-5)):.2f}"
        )

        st.markdown("---")
        st.subheader("Cooling Strategy Recommendation")

        if live_predicted_lst <= 31.0 or cooling_differential >= 3.0:
            st.success(f"**High Cooling Efficiency**: Proposed configuration lowers surface temperature by **{cooling_differential:.2f} °C** below average. Excellent microclimate resilience.")
        elif live_predicted_lst >= 38.0 or cooling_differential <= -2.0:
            st.error(f"**Elevated Thermal Risk**: Predicted surface temperature reaches **{live_predicted_lst:.2f} °C** (+{-cooling_differential:.2f} °C higher than average). Consider adding green buffers, high-albedo cool pavements, or tree corridors.")
        else:
            st.warning(f"**Moderate Thermal Balance**: Predicted surface temperature is **{live_predicted_lst:.2f} °C** (cooling impact: **{cooling_differential:+.2f} °C**). Increasing tree canopy cover can yield additional benefits.")

        st.markdown("---")
        st.subheader("Baseline vs. Proposed Intervention")
        comparison_data = pd.DataFrame({
            "Scenario": ["Dataset Average LST", "Proposed Intervention LST"],
            "Temperature (°C)": [baseline_lst, live_predicted_lst]
        }).set_index("Scenario")
        st.bar_chart(comparison_data)

    else:
        st.info("Please load a dataset and click 'Train Predictive Model' in the **Data Setup** tab to activate live inference.")

# ==============================================================================
# TAB 3: MODEL EXPLANATION
# ==============================================================================
with tab_explanation:
    st.header("Explainable AI: SHAP Contribution Analysis")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) decomposes the specific live prediction into 
    exact marginal temperature contributions from each land-cover feature:
    $$\\text{Predicted } LST = E[f(X)] + \\phi_{\\text{Vegetation}} + \\phi_{\\text{Concrete}}$$
    """)

    if st.session_state.model is not None and st.session_state.feature_cols:
        model = st.session_state.model
        f_cols = st.session_state.feature_cols

        # Retrieve slider values
        current_ndvi = st.session_state.get("slider_ndvi", 0.45)
        current_concrete = st.session_state.get("slider_concrete", 0.35)

        input_vector = pd.DataFrame(
            [[current_ndvi, current_concrete]],
            columns=[f_cols["ndvi"], f_cols["concrete"]]
        )
        live_pred = float(model.predict(input_vector)[0])

        if not SHAP_AVAILABLE:
            st.warning("Install the `shap` package (`pip install shap`) to generate SHAP explanations.")
        else:
            try:
                # Ensure explainer is initialized
                if st.session_state.explainer is None:
                    st.session_state.explainer = shap.TreeExplainer(model)

                explainer = st.session_state.explainer
                shap_explanation = explainer(input_vector)

                base_val = float(shap_explanation.base_values[0])
                phi_veg = float(shap_explanation.values[0][0])
                phi_conc = float(shap_explanation.values[0][1])

                # Attribution breakdown metrics
                st.subheader("Feature Contribution Breakdown")
                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Base Value E[f(X)]", f"{base_val:.2f} °C", help="Expected model value across training dataset")
                sm2.metric(
                    f"Vegetation Impact (NDVI = {current_ndvi:.2f})",
                    f"{phi_veg:+.2f} °C",
                    delta=f"{phi_veg:+.2f} °C",
                    delta_color="inverse",
                    help="Negative value indicates cooling; positive indicates warming"
                )
                sm3.metric(
                    f"Concrete Impact ({current_concrete*100:.0f}%)",
                    f"{phi_conc:+.2f} °C",
                    delta=f"{phi_conc:+.2f} °C",
                    delta_color="inverse",
                    help="Positive value indicates thermal retention; negative indicates cooling"
                )
                sm4.metric("Live Prediction", f"{live_pred:.2f} °C")

                # Decision Justification Narrative
                st.markdown("---")
                st.subheader("Decision Justification")
                veg_state = "cooling" if phi_veg <= 0 else "warming"
                conc_state = "thermal heat retention" if phi_conc >= 0 else "cooling"

                st.info(
                    f"Starting from the baseline expected temperature of **{base_val:.2f} °C**:\n\n"
                    f"• **Vegetation Cover (NDVI = {current_ndvi:.2f})** delivers **{phi_veg:+.2f} °C** of {veg_state} through canopy shading and evapotranspiration.\n"
                    f"• **Concrete Surface ({current_concrete*100:.1f}%)** contributes **{phi_conc:+.2f} °C** of {conc_state} through solar heat absorption.\n\n"
                    f"**Result**: The live predicted Land Surface Temperature is **{live_pred:.2f} °C**."
                )

                # SHAP Waterfall Plot
                st.markdown("---")
                st.subheader("Visual Attribution: SHAP Waterfall Plot")

                if MATPLOTLIB_AVAILABLE:
                    # Dark mode styled plot matching crypto dashboard
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(9, 4.2))
                    fig.patch.set_facecolor('#0b0f19')
                    ax.set_facecolor('#0b0f19')

                    try:
                        shap.plots.waterfall(shap_explanation[0], show=False)
                    except Exception:
                        explanation_obj = shap.Explanation(
                            values=np.array([phi_veg, phi_conc]),
                            base_values=base_val,
                            data=np.array([current_ndvi, current_concrete]),
                            feature_names=[f"Vegetation ({f_cols['ndvi']})", f"Concrete ({f_cols['concrete']})"]
                        )
                        shap.plots.waterfall(explanation_obj, show=False)

                    plt.title(f"SHAP Waterfall Attribution: Live Predicted LST = {live_pred:.2f} °C", fontsize=11, fontweight="bold", color="#00f2fe", pad=12)
                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                else:
                    st.warning("Install `matplotlib` (`pip install matplotlib`) to view the visual waterfall chart.")

                # Detailed Contribution Table
                with st.expander("View Detailed Attribution Table"):
                    attr_table = pd.DataFrame({
                        "Feature": [
                            f"Vegetation ({f_cols['ndvi']})",
                            f"Impervious Concrete ({f_cols['concrete']})",
                            "Base Expectation E[f(X)]"
                        ],
                        "Proposed Setting": [
                            f"{current_ndvi:.2f}",
                            f"{current_concrete:.2f} ({current_concrete*100:.0f}%)",
                            "Dataset Baseline"
                        ],
                        "Contribution (°C)": [
                            f"{phi_veg:+.3f}",
                            f"{phi_conc:+.3f}",
                            f"{base_val:.3f}"
                        ],
                        "Thermal Direction": [
                            "Cooling (Evapotranspiration)" if phi_veg <= 0 else "Elevated Heat",
                            "Heat Retention (Thermal Mass)" if phi_conc >= 0 else "Mitigated Absorption",
                            "Starting Model Baseline"
                        ]
                    })
                    st.dataframe(attr_table, use_container_width=True)

            except Exception as e:
                st.error(f"Error computing SHAP attribution: {e}")
    else:
        st.info("Train the model in the **Data Setup** tab to view Explainable AI attributions.")

# Footer
st.markdown("---")
st.caption("Urban Heat Mitigation and Cooling Strategy Decision Tool | Developed by Kulrajveer Singh & Shubh Singh Bhadauriya")
