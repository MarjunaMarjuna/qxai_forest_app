import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import textwrap

from gee_engine import fetch_live_sentinel_tile
from model import (
    load_trained_model,
    generate_qxai_heatmap,
    calculate_carbon_loss,
    process_sar_fusion,
)
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Q-XAI | Forest Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================
if "curr_img" not in st.session_state:
    st.session_state.curr_img = None

if "image_source" not in st.session_state:
    st.session_state.image_source = "Google Earth Engine (Live GPS)"

if "image_coords" not in st.session_state:
    st.session_state.image_coords = None


def render_html(content: str) -> None:
    """Render HTML safely formatted inside a Streamlit Markdown block."""
    st.markdown(textwrap.dedent(content).strip(), unsafe_allow_html=True)


# =========================================================
# HELPER: LOCATION NAME MAPPING
# =========================================================
def get_location_name(coords):
    if coords is not None:
        lat, lon = coords
        if abs(lat - 10.3250) < 0.01 and abs(lon - 76.9650) < 0.01:
            return "Valparai, Western Ghats, India"
        elif abs(lat - 11.1350) < 0.01 and abs(lon - 76.4350) < 0.01:
            return "Silent Valley National Park, Kerala"
        elif abs(lat - (-3.4653)) < 0.01 and abs(lon - (-62.2159)) < 0.01:
            return "Amazon Rainforest, Brazil"
        else:
            return f"Custom Coordinates ({lat:.4f}, {lon:.4f})"
    return "Uploaded Satellite Image"


# =========================================================
# GLOBAL UI STYLES
# =========================================================
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --forest: #245B43;
    --forest-dark: #173D2E;
    --forest-light: #EAF3EC;
    --page: #F5F8F5;
    --surface: #FFFFFF;
    --text: #1C2B23;
    --muted: #586B5F;
    --border: #DFE8E0;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--page);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(245, 248, 245, 0.96);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #263D30 !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-weight: 700;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #263D30 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] p {
    color: #263D30 !important;
    font-size: 13px;
}

/* Inputs */
.stNumberInput input,
.stTextInput input {
    background: #FFFFFF !important;
    color: #1C2B23 !important;
    -webkit-text-fill-color: #1C2B23 !important;
    border: 1px solid #CBD8CE !important;
    border-radius: 9px !important;
    min-height: 42px;
}
.stNumberInput button {
    background: #EDF4EE !important;
    color: #245B43 !important;
    border: 1px solid #D7E3D9 !important;
}
.stNumberInput button svg {
    color: #245B43 !important;
    fill: #245B43 !important;
}
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #CBD8CE !important;
    border-radius: 9px !important;
}

/* Main header */
.brand-header {
    background: linear-gradient(120deg, #173D2E 0%, #2D674A 100%);
    padding: 30px 32px;
    border-radius: 17px;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(23, 61, 46, 0.09);
}
.brand-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #D1E6D6 !important;
    margin-bottom: 12px;
}
.brand-title {
    font-size: clamp(26px, 3vw, 36px);
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.25;
    margin-bottom: 10px;
    color: #FFFFFF !important;
}
.brand-subtitle {
    font-size: 14px;
    line-height: 1.8;
    color: #E0EDE3 !important;
    max-width: 760px;
}

/* Section headings */
.section-heading {
    color: #173D2E !important;
    font-size: 21px;
    font-weight: 700;
    letter-spacing: -0.4px;
    margin: 8px 0 5px;
}
.section-caption {
    color: #586B5F !important;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 20px;
}

/* Metric cards */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #DFE8E0;
    border-radius: 14px;
    padding: 21px 18px;
    min-height: 126px;
    box-shadow: 0 2px 8px rgba(22, 46, 31, 0.025);
    transition: border-color 0.2s ease, transform 0.2s ease;
    overflow-wrap: anywhere;
}
.metric-card:hover {
    border-color: #A9C8B1;
    transform: translateY(-2px);
}
.metric-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #586B5F !important;
    margin-bottom: 14px;
    text-transform: uppercase;
    line-height: 1.6;
}
.metric-value {
    font-size: clamp(19px, 1.7vw, 26px);
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.4;
    color: #1C2B23 !important;
    overflow-wrap: anywhere;
}
.metric-note {
    color: #586B5F !important;
    font-size: 11px;
    line-height: 1.6;
    margin-top: 9px;
}
.safe { color: #23784D !important; }
.danger { color: #C3443F !important; }

/* Tabs: force all three tab labels to black */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #DFE8E0;
}
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] *,
.stTabs button[role="tab"],
.stTabs button[role="tab"] * {
    background: transparent;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    font-size: 13px;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px 9px 0 0;
    padding: 13px 16px;
}
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs button[role="tab"][aria-selected="true"] {
    background: #EAF3EC !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    border-bottom: 2px solid #245B43 !important;
}

/* HARD OVERRIDE: Streamlit tab labels (all tabs, selected and unselected) */
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] span,
div[data-testid="stTabs"] button[role="tab"] div,
div[data-testid="stTabs"] button[role="tab"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

/* Buttons */
.stButton > button {
    background: #245B43 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #245B43 !important;
    border-radius: 9px !important;
    min-height: 43px;
    font-weight: 600 !important;
    font-size: 13px;
    transition: background 0.2s ease;
}
.stButton > button p,
.stButton > button span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button:hover {
    background: #173D2E !important;
    border-color: #173D2E !important;
}

/* Custom Forest Alert Dispatcher */
.forest-alert {
    display: block;
    box-sizing: border-box;
    width: 100%;
    background: #FFF8DB !important;
    border: 1px solid #E9D88A;
    border-left: 5px solid #D4A72C;
    border-radius: 11px;
    padding: 16px 18px;
    margin: 10px 0 14px;
    color: #594817 !important;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 500;
    line-height: 1.75;
}
.forest-alert * {
    color: #594817 !important;
    -webkit-text-fill-color: #594817 !important;
}
.forest-alert-title {
    color: #49390F !important;
    -webkit-text-fill-color: #49390F !important;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 4px;
}
.forest-alert-copy {
    color: #594817 !important;
    -webkit-text-fill-color: #594817 !important;
    font-size: 13px;
    font-weight: 500;
}

/* Empty state */
.empty-state {
    background: #FFFFFF;
    border: 1px solid #DFE8E0;
    border-radius: 16px;
    padding: 38px 28px;
    text-align: center;
    margin-top: 12px;
}
.empty-title {
    font-size: 23px;
    font-weight: 800;
    color: #173D2E !important;
    margin-bottom: 10px;
}
.empty-description {
    font-size: 14px;
    color: #586B5F !important;
    line-height: 1.9;
    max-width: 650px;
    margin: auto;
}
.empty-step {
    background: #FFFFFF;
    border: 1px solid #DFE8E0;
    border-radius: 12px;
    padding: 18px;
    min-height: 100px;
}
.app-footer {
    text-align: center;
    color: #718076 !important;
    font-size: 11px;
    padding-top: 24px;
}

/* FINAL TAB LABEL COLOR OVERRIDE */
[role="tab"],
[role="tab"] *,
button[role="tab"],
button[role="tab"] *,
[data-baseweb="tab"],
[data-baseweb="tab"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}
[role="tab"][aria-selected="true"],
[role="tab"][aria-selected="true"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* FORCE BLACK TEXT ONLY FOR THE THREE MONITORING TABS */
div[data-testid="stTabs"] button[data-baseweb="tab"],
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] [data-baseweb="tab"],
div[data-testid="stTabs"] button[data-baseweb="tab"] *,
div[data-testid="stTabs"] button[role="tab"] *,
div[data-testid="stTabs"] [data-baseweb="tab"] p,
div[data-testid="stTabs"] [data-baseweb="tab"] span,
div[data-testid="stTabs"] [data-baseweb="tab"] div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

@media (max-width: 768px) {
    .block-container { padding: 1rem 0.8rem 2rem; }
    .brand-header { padding: 23px 19px; border-radius: 13px; }
    .metric-card { padding: 15px; min-height: 110px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 8px; font-size: 11px; }
}
</style>
""")


# =========================================================
# BRAND HEADER
# =========================================================
render_html("""
<div class="brand-header">
    <div class="brand-eyebrow">
        EARTH OBSERVATION &amp; ENVIRONMENTAL INTELLIGENCE
    </div>
    <div class="brand-title">Q-XAI Forest Intelligence</div>
    <div class="brand-subtitle">
        Quantum-enhanced explainable AI for satellite-based deforestation monitoring,
        forest-loss estimation and carbon impact assessment.
    </div>
</div>
""")


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def get_model():
    return load_trained_model()


model = get_model()


# =========================================================
# SIDEBAR: DATA ACQUISITION
# =========================================================
with st.sidebar:
    st.markdown("## 🛰️ Q-XAI")
    st.caption("FOREST MONITORING CONSOLE")
    st.divider()
    st.markdown("### Data acquisition")

    stream_mode = st.radio(
        "Select imagery source",
        ["Google Earth Engine (Live GPS)", "Upload Satellite Image File"],
        key="stream_mode",
    )

    if stream_mode != st.session_state.image_source:
        st.session_state.image_source = stream_mode
        st.session_state.curr_img = None
        st.session_state.image_coords = None

    if stream_mode == "Google Earth Engine (Live GPS)":
        st.markdown("#### Geographic coordinates")

        lat = st.number_input(
            "Latitude",
            value=10.1230,
            format="%.4f",
            min_value=-90.0,
            max_value=90.0,
            key="latitude",
        )
        lon = st.number_input(
            "Longitude",
            value=76.9870,
            format="%.4f",
            min_value=-180.0,
            max_value=180.0,
            key="longitude",
        )

        if st.button(
            "🛰️ Acquire Sentinel-2 Tile",
            use_container_width=True,
            key="acquire_tile",
        ):
            with st.spinner("Acquiring satellite imagery..."):
                try:
                    fetched_img = fetch_live_sentinel_tile(lat, lon)

                    if fetched_img is None:
                        st.warning("No image was returned by the GEE pipeline.")
                    else:
                        if not isinstance(fetched_img, Image.Image):
                            fetched_img = Image.fromarray(np.asarray(fetched_img))

                        st.session_state.curr_img = fetched_img.convert("RGB")
                        st.session_state.image_coords = (lat, lon)
                        st.success("Satellite tile acquired.")

                except Exception as e:
                    st.error(f"Satellite acquisition failed: {e}")

    else:
        uploaded_file = st.file_uploader(
            "Upload satellite imagery",
            type=["jpg", "jpeg", "png"],
            key="satellite_upload",
        )

        if uploaded_file is not None:
            try:
                st.session_state.curr_img = Image.open(uploaded_file).convert("RGB")
                st.session_state.image_coords = None
            except Exception as e:
                st.error(f"Unable to read image: {e}")

        if st.session_state.curr_img is not None:
            st.success("Image ready for analysis.")

    st.divider()
    st.markdown("### ⚛️ Quantum configuration")

    noise_toggle = st.checkbox(
        "Simulate hardware noise",
        value=False,
        key="noise_toggle",
    )
    zne_toggle = st.checkbox(
        "Apply ZNE error mitigation",
        value=True,
        disabled=not noise_toggle,
        key="zne_toggle",
    )

    st.divider()
    st.caption("Q-XAI · Environmental Intelligence")
    st.caption("Satellite · Explainability · Quantum")


# =========================================================
# IMAGE TRANSFORMATION
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# MAIN DASHBOARD
# =========================================================
curr_img = st.session_state.curr_img

if curr_img is not None:
    curr_img = curr_img.convert("RGB")
    curr_np = np.array(curr_img)

    # Get location name based on active coordinates
    location_title = get_location_name(st.session_state.image_coords)

    render_html(f"""
    <div class="section-heading">Monitoring Workspace — {location_title}</div>
    <div class="section-caption">
        Inspect satellite imagery, model explanations and estimated environmental impact.
    </div>
    """)

    tab1, tab2, tab3 = st.tabs([
        "🌿 Forest & Carbon",
        "📡 SAR Analysis",
        "⚛️ Quantum Circuit",
    ])

    # -----------------------------------------------------
    # TAB 1: CLASSIFICATION, HEATMAP AND CARBON IMPACT
    # -----------------------------------------------------
    with tab1:
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("#### Satellite imagery")
            st.caption("Image supplied to the analysis pipeline.")
            st.image(curr_img, use_container_width=True)

        with c2:
            st.markdown("#### Explainable AI heatmap")
            st.caption("Q-XAI Grad-CAM visualization.")

            try:
                tensor = transform(curr_img).unsqueeze(0)

                with torch.no_grad():
                    output, feature_maps = model(
                        tensor,
                        noise_enabled=noise_toggle,
                        zne_enabled=zne_toggle,
                    )
                    probs = torch.softmax(output, dim=1)
                    
                    # --- CONFIDENCE THRESHOLD TUNING (0.60) ---
                    deforested_prob = probs[0][1].item()
                    if deforested_prob > 0.60:
                        pred_class = 1
                        confidence = deforested_prob * 100
                    else:
                        pred_class = 0
                        confidence = probs[0][0].item() * 100
                    # -------------------------------------------

                overlay, deforested_ratio = generate_qxai_heatmap(
                    curr_np,
                    feature_maps,
                )
                st.image(overlay, use_container_width=True)

            except Exception as e:
                st.error(f"Model analysis failed: {e}")
                st.stop()

        st.divider()
        st.markdown("#### Environmental impact indicators")
        st.caption("Estimates calculated from the current model output.")

        try:
            hectares_lost, co2_loss = calculate_carbon_loss(
                deforested_ratio if pred_class == 1 else 0.0
            )
        except Exception as e:
            st.error(f"Carbon impact calculation failed: {e}")
            st.stop()

        status_text = "Deforested" if pred_class == 1 else "Canopy Safe"
        status_class = "danger" if pred_class == 1 else "safe"

        m1, m2, m3, m4 = st.columns(4, gap="medium")

        with m1:
            render_html(f"""
            <div class="metric-card">
                <div class="metric-label">Forest status</div>
                <div class="metric-value {status_class}">{status_text}</div>
                <div class="metric-note">Model classification</div>
            </div>
            """)

        with m2:
            render_html(f"""
            <div class="metric-card">
                <div class="metric-label">Model confidence</div>
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="metric-note">Predicted class probability</div>
            </div>
            """)

        with m3:
            render_html(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated forest loss</div>
                <div class="metric-value">{hectares_lost:.1f} Ha</div>
                <div class="metric-note">Calculated estimate</div>
            </div>
            """)

        with m4:
            render_html(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated CO₂ impact</div>
                <div class="metric-value">{co2_loss:,.0f} T</div>
                <div class="metric-note">Calculated estimate</div>
            </div>
            """)

        if pred_class == 1:
            st.markdown("")
            st.markdown("#### 🚨 Forest alert dispatcher")

            render_html("""
            <div class="forest-alert">
                <div class="forest-alert-title">
                    Deforestation pattern detected
                </div>
                <div class="forest-alert-copy">
                    The model classified this image as deforested.
                    Review the heatmap and validate the finding before
                    taking operational action.
                </div>
            </div>
            """)

            if st.button(
                "📢 Dispatch Alert to Authorities",
                use_container_width=True,
                key="dispatch_alert",
            ):
                coords = st.session_state.image_coords

                if coords is not None:
                    location_text = (
                        f"Coordinates: ({coords[0]:.4f}, {coords[1]:.4f})"
                    )
                else:
                    location_text = (
                        "GPS coordinates unavailable for uploaded image."
                    )

                st.success(
                    "Alert confirmation generated. "
                    f"{location_text} | "
                    f"Estimated impact: {co2_loss:,.0f} Tons CO₂."
                )
                st.caption(
                    "Demo confirmation only. No external authority has been contacted."
                )

    # -----------------------------------------------------
    # TAB 2: SAR VISUALIZATION
    # -----------------------------------------------------
    with tab2:
        st.markdown("#### Synthetic Aperture Radar visualization")
        st.caption("Output generated by the existing SAR processing function.")

        try:
            with st.spinner("Processing SAR visualization..."):
                sar_img = process_sar_fusion(curr_np)

            st.image(
                sar_img,
                caption="SAR texture and canopy-density visualization",
                use_container_width=True,
            )
            st.info(
                "This view displays the output of process_sar_fusion. "
                "Actual radar-based cloud penetration requires appropriate "
                "Sentinel-1 SAR input data and processing."
            )
        except Exception as e:
            st.error(f"SAR processing failed: {e}")

    # -----------------------------------------------------
    # TAB 3: QUANTUM CIRCUIT TOPOLOGY
    # -----------------------------------------------------
    with tab3:
        st.markdown("#### Parameterized quantum circuit topology")
        st.caption("Four-qubit feature map and variational circuit.")

        try:
            qc = QuantumCircuit(4)
            qc.compose(ZZFeatureMap(4, reps=1), inplace=True)
            qc.compose(RealAmplitudes(4, reps=1), inplace=True)

            circuit_col, details_col = st.columns([1.5, 1], gap="large")

            with circuit_col:
                st.markdown("##### Circuit representation")
                st.code(qc.draw(output="text"), language="text")

            with details_col:
                st.markdown("##### Circuit configuration")
                st.markdown("""
                **Feature map**

                `ZZFeatureMap`

                **Variational ansatz**

                `RealAmplitudes`

                **Qubit count**

                4 qubits

                **Repetitions**

                1 feature-map repetition and 1 ansatz repetition
                """)
                st.info(
                    "This view displays the configured Qiskit circuit. "
                    "Displaying the topology does not itself execute a quantum computation."
                )

        except Exception as e:
            st.error(f"Quantum circuit generation failed: {e}")

else:
    # -----------------------------------------------------
    # EMPTY DASHBOARD
    # -----------------------------------------------------
    render_html("""
    <div class="empty-state">
        <div style="font-size:42px; margin-bottom:16px;">🛰️</div>
        <div class="empty-title">Your forest intelligence workspace</div>
        <div class="empty-description">
            Select an imagery source from the sidebar to begin.
            Acquire a Sentinel-2 tile through Google Earth Engine or upload
            a satellite image to explore forest classification, explainability
            and estimated carbon impact.
        </div>
    </div>
    """)

    st.markdown("")
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        render_html("""
        <div class="empty-step">
            <strong style="color:#245B43;">01 · Acquire</strong>
            <p style="color:#586B5F;font-size:13px;line-height:1.7;">
                Load satellite imagery from your chosen source.
            </p>
        </div>
        """)

    with col2:
        render_html("""
        <div class="empty-step">
            <strong style="color:#245B43;">02 · Analyze</strong>
            <p style="color:#586B5F;font-size:13px;line-height:1.7;">
                Explore model predictions and explainability maps.
            </p>
        </div>
        """)

    with col3:
        render_html("""
        <div class="empty-step">
            <strong style="color:#245B43;">03 · Assess</strong>
            <p style="color:#586B5F;font-size:13px;line-height:1.7;">
                Review estimated forest loss and carbon impact.
            </p>
        </div>
        """)


# =========================================================
# FOOTER
# =========================================================
st.markdown("")
render_html("""
<div class="app-footer">
    Q-XAI FOREST INTELLIGENCE · EARTH OBSERVATION &amp; AI
</div>
""")