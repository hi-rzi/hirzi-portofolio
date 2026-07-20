import streamlit as st
import numpy as np
import plotly.graph_objects as go
import cmath
import math
import io
import datetime

# ReportLab imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# =====================================================================
# 1. CORE RELAY PROTECTION ENGINE
# =====================================================================
class Advanced87GRelay:
    def __init__(self, mva_rated, kv_rated, ct_ratio_N, ct_ratio_T, 
                 i_pickup, slope_1, i_breakpoint, slope_2, i_unrestrained,
                 harmonic_block_threshold=15.0, convention="IEEE", ct_polarity="OPPOSITE"):
        self.mva_rated = mva_rated
        self.kv_rated = kv_rated
        self.ct_ratio_N = ct_ratio_N
        self.ct_ratio_T = ct_ratio_T
        self.i_pickup = i_pickup
        self.s1 = slope_1 / 100.0
        self.i_bp = i_breakpoint
        self.s2 = slope_2 / 100.0
        self.i_unrestrained = i_unrestrained
        self.harmonic_block_threshold = harmonic_block_threshold
        self.convention = convention.upper()
        self.ct_polarity = ct_polarity

        # Rated primary and secondary currents
        self.i_rated_pri = (mva_rated * 1000.0) / (math.sqrt(3) * kv_rated) if kv_rated > 0 else 1.0
        self.i_rated_sec_N = self.i_rated_pri / ct_ratio_N if ct_ratio_N > 0 else 1.0
        self.i_rated_sec_T = self.i_rated_pri / ct_ratio_T if ct_ratio_T > 0 else 1.0

    def calculate_trip_threshold(self, i_rest_pu):
        """Calculates boundary operating current threshold for dual-slope curve."""
        if i_rest_pu <= self.i_bp:
            return self.i_pickup + (self.s1 * i_rest_pu)
        else:
            return self.i_pickup + (self.s1 * self.i_bp) + (self.s2 * (i_rest_pu - self.i_bp))

    def evaluate_phase(self, i_neutral_pri, angle_N_deg, i_terminal_pri, angle_T_deg, harmonic_2nd_pct=0.0):
        # Convert to Complex Secondary Per-Unit Currents
        i_N_sec_mag = i_neutral_pri / self.ct_ratio_N if self.ct_ratio_N > 0 else 0
        i_T_sec_mag = i_terminal_pri / self.ct_ratio_T if self.ct_ratio_T > 0 else 0

        i_N_pu_mag = i_N_sec_mag / self.i_rated_sec_N if self.i_rated_sec_N > 0 else 0
        i_T_pu_mag = i_T_sec_mag / self.i_rated_sec_T if self.i_rated_sec_T > 0 else 0

        # Complex vector currents (phasors)
        rad_N = math.radians(angle_N_deg)
        rad_T = math.radians(angle_T_deg)
        vec_N_pu = cmath.rect(i_N_pu_mag, rad_N)
        vec_T_pu = cmath.rect(i_T_pu_mag, rad_T)

        # Vector Differential Operating Current: I_op
        if self.ct_polarity == "SAME":
            vec_op = vec_T_pu + vec_N_pu
        else:
            vec_op = vec_T_pu - vec_N_pu

        i_op_pu = abs(vec_op)

        # Restraining Current Calculation
        if self.convention == "IEEE":
            i_rest_pu = (abs(vec_T_pu) + abs(vec_N_pu)) / 2.0
        else:
            i_rest_pu = abs(vec_T_pu) + abs(vec_N_pu)

        i_threshold_pu = self.calculate_trip_threshold(i_rest_pu)

        # Harmonic Restraint Check
        harmonic_blocked = harmonic_2nd_pct >= self.harmonic_block_threshold

        # Trip Condition Logic
        is_unrestrained_trip = i_op_pu >= self.i_unrestrained
        is_restrained_trip = (i_op_pu > i_threshold_pu) and not harmonic_blocked
        is_trip = is_unrestrained_trip or is_restrained_trip

        status_text = "SAFE"
        if is_unrestrained_trip:
            status_text = "UNRESTRAINED TRIP"
        elif is_restrained_trip:
            status_text = "SLOPE TRIP"
        elif harmonic_blocked and (i_op_pu > i_threshold_pu):
            status_text = "HARMONIC BLOCKED"

        return {
            "i_op_pu": i_op_pu,
            "i_rest_pu": i_rest_pu,
            "i_threshold_pu": i_threshold_pu,
            "is_trip": is_trip,
            "is_unrestrained": is_unrestrained_trip,
            "harmonic_blocked": harmonic_blocked,
            "status": status_text
        }

# =====================================================================
# 2. PDF SHIFT LOG REPORT GENERATOR
# =====================================================================
def generate_pdf_report(unit_name, relay_obj, evals, phases):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    # Title & Header (No Emojis to avoid ReportLab font crashes)
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor("#1E3A8A"))
    story.append(Paragraph("Generator Protection (87G/87U) Evaluation Report", title_style))
    story.append(Spacer(1, 10))

    meta_text = f"<b>Date/Time:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <b>Unit Configuration:</b> {unit_name}"
    story.append(Paragraph(meta_text, styles['Normal']))
    story.append(Spacer(1, 15))

    # Ratings & Settings Table
    story.append(Paragraph("<b>1. Generator & Relay Parameters</b>", styles['Heading2']))
    params_data = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Generator Rating", f"{relay_obj.mva_rated} MVA", "Minimum Pickup", f"{relay_obj.i_pickup} pu"],
        ["Rated Voltage", f"{relay_obj.kv_rated} kV", "Slope 1", f"{relay_obj.s1*100:.1f} %"],
        ["Rated Current (Pri)", f"{relay_obj.i_rated_pri:.2f} A", "Breakpoint (Knee)", f"{relay_obj.i_bp} pu"],
        ["Neutral CT Ratio", f"{relay_obj.ct_ratio_N}", "Slope 2", f"{relay_obj.s2*100:.1f} %"],
        ["Terminal CT Ratio", f"{relay_obj.ct_ratio_T}", "Unrestrained (87U)", f"{relay_obj.i_unrestrained} pu"]
    ]
    t_params = Table(params_data, colWidths=[130, 130, 130, 130])
    t_params.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#1F2937")),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_params)
    story.append(Spacer(1, 15))

    # Phase Results Table
    story.append(Paragraph("<b>2. Evaluation Results</b>", styles['Heading2']))
    results_data = [["Phase", "I_op [pu]", "I_rest [pu]", "Threshold [pu]", "Status"]]
    for p in phases:
        e = evals[p]
        results_data.append([p, f"{e['i_op_pu']:.3f}", f"{e['i_rest_pu']:.3f}", f"{e['i_threshold_pu']:.3f}", e['status']])

    t_results = Table(results_data, colWidths=[90, 90, 90, 100, 150])
    t_results.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_results)

    doc.build(story)
    buffer.seek(0)
    return buffer

# =====================================================================
# 3. STREAMLIT WEB APP DASHBOARD
# =====================================================================
st.set_page_config(page_title="Generator 87G/87U Enterprise Tool", layout="wide")

st.title("⚡ Enterprise Generator Differential Protection (87G / 87U) Simulator")
st.caption("Power Plant Relay Evaluation, Vector Analysis, 2nd Harmonic Blocking & Dual-Slope Curve Engine")

# PRESET PROFILES
PRESETS = {
    "Unit 7 - 846 MVA": {"mva": 846.0, "kv": 23.0, "ct_n": 20000, "ct_t": 20000, "pickup": 0.10, "s1": 15, "bp": 1.5, "s2": 60, "u87": 6.0},
    "Unit 8 - 846 MVA": {"mva": 846.0, "kv": 23.0, "ct_n": 20000, "ct_t": 20000, "pickup": 0.10, "s1": 15, "bp": 1.5, "s2": 60, "u87": 6.0},
    "Custom Configuration": {"mva": 500.0, "kv": 20.0, "ct_n": 16000, "ct_t": 16000, "pickup": 0.10, "s1": 15, "bp": 1.5, "s2": 60, "u87": 5.0}
}

# SIDEBAR CONTROLS
st.sidebar.header("📋 Preset Configuration")
selected_preset = st.sidebar.selectbox("Load Preset Profile", list(PRESETS.keys()))
p = PRESETS[selected_preset]

st.sidebar.header("1. Generator & CT Settings")
mva = st.sidebar.number_input("Generator Rating (MVA)", value=p["mva"], step=10.0)
kv = st.sidebar.number_input("Rated Voltage (kV)", value=p["kv"], step=0.5)
ct_ratio_N = st.sidebar.number_input("Neutral CT Ratio", value=p["ct_n"])
ct_ratio_T = st.sidebar.number_input("Terminal CT Ratio", value=p["ct_t"])

st.sidebar.header("2. Dual-Slope Relay Settings")
i_pickup = st.sidebar.slider("Minimum Pickup $I_{pk}$ (pu)", 0.05, 0.50, p["pickup"], 0.01)
slope_1 = st.sidebar.slider("Slope 1 (%)", 5, 40, p["s1"], 1)
i_bp = st.sidebar.slider("Breakpoint Knee-point $I_{bp}$ (pu)", 0.5, 4.0, p["bp"], 0.1)
slope_2 = st.sidebar.slider("Slope 2 (%)", 30, 100, p["s2"], 5)
i_unrestrained = st.sidebar.slider("High-Set Unrestrained $87U$ (pu)", 3.0, 10.0, p["u87"], 0.5)
harmonic_block_thresh = st.sidebar.slider("2nd Harmonic Blocking Threshold (%)", 10, 30, 15, 1)

col_conv, col_pol = st.sidebar.columns(2)
with col_conv:
    convention = st.radio("Restraint Conv.", ["IEEE", "IEC"])
with col_pol:
    ct_polarity = st.radio("CT Polarity", ["OPPOSITE", "SAME"], help="OPPOSITE: I_op = |I_T - I_N|. SAME: I_op = |I_T + I_N|")

# Instantiate Relay Engine
relay = Advanced87GRelay(
    mva_rated=mva, kv_rated=kv, 
    ct_ratio_N=ct_ratio_N, ct_ratio_T=ct_ratio_T,
    i_pickup=i_pickup, slope_1=slope_1, 
    i_breakpoint=i_bp, slope_2=slope_2, 
    i_unrestrained=i_unrestrained,
    harmonic_block_threshold=harmonic_block_thresh,
    convention=convention,
    ct_polarity=ct_polarity
)

# TABS LAYOUT
tab1, tab2 = st.tabs(["📊 Live Protection Simulator", "🧰 Commissioning & Injection Tool"])

with tab1:
    col_inputs, col_results = st.columns([1.1, 1.1])

    with col_inputs:
        st.subheader("Phase Vector & Current Inputs")
        st.info(f"Nominal Rated Generator Current: **{relay.i_rated_pri:.2f} A Primary**")

        phases = ["Phase A", "Phase B", "Phase C"]
        inputs = {}

        for idx, phase in enumerate(phases):
            with st.expander(f"📌 {phase} Inputs", expanded=(phase == "Phase A")):
                c1, c2, c3 = st.columns(3)
                def_val = relay.i_rated_pri if phase == "Phase A" else 0.0
                def_ang = -120.0 * idx
                with c1:
                    i_N = st.number_input(f"Neutral Amps [A]", value=def_val, key=f"N_i_{phase}")
                    a_N = st.number_input(f"Neutral Angle (°)", value=def_ang, key=f"N_a_{phase}")
                with c2:
                    i_T = st.number_input(f"Terminal Amps [A]", value=def_val, key=f"T_i_{phase}")
                    a_T = st.number_input(f"Terminal Angle (°)", value=def_ang, key=f"T_a_{phase}")
                with c3:
                    h2 = st.number_input(f"2nd Harmonic (%)", value=0.0, key=f"H_{phase}")

                inputs[phase] = {"i_N": i_N, "a_N": a_N, "i_T": i_T, "a_T": a_T, "h2": h2}

        # Evaluate protection state
        evals = {p: relay.evaluate_phase(
            inputs[p]["i_N"], inputs[p]["a_N"], 
            inputs[p]["i_T"], inputs[p]["a_T"], 
            inputs[p]["h2"]
        ) for p in phases}

    with col_results:
        st.subheader("System Status")
        
        any_trip = any(res["is_trip"] for res in evals.values())
        if any_trip:
            st.error("🚨 RELAY TRIP DETECTED! Action Required.")
        else:
            st.success("✅ SYSTEM NORMAL / RESTRAIN (No Trip)")

        # Results table
        table_rows = []
        for p in phases:
            e = evals[p]
            table_rows.append({
                "Phase": p,
                "I_op [pu]": f"{e['i_op_pu']:.3f}",
                "I_rest [pu]": f"{e['i_rest_pu']:.3f}",
                "Threshold [pu]": f"{e['i_threshold_pu']:.3f}",
                "Status": e["status"]
            })
        st.table(table_rows)

        # Download PDF Report Button
        pdf_bytes = generate_pdf_report(selected_preset, relay, evals, phases)
        st.download_button(
            label="📄 Download PDF Shift Log Report",
            data=pdf_bytes,
            file_name=f"87G_Protection_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

    # INTERACTIVE PLOTLY DUAL-SLOPE GRAPH
    st.subheader("📈 Interactive Dual-Slope Operating Characteristic (Plotly)")

    max_x = max(6.0, max(e["i_rest_pu"] for e in evals.values()) + 1.5)
    x_curve = np.linspace(0, max_x, 400)
    y_curve = [relay.calculate_trip_threshold(x) for x in x_curve]

    fig = go.Figure()

    # Slope Characteristic Line
    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve, mode='lines', name='Dual-Slope Boundary (87G)',
        line=dict(color='#2563EB', width=3)
    ))

    # Unrestrained High-Set 87U Horizontal Line
    fig.add_trace(go.Scatter(
        x=[0, max_x], y=[relay.i_unrestrained, relay.i_unrestrained],
        mode='lines', name='Unrestrained High-Set Line (87U)',
        line=dict(color='#DC2626', width=2, dash='dash')
    ))

    # Plot Operating Points for Phases
    phase_colors = {"Phase A": "red", "Phase B": "green", "Phase C": "blue"}
    for p in phases:
        e = evals[p]
        fig.add_trace(go.Scatter(
            x=[e["i_rest_pu"]], y=[e["i_op_pu"]],
            mode='markers+text', name=f"{p}",
            text=[f"{p}"], textposition="top center",
            marker=dict(size=14, color=phase_colors[p], symbol='x' if e["is_trip"] else 'circle'),
            hovertemplate=f"<b>{p}</b><br>I_rest: %{{x:.3f}} pu<br>I_op: %{{y:.3f}} pu<br>Status: {e['status']}<extra></extra>"
        ))

    # Styling layout
    fig.update_layout(
        title="87G / 87U Relay Operating Point vs Dual-Slope Characteristic",
        xaxis_title="Restraining Current I_rest (pu)",
        yaxis_title="Operating Current I_op (pu)",
        xaxis=dict(range=[0, max_x]),
        yaxis=dict(range=[0, max(relay.i_unrestrained + 2.0, max(y_curve) + 1.0)]),
        template="plotly_white",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

# COMMISSIONING & INJECTION ASSISTANT TAB
with tab2:
    st.subheader("🧰 Secondary Injection Testing Assistant")
    st.write("Use these values when testing the physical relay using an Omicron or Doble test set.")

    col1, col2 = st.columns(2)
    with col1:
        test_i_rest = st.slider("Target Restraint Current $I_{rest}$ (pu)", 0.2, 5.0, 1.0, 0.1)
    
    # Calculate required secondary currents to test boundary
    boundary_i_op = relay.calculate_trip_threshold(test_i_rest)
    
    sec_N_pickup = (test_i_rest + boundary_i_op/2.0) * relay.i_rated_sec_N
    sec_T_pickup = (test_i_rest - boundary_i_op/2.0) * relay.i_rated_sec_T

    with col2:
        st.metric(label="Theoretical Boundary Operating Current ($I_{op}$)", value=f"{boundary_i_op:.3f} pu")

    st.markdown("---")
    st.write("### Calculated Secondary Current Injection Values for Boundary Testing:")
    
    c_a, c_b = st.columns(2)
    with c_a:
        st.info(f"**Neutral CT Secondary Channel ($I_N$):**\n# {sec_N_pickup:.3f} Amps AC")
    with c_b:
        st.info(f"**Terminal CT Secondary Channel ($I_T$):**\n# {sec_T_pickup:.3f} Amps AC")
