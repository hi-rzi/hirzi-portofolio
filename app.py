import streamlit as st
import numpy as np
import pandas as pd
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
# 1. CORE GENERATOR DIFFERENTIAL RELAY ENGINE (87G)
#    Modes: GENERATOR (modern numerical dual-slope) and
#           GENERATOR_LEGACY (fixed single-slope electromechanical/solid-state, e.g. GE CFD22B4A)
# =====================================================================
class AdvancedDifferentialRelay:
    def __init__(self, mode, mva_rated, kv_rated,
                 ct_ratio_N=1.0, ct_ratio_T=1.0, ct_secondary_rating=5.0,
                 i_pickup=0.15, slope_1=15.0, i_breakpoint=1.0, slope_2=50.0, i_unrestrained=8.0,
                 convention="IEEE", ct_polarity="OPPOSITE",
                 target_amps=None):
        self.mode = mode.upper()  # 'GENERATOR' or 'GENERATOR_LEGACY'
        self.mva_rated = mva_rated
        self.kv_rated = kv_rated
        # ct_ratio_N / ct_ratio_T are entered as CT nameplate PRIMARY current (e.g. the "2000"
        # in a "2000:5" CT), matching how CTs are actually specified in the field.
        # ct_secondary_rating is the CT's rated secondary current (1 A or 5 A), which the
        # earlier version of this model silently assumed was baked into the ratio already.
        # The TRUE turns ratio used for all scaling is primary_rating / secondary_rating.
        self.ct_ratio_N = ct_ratio_N  # Neutral side CT primary rating
        self.ct_ratio_T = ct_ratio_T  # Terminal side CT primary rating
        self.ct_secondary_rating = ct_secondary_rating
        self.effective_ratio_N = (ct_ratio_N / ct_secondary_rating) if ct_secondary_rating > 0 else ct_ratio_N
        self.effective_ratio_T = (ct_ratio_T / ct_secondary_rating) if ct_secondary_rating > 0 else ct_ratio_T
        self.i_pickup = i_pickup
        self.s1 = slope_1 / 100.0
        self.i_bp = i_breakpoint
        self.s2 = slope_2 / 100.0
        self.i_unrestrained = i_unrestrained
        self.convention = convention.upper()
        self.ct_polarity = ct_polarity
        self.target_amps = target_amps

        # Rated primary current (single winding voltage class — generator has no HV/LV split)
        self.i_rated_pri = (mva_rated * 1000.0) / (math.sqrt(3) * self.kv_rated) if self.kv_rated > 0 else 1.0

        # Secondary ratings on both terminals — correctly divides by the TRUE ratio
        # (primary rating / secondary rating), not the raw nameplate primary rating alone.
        self.i_rated_sec_N = self.i_rated_pri / self.effective_ratio_N if self.effective_ratio_N > 0 else 1.0
        self.i_rated_sec_T = self.i_rated_pri / self.effective_ratio_T if self.effective_ratio_T > 0 else 1.0

        # GENERATOR_LEGACY (e.g. GE CFD22B4A-type electromechanical/solid-state relays):
        # the real-world setting sheet specifies pickup directly as a "Target and Seal-in"
        # current in SECONDARY AMPS (e.g. 0.2 A), not as a per-unit fraction. Convert that
        # into the per-unit pickup this engine works in, referenced to the neutral-side CT.
        if self.mode == "GENERATOR_LEGACY" and target_amps is not None and self.i_rated_sec_N > 0:
            self.i_pickup = target_amps / self.i_rated_sec_N
            # This relay type has no field-adjustable breakpoint, second slope, or
            # unrestrained high-set element — those simply don't exist as settings on it.
            # Force the characteristic to a single fixed-percentage slope with no upper
            # discontinuity, and disable the unrestrained element (set effectively unreachable).
            self.i_bp = 1e6
            self.s2 = self.s1
            self.i_unrestrained = 1e6

    def calculate_trip_threshold(self, i_rest_pu):
        """Calculates boundary operating current threshold.
        Dual-slope (modern numerical relays): pickup + slope1 up to breakpoint, then slope2 beyond it.
        GENERATOR_LEGACY (e.g. CFD22B4A): single fixed percentage slope, no breakpoint/second slope —
        those aren't settings this hardware has, so i_bp is forced unreachable and s2==s1 in __init__,
        making this formula collapse to a straight single-slope line for that mode automatically.
        """
        if i_rest_pu <= self.i_bp:
            return self.i_pickup + (self.s1 * i_rest_pu)
        else:
            return self.i_pickup + (self.s1 * self.i_bp) + (self.s2 * (i_rest_pu - self.i_bp))

    def evaluate_protection(self, i_primary_N, angle_N_deg, i_primary_T, angle_T_deg):
        """
        N = Neutral Side, T = Terminal Side (same winding, opposite ends — a generator
        has no HV/LV split the way a transformer does, so there is no vector-group
        phase-shift compensation needed or applied here).
        """
        # Step 1: Scale primary currents into secondary terms using the TRUE CT ratio
        i_N_sec_mag = i_primary_N / self.effective_ratio_N if self.effective_ratio_N > 0 else 0.0
        i_T_sec_mag = i_primary_T / self.effective_ratio_T if self.effective_ratio_T > 0 else 0.0

        # Step 2: Convert secondary values into per-unit base settings
        i_N_pu_mag = i_N_sec_mag / self.i_rated_sec_N if self.i_rated_sec_N > 0 else 0.0
        i_T_pu_mag = i_T_sec_mag / self.i_rated_sec_T if self.i_rated_sec_T > 0 else 0.0

        # Step 3: Complex Phasors calculation
        rad_N = math.radians(angle_N_deg)
        rad_T = math.radians(angle_T_deg)

        vec_N_pu = cmath.rect(i_N_pu_mag, rad_N)
        vec_T_pu = cmath.rect(i_T_pu_mag, rad_T)

        # Step 4: Vector Differential Operating Current (I_op)
        if self.ct_polarity == "SAME":
            # CT polarities pointing in same direction through protected zone
            vec_op = vec_T_pu + vec_N_pu
        else:
            # Traditional differential CT facing inward
            vec_op = vec_T_pu - vec_N_pu

        i_op_pu = abs(vec_op)

        # Step 5: Restraining Current Calculation
        if self.convention == "IEEE":
            i_rest_pu = (abs(vec_T_pu) + abs(vec_N_pu)) / 2.0
        else:
            i_rest_pu = abs(vec_T_pu) + abs(vec_N_pu)

        i_threshold_pu = self.calculate_trip_threshold(i_rest_pu)

        # Step 6: Main Tripping Decision Engine
        # Note: 2nd/5th harmonic inrush/overexcitation blocking is intentionally NOT modeled
        # here — that's a transformer-only concept tied to magnetizing inrush from a magnetic
        # core, which generators don't have. Generator differential (87G) instead relies on
        # CT saturation detection/supervision, which is a different mechanism.
        is_unrestrained_trip = i_op_pu >= self.i_unrestrained
        is_restrained_trip = i_op_pu > i_threshold_pu
        is_trip = is_unrestrained_trip or is_restrained_trip

        status_text = "SAFE"
        if is_unrestrained_trip:
            status_text = "UNRESTRAINED TRIP"
        elif is_restrained_trip:
            status_text = "SLOPE TRIP"

        return {
            "i_op_pu": i_op_pu,
            "i_rest_pu": i_rest_pu,
            "i_threshold_pu": i_threshold_pu,
            "is_trip": is_trip,
            "is_unrestrained": is_unrestrained_trip,
            "status": status_text,
            "i_N_pu_mag": i_N_pu_mag,
            "i_T_pu_mag": i_T_pu_mag
        }


def compute_slope_margin_pct(e):
    """Positive = % headroom remaining before a slope trip. Negative = % already past threshold."""
    if e["i_threshold_pu"] > 0:
        return ((e["i_threshold_pu"] - e["i_op_pu"]) / e["i_threshold_pu"]) * 100.0
    return 0.0


def compute_87u_margin_pct(e, relay_obj):
    """Positive = % headroom before the unrestrained high-set element operates. None if not applicable."""
    if relay_obj.i_unrestrained >= 1e5:
        return None
    if relay_obj.i_unrestrained > 0:
        return ((relay_obj.i_unrestrained - e["i_op_pu"]) / relay_obj.i_unrestrained) * 100.0
    return None


# =====================================================================
# 2. PDF SHIFT LOG REPORT GENERATOR
# =====================================================================
def generate_pdf_report(unit_name, relay_obj, evals, phases):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor("#1E3A8A"))
    story.append(Paragraph(f"Generator Differential Protection (87G) Evaluation Report - {relay_obj.mode} Mode", title_style))
    story.append(Spacer(1, 10))

    meta_text = f"<b>Date/Time:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <b>Configuration:</b> {unit_name}"
    story.append(Paragraph(meta_text, styles['Normal']))
    story.append(Spacer(1, 15))

    # Ratings & Settings Table
    story.append(Paragraph("<b>1. Generator & Relay Parameters</b>", styles['Heading2']))

    if relay_obj.mode == "GENERATOR_LEGACY":
        params_data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Generator Rating", f"{relay_obj.mva_rated} MVA", "Target/Seal-in Pickup", f"{relay_obj.target_amps} A sec." if relay_obj.target_amps is not None else "N/A"],
            ["Rated Voltage", f"{relay_obj.kv_rated} kV", "Equivalent Pickup", f"{relay_obj.i_pickup:.3f} pu"],
            ["Rated Current (Pri)", f"{relay_obj.i_rated_pri:.2f} A", "Restraint Slope (assumed)", f"{relay_obj.s1*100:.1f} %"],
            ["Neutral CT Ratio", f"{relay_obj.ct_ratio_N:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Breakpoint / 2nd Slope / 87U", "N/A - fixed by relay design"],
            ["Terminal CT Ratio", f"{relay_obj.ct_ratio_T:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Relay Type", "GE CFD22B4A (GEK-34124)"]
        ]
    else:  # GENERATOR
        params_data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Generator Rating", f"{relay_obj.mva_rated} MVA", "Minimum Pickup", f"{relay_obj.i_pickup} pu"],
            ["Rated Voltage", f"{relay_obj.kv_rated} kV", "Slope 1", f"{relay_obj.s1*100:.1f} %"],
            ["Rated Current (Pri)", f"{relay_obj.i_rated_pri:.2f} A", "Breakpoint", f"{relay_obj.i_bp} pu"],
            ["Neutral CT Ratio", f"{relay_obj.ct_ratio_N:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Slope 2", f"{relay_obj.s2*100:.1f} %"],
            ["Terminal CT Ratio", f"{relay_obj.ct_ratio_T:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Unrestrained (87U)", f"{relay_obj.i_unrestrained} pu"]
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
    results_data = [["Phase", "I_op [pu]", "I_rest [pu]", "Threshold [pu]", "Margin [%]", "Status"]]
    for p in phases:
        e = evals[p]
        margin = compute_slope_margin_pct(e)
        results_data.append([p, f"{e['i_op_pu']:.3f}", f"{e['i_rest_pu']:.3f}", f"{e['i_threshold_pu']:.3f}", f"{margin:+.1f}%", e['status']])

    t_results = Table(results_data, colWidths=[75, 80, 80, 90, 80, 135])
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
# 3. STREAMLIT WEB APP MAIN PANEL & SYSTEM MENU
# =====================================================================
st.set_page_config(page_title="Generator Differential Relay Suite", layout="wide")

st.title("⚡ Enterprise Generator Differential Protection (87G) Suite")
st.caption("Active Phase Vector Analysis, Dual-Slope Curve Engine, Scenario Testing & Shift Log Export")

# Persistent session state for the shift test log and sweep table
if "test_log" not in st.session_state:
    st.session_state.test_log = []
if "last_scenario" not in st.session_state:
    st.session_state.last_scenario = "Manual / Custom"

# TECHNICIAN / SESSION INFO
st.sidebar.header("👷 Test Session Info")
tech_name = st.sidebar.text_input("Technician / Tester Name", value="")

# MAIN NAVIGATION MENU - PICK RELAY TYPE
st.markdown("### 🎛️ Generator Relay Type Select")
mode_selection = st.radio(
    "Choose Relay Implementation:",
    ["Generator Winding (87G) - Modern Numerical", "Generator Winding (87G) - Legacy Fixed-% (GE CFD22B4A)"],
    horizontal=True
)

# Convert selection to internal mode
if "Legacy" in mode_selection:
    current_mode = "GENERATOR_LEGACY"
else:
    current_mode = "GENERATOR"

# PRESET PROFILE MANAGEMENT
PRESETS = {
    "GENERATOR": {
        "Gen Unit 7 - 846 MVA": {"mva": 846.231, "kv": 23.0, "ct_n": 20000, "ct_t": 20000, "pickup": 0.10, "s1": 15, "bp": 1.5, "s2": 60, "u87": 6.0},
        "Gen Unit 8 - 846 MVA": {"mva": 846.231, "kv": 23.0, "ct_n": 20000, "ct_t": 20000, "pickup": 0.10, "s1": 15, "bp": 1.5, "s2": 60, "u87": 6.0}
    },
    "GENERATOR_LEGACY": {
        # Real Paiton Units 7 & 8 generator differential data, from setting sheet
        # P101-17-1823.16-0001 Rev.5 and generator nameplate:
        #   kVA=846,231 / 23,000V / CT ratio 24000:5 / relay GE CFD22B4A (GEK-34124)
        #   "Target and Seal-in" pickup set to 0.2 A secondary.
        # The 10% slope value below is NOT from the setting sheet (the CFD22B4A has no
        # separate field-adjustable slope setting — it's fixed into the relay's internal
        # design) — treat it as a placeholder until confirmed against the actual CFD22B4A
        # characteristic curve in GEK-34124 (or Appendix B of the setting document).
        "Paiton Unit 7 - CFD22B4A (846 MVA)": {"mva": 846.231, "kv": 23.0, "ct_n": 24000, "ct_t": 24000, "target_amps": 0.2, "s1": 10},
        "Paiton Unit 8 - CFD22B4A (846 MVA)": {"mva": 846.231, "kv": 23.0, "ct_n": 24000, "ct_t": 24000, "target_amps": 0.2, "s1": 10}
    }
}

current_mode_presets = PRESETS[current_mode]
st.sidebar.header("📋 Equipment Presets")
selected_preset = st.sidebar.selectbox("Load Standard Profile", list(current_mode_presets.keys()))
p_data = current_mode_presets[selected_preset]


# DYNAMIC SIDEBAR CONTROLS
st.sidebar.header("1. Generator & CT Spec")
mva = st.sidebar.number_input("Generator Rating (MVA)", value=p_data["mva"], step=10.0)
kv = st.sidebar.number_input("Rated Voltage (kV)", value=p_data["kv"], step=1.0)
ct_ratio_N = st.sidebar.number_input("Neutral Side CT Rating (Primary A, e.g. 20000 in '20000:5')", value=p_data["ct_n"])
ct_ratio_T = st.sidebar.number_input("Terminal Side CT Rating (Primary A)", value=p_data["ct_t"])

ct_secondary_rating = st.sidebar.selectbox(
    "CT Secondary Rating (A)", [1.0, 5.0], index=1,
    help="The rated secondary current stamped on the CT nameplate (e.g. the '5' in '2000:5'). "
         "This is applied to both CTs and determines the true turns ratio used in all "
         "per-unit scaling — entering only the primary rating without this was a labelling bug."
)
st.sidebar.caption(
    f"Effective ratio → Neutral: **{ct_ratio_N:.0f} : {ct_secondary_rating:.0f}** "
    f"(= {ct_ratio_N/ct_secondary_rating:.1f}:1)  |  "
    f"Terminal: **{ct_ratio_T:.0f} : {ct_secondary_rating:.0f}** "
    f"(= {ct_ratio_T/ct_secondary_rating:.1f}:1)"
)

st.sidebar.header("2. Protection Characteristic")
target_amps = None
if current_mode == "GENERATOR_LEGACY":
    st.sidebar.caption(
        "ℹ️ This relay (e.g. GE CFD22B4A) has only ONE field setting — everything else "
        "below is fixed by the relay's internal design, not adjustable on site."
    )
    target_amps = st.sidebar.slider(
        "Target / Seal-in Pickup (Secondary Amps)", 0.2, 1.0, p_data["target_amps"], 0.1,
        help="The relay's actual nameplate setting range is 0.2–1.0 A secondary — this is "
             "the ONLY field-adjustable setting on this relay type."
    )
    slope_1 = st.sidebar.slider(
        "Assumed Fixed Restraint Slope (%)", 5, 30, p_data["s1"], 1,
        help="NOT a field setting — this relay's percentage-restraint slope is fixed by its "
             "internal design and isn't in the setting sheet provided. Treat this as a "
             "placeholder until confirmed against the actual CFD22B4A characteristic curve "
             "in GEK-34124."
    )
    i_pickup = 0.0  # overridden inside the relay class from target_amps for this mode
    i_bp, slope_2, i_unrestrained = 1e6, slope_1, 1e6  # no breakpoint/2nd slope/87U on this hardware
else:
    i_pickup = st.sidebar.slider("Minimum Pickup $I_{pk}$ (pu)", 0.05, 0.50, p_data["pickup"], 0.01)
    slope_1 = st.sidebar.slider("Slope 1 (%)", 5, 40, p_data["s1"], 1)
    i_bp = st.sidebar.slider("Breakpoint Knee-point $I_{bp}$ (pu)", 0.5, 4.0, p_data["bp"], 0.1)
    slope_2 = st.sidebar.slider("Slope 2 (%)", 30, 100, p_data["s2"], 5)
    i_unrestrained = st.sidebar.slider("High-Set Unrestrained $87U$ (pu)", 3.0, 15.0, p_data["u87"], 0.5)

st.sidebar.header("3. Wiring & Convention")
if current_mode == "GENERATOR_LEGACY":
    st.sidebar.caption(
        "ℹ️ This relay has no harmonic restraint capability — it's a fixed "
        "percentage-differential design with a single pickup setting only."
    )
else:
    st.sidebar.caption(
        "ℹ️ 2nd/5th harmonic blocking is not applicable to generators — "
        "generators don't produce magnetizing inrush the way transformer "
        "cores do, so this element isn't modeled here."
    )

col_conv, col_pol = st.sidebar.columns(2)
with col_conv:
    convention = st.radio("Restraint Standard", ["IEEE", "IEC"], help="IEEE: Average current. IEC: Arithmetic sum.")
with col_pol:
    ct_polarity = st.radio("Polarity Reference", ["OPPOSITE", "SAME"], help="OPPOSITE: standard facing inwards. SAME: facing identical directions.")

# Create main relay object
relay = AdvancedDifferentialRelay(
    mode=current_mode, mva_rated=mva, kv_rated=kv,
    ct_ratio_N=ct_ratio_N, ct_ratio_T=ct_ratio_T, ct_secondary_rating=ct_secondary_rating,
    i_pickup=i_pickup, slope_1=slope_1, i_breakpoint=i_bp, slope_2=slope_2, i_unrestrained=i_unrestrained,
    convention=convention, ct_polarity=ct_polarity,
    target_amps=target_amps
)

# TABS CONFIG
tab1, tab2, tab3 = st.tabs(["📊 Live Vector Simulation", "🧰 Commissioning & Injection Tool", "🗂️ Test Log & Export"])


with tab1:
    col_inputs, col_results = st.columns([1.2, 1.0])

    with col_inputs:
        st.subheader("Primary (Generator) Operating Phase Inputs")
        st.caption(
            "Enter the actual PRIMARY-side current in Amps (e.g. generator load current or "
            "fault current at the machine terminals) — the app converts this through the CT "
            "ratio and rated base automatically. You do not need to divide by the CT ratio "
            "yourself. For the actual 0–5 A (or 0–1 A) secondary current you'd inject into "
            "the physical relay during testing, see the Commissioning & Injection Tool tab."
        )

        st.info(f"Generator Nominal Rated Current: **{relay.i_rated_pri:.1f} A**")

        phases = ["Phase A", "Phase B", "Phase C"]

        # Generator: both CTs sit on the SAME winding at the same voltage (neutral end vs
        # terminal end).
        n_side_label, t_side_label = "Neutral Side (End 1)", "Terminal Side (End 2)"

        # -------------------------------------------------------------
        # QUICK SCENARIO LOADER
        # Pre-fills the phase inputs below with realistic textbook values for common
        # test/event conditions, so you don't have to hand-derive angles and magnitudes
        # for every test. Values are illustrative starting points — always compare
        # against your actual event/test record.
        # -------------------------------------------------------------
        st.markdown("#### ⚡ Quick Scenario Loader")
        scenario = st.selectbox(
            "Load a test scenario into the phase inputs below",
            ["Manual / Custom", "Normal Load (Healthy, Balanced)",
             "External / Through-Fault (Should NOT Trip)",
             "Internal Fault (Should Trip)",
             "CT Saturation on Through-Fault (False-Trip Risk)"]
        )
        apply_scenario = st.button("↪️ Apply Scenario to Phase Inputs Below")

        if apply_scenario and scenario != "Manual / Custom":
            base = relay.i_rated_pri
            for idx, ph in enumerate(phases):
                ang_N = -120.0 * idx
                ang_T = ang_N + 180.0 if ct_polarity == "OPPOSITE" else ang_N

                if scenario == "Normal Load (Healthy, Balanced)":
                    # Balanced full-load current, CTs track each other perfectly -> near-zero operate
                    i_n, i_t = base, base
                    a_n, a_t = ang_N, ang_T

                elif scenario == "External / Through-Fault (Should NOT Trip)":
                    # Large symmetric through-current on all 3 phases, both CTs track it identically
                    # -> high restraint current, but operate current stays near zero
                    i_n, i_t = base * 4.0, base * 4.0
                    a_n, a_t = ang_N, ang_T

                elif scenario == "Internal Fault (Should Trip)":
                    # Fault current feeds in mainly from the terminal side; little/no current
                    # returns via the neutral side -> large genuine operate current
                    i_n, i_t = base * 0.3, base * 5.0
                    a_n, a_t = ang_N, ang_N  # no 180° reversal -> vectors add as real operate current

                else:  # CT Saturation on Through-Fault
                    # Same large through-fault as above, but the Terminal CT partially saturates:
                    # its reported magnitude sags and its phase angle shifts -> spurious operate
                    # current that could mimic an internal fault if the slope/margin isn't adequate
                    i_n = base * 5.0
                    i_t = base * 3.5
                    a_n = ang_N
                    a_t = ang_T + 15.0

                st.session_state[f"N_i_{ph}"] = float(i_n)
                st.session_state[f"N_a_{ph}"] = float(a_n)
                st.session_state[f"T_i_{ph}"] = float(i_t)
                st.session_state[f"T_a_{ph}"] = float(a_t)

            st.session_state.last_scenario = scenario
            st.success(f"Scenario '{scenario}' applied below. Values can still be edited manually.")

        inputs = {}

        # Capture Phase inputs in tabs/expanders
        for idx, phase in enumerate(phases):
            with st.expander(f"📌 {phase} Settings", expanded=(phase == "Phase A")):
                c1, c2 = st.columns(2)

                # Default values for anti-parallel current flow under healthy conditions
                def_val = relay.i_rated_pri if phase == "Phase A" else 0.0
                def_ang_N = -120.0 * idx
                # Under opposite CT polarity, normal load will show terminal side shifted by 180 deg
                def_ang_T = def_ang_N + 180.0 if ct_polarity == "OPPOSITE" else def_ang_N

                with c1:
                    i_N = st.number_input(f"{n_side_label} Primary Amps [A]", value=def_val, key=f"N_i_{phase}")
                    a_N = st.number_input(f"{n_side_label} Angle (°)", value=def_ang_N, key=f"N_a_{phase}")
                with c2:
                    i_T = st.number_input(f"{t_side_label} Primary Amps [A]", value=def_val, key=f"T_i_{phase}")
                    a_T = st.number_input(f"{t_side_label} Angle (°)", value=def_ang_T, key=f"T_a_{phase}")

                inputs[phase] = {"i_N": i_N, "a_N": a_N, "i_T": i_T, "a_T": a_T}

        # Calculate live state evaluation
        evals = {p: relay.evaluate_protection(
            inputs[p]["i_N"], inputs[p]["a_N"],
            inputs[p]["i_T"], inputs[p]["a_T"]
        ) for p in phases}

    with col_results:
        st.subheader("Real-time Protection Verdict")

        any_trip = any(res["is_trip"] for res in evals.values())
        if any_trip:
            st.error("🚨 PROTECTIVE RELAY TRIP INITIATED!")
        else:
            st.success("✅ SYSTEM HEALTHY (Stability / Restraint Zone)")

        # Summary Metrics Table (now includes trip margin)
        has_unrestrained_element = relay.i_unrestrained < 1e5
        table_rows = []
        for p in phases:
            e = evals[p]
            margin_slope = compute_slope_margin_pct(e)
            row = {
                "Phase": p,
                "I_op [pu]": f"{e['i_op_pu']:.3f}",
                "I_rest [pu]": f"{e['i_rest_pu']:.3f}",
                "Threshold [pu]": f"{e['i_threshold_pu']:.3f}",
                "Margin to Slope Trip": f"{margin_slope:+.1f}%",
            }
            if has_unrestrained_element:
                margin_87u = compute_87u_margin_pct(e, relay)
                row["Margin to 87U"] = f"{margin_87u:+.1f}%" if margin_87u is not None else "N/A"
            row["Action Verdict"] = e["status"]
            table_rows.append(row)
        st.table(table_rows)
        st.caption("Margin is **positive** = % headroom remaining before that element operates. **Negative** = already past the threshold (tripped on that element).")

        # PDF Export Process
        pdf_bytes = generate_pdf_report(selected_preset, relay, evals, phases)
        st.download_button(
            label="📄 Export Certified Protection Audit Report",
            data=pdf_bytes,
            file_name=f"Generator_Differential_Protection_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

        # -------------------------------------------------------------
        # LOG THIS TEST TO THE SHIFT LOG
        # -------------------------------------------------------------
        st.markdown("---")
        st.write("**Save this result to the shift test log:**")
        test_note = st.text_input("Test note (optional)", key="test_note_input")
        if st.button("📌 Log This Test"):
            entry = {
                "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Technician": tech_name if tech_name else "N/A",
                "Mode": current_mode,
                "Preset": selected_preset,
                "Scenario": st.session_state.last_scenario,
                "Note": test_note,
            }
            for p in phases:
                e = evals[p]
                entry[f"{p} I_op(pu)"] = round(e["i_op_pu"], 3)
                entry[f"{p} I_rest(pu)"] = round(e["i_rest_pu"], 3)
                entry[f"{p} Threshold(pu)"] = round(e["i_threshold_pu"], 3)
                entry[f"{p} Margin(%)"] = round(compute_slope_margin_pct(e), 1)
                entry[f"{p} Status"] = e["status"]
            entry["Overall Verdict"] = "TRIP" if any_trip else "SAFE"
            st.session_state.test_log.append(entry)
            st.success("Test logged — see the 🗂️ Test Log & Export tab.")

    # -------------------------------------------------------------
    # CT / WIRING SANITY CHECKS
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔍 System Health & Wiring Sanity Checks")

    hc1, hc2 = st.columns(2)

    with hc1:
        ratio_mismatch_pct = abs(relay.effective_ratio_N - relay.effective_ratio_T) / max(relay.effective_ratio_N, relay.effective_ratio_T) * 100.0
        if ratio_mismatch_pct > 5.0:
            st.warning(
                f"⚠️ **CT Ratio Mismatch:** Neutral and Terminal CT turns ratios differ by "
                f"**{ratio_mismatch_pct:.1f}%**. This alone will produce a false differential "
                f"current even under perfectly healthy load — verify CT nameplates and wiring "
                f"before trusting any trip/no-trip result."
            )
        else:
            st.success(f"✅ CT Ratio Match OK (Neutral vs Terminal turns ratio differ by {ratio_mismatch_pct:.1f}%)")

    with hc2:
        mags = [evals[p]["i_N_pu_mag"] for p in phases]
        avg_mag = sum(mags) / len(mags) if mags else 0.0
        max_dev_pct = (max(abs(m - avg_mag) for m in mags) / avg_mag * 100.0) if avg_mag > 0 else 0.0
        if max_dev_pct > 15.0:
            st.warning(
                f"⚠️ **Phase Current Unbalance:** Neutral-side currents across A/B/C differ by "
                f"up to **{max_dev_pct:.1f}%** from their average. Expected during a real "
                f"unbalanced fault, but worth double-checking for a CT/wiring problem if this "
                f"wasn't the intended test condition."
            )
        else:
            st.success(f"✅ Phase Balance OK (Neutral-side currents within {max_dev_pct:.1f}% of each other)")

    # INTERACTIVE PLOTLY GRAPHIC
    st.subheader("📈 Dual-Slope Characteristic Trip Curve Visualization")

    max_x_val = max(6.0, max(e["i_rest_pu"] for e in evals.values()) + 1.5)
    x_axis_line = np.linspace(0, max_x_val, 400)
    y_axis_line = [relay.calculate_trip_threshold(x) for x in x_axis_line]

    fig = go.Figure()

    # Slope boundary
    fig.add_trace(go.Scatter(
        x=x_axis_line, y=y_axis_line, mode='lines', name='Trip Slopes Boundary',
        line=dict(color='#2563EB', width=3)
    ))

    # High-set boundary — only meaningful when this relay actually has an unrestrained
    # element. GENERATOR_LEGACY has i_unrestrained forced to 1e6 (unreachable) because
    # this hardware has no such setting, so skip drawing/scaling around it.
    if has_unrestrained_element:
        fig.add_trace(go.Scatter(
            x=[0, max_x_val], y=[relay.i_unrestrained, relay.i_unrestrained],
            mode='lines', name='Unrestrained High-Set (87U)',
            line=dict(color='#DC2626', width=2, dash='dash')
        ))

    # Render dynamic operating points
    phase_colors = {"Phase A": "red", "Phase B": "green", "Phase C": "blue"}
    for p in phases:
        e = evals[p]
        fig.add_trace(go.Scatter(
            x=[e["i_rest_pu"]], y=[e["i_op_pu"]],
            mode='markers+text', name=f"{p} Current Point",
            text=[f"{p}"], textposition="top center",
            marker=dict(size=14, color=phase_colors[p], symbol='x' if e["is_trip"] else 'circle'),
            hovertemplate=f"<b>{p}</b><br>I_rest: %{{x:.3f}} pu<br>I_op: %{{y:.3f}} pu<br>State: {e['status']}<extra></extra>"
        ))

    # Plot styling
    y_upper = max(relay.i_unrestrained + 2.0, max(y_axis_line) + 1.0) if has_unrestrained_element else max(y_axis_line) + 1.0
    fig.update_layout(
        title=f"{'Single-Slope' if relay.mode == 'GENERATOR_LEGACY' else 'Dual-Slope'} Restraint Plot ({relay.mode})",
        xaxis_title="Restraint Current I_rest (pu)",
        yaxis_title="Operating Current I_op (pu)",
        xaxis=dict(range=[0, max_x_val]),
        yaxis=dict(range=[0, y_upper]),
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


# SECONDARY TESTING INJECTION WORKBENCH
with tab2:
    st.subheader("🧰 Commissioning & Secondary Current Injection Assistant")
    st.write("Determine the exact test currents needed for field testing using Doble/Omicron test sets.")

    n_inj_label, t_inj_label = "Neutral Side", "Terminal Side"

    col_test1, col_test2 = st.columns(2)
    with col_test1:
        test_restraint = st.slider("Required Target Restraint Current (pu)", 0.2, 5.0, 1.2, 0.1)

    # Secondary injection values calculations
    boundary_op_curr = relay.calculate_trip_threshold(test_restraint)

    sec_N_injection = (test_restraint + boundary_op_curr / 2.0) * relay.i_rated_sec_N
    sec_T_injection = (test_restraint - boundary_op_curr / 2.0) * relay.i_rated_sec_T

    with col_test2:
        st.metric(label="Calculated Boundary Operating Current (I_op)", value=f"{boundary_op_curr:.3f} pu")

    st.markdown("---")
    st.write("### Target Relay Secondary Terminal Current Injection Parameters:")

    c_sec_a, c_sec_b = st.columns(2)
    with c_sec_a:
        st.info(f"**{n_inj_label} Secondary Injection Current ($I_N$):**\n# {sec_N_injection:.3f} Amps AC")
    with c_sec_b:
        st.info(f"**{t_inj_label} Secondary Injection Current ($I_T$):**\n# {sec_T_injection:.3f} Amps AC")

    # -------------------------------------------------------------
    # AUTO-SWEEP FULL CURVE TEST TABLE
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔁 Auto-Sweep Full Curve Test Table")
    st.write(
        "Generates a full table of boundary test points across the restraint range in one go, "
        "instead of testing one point at a time — useful for a complete commissioning verification."
    )

    sw1, sw2, sw3 = st.columns(3)
    with sw1:
        sweep_start = st.number_input("Sweep Start (pu)", value=0.2, min_value=0.0, step=0.1)
    with sw2:
        default_end = float(relay.i_unrestrained) if relay.i_unrestrained < 1e5 else 6.0
        sweep_end = st.number_input("Sweep End (pu)", value=max(6.0, default_end), step=0.5)
    with sw3:
        sweep_step = st.number_input("Sweep Step (pu)", value=0.5, min_value=0.1, step=0.1)

    if st.button("▶️ Generate Sweep Table"):
        if sweep_end <= sweep_start or sweep_step <= 0:
            st.error("Sweep End must be greater than Sweep Start, and Sweep Step must be positive.")
        else:
            sweep_points = np.arange(sweep_start, sweep_end + sweep_step / 2.0, sweep_step)
            sweep_rows = []
            for i_rest in sweep_points:
                boundary_op = relay.calculate_trip_threshold(i_rest)
                sec_n = (i_rest + boundary_op / 2.0) * relay.i_rated_sec_N
                sec_t = (i_rest - boundary_op / 2.0) * relay.i_rated_sec_T
                sweep_rows.append({
                    "I_rest (pu)": round(float(i_rest), 3),
                    "Boundary I_op (pu)": round(boundary_op, 3),
                    "Neutral Injection I_N (A)": round(sec_n, 3),
                    "Terminal Injection I_T (A)": round(sec_t, 3),
                })
            st.session_state["sweep_df"] = pd.DataFrame(sweep_rows)

    if "sweep_df" in st.session_state:
        st.dataframe(st.session_state["sweep_df"], use_container_width=True)
        csv_sweep = st.session_state["sweep_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Sweep Table as CSV",
            data=csv_sweep,
            file_name=f"87G_Sweep_Test_Table_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )


# TEST LOG & EXPORT TAB
with tab3:
    st.subheader("🗂️ Test Log & Shift Report Export")
    st.write("Every test you click **📌 Log This Test** for (in the Live Vector Simulation tab) is collected here for the shift.")

    if len(st.session_state.test_log) == 0:
        st.info("No tests logged yet. Go to the 📊 Live Vector Simulation tab, run a test, and click '📌 Log This Test'.")
    else:
        log_df = pd.DataFrame(st.session_state.test_log)
        st.dataframe(log_df, use_container_width=True)

        csv_bytes = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Test Log as CSV",
            data=csv_bytes,
            file_name=f"87G_Test_Log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

        st.markdown("---")
        if st.button("🗑️ Clear Test Log"):
            st.session_state.test_log = []
            st.rerun()
