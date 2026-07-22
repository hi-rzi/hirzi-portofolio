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
#    Modes:
#      GENERATOR         - GE G60-style dual-breakpoint numerical characteristic:
#                           flat at Pickup until Break1, Slope1 from Break1 to Break2,
#                           Slope2 beyond Break2. Settings/ranges per G60 instruction manual.
#      GENERATOR_LEGACY   - GE CFD22A/B (e.g. CFD22B4A), per GEK-34124E: a PRODUCT-RESTRAINT
#                           relay. Restraint is based on the SMALLER of the two terminal
#                           currents (not their average/sum), balancing at a fixed 10%
#                           differential up to ~rated current. No breakpoints, no field-
#                           adjustable 2nd slope, no unrestrained high-set element.
# =====================================================================
class AdvancedDifferentialRelay:
    def __init__(self, mode, mva_rated, kv_rated,
                 ct_ratio_N=1.0, ct_ratio_T=1.0, ct_secondary_rating=5.0,
                 i_pickup=0.10, slope_1=15.0, slope_2=60.0,
                 break_1=1.10, break_2=6.00,
                 i_unrestrained=None,
                 convention="IEEE", ct_polarity="OPPOSITE",
                 target_amps=None):
        self.mode = mode.upper()  # 'GENERATOR' (GE G60) or 'GENERATOR_LEGACY' (GE CFD22B4A)
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
        self.s2 = slope_2 / 100.0
        self.break_1 = break_1
        self.break_2 = break_2
        # Unrestrained/high-set element: NOT assumed present. Only modeled if the caller
        # explicitly passes a value (i.e. the user confirmed it exists in their manual and
        # enabled it in the UI). None/unset -> effectively disabled (unreachable).
        self.i_unrestrained = i_unrestrained if i_unrestrained is not None else 1e6
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
            # This relay type has no field-adjustable breakpoints, second slope, or
            # unrestrained high-set element — those simply don't exist as settings on it.
            self.s2 = self.s1
            self.i_unrestrained = 1e6

    def calculate_trip_threshold(self, i_rest_pu):
        """Calculates boundary operating current threshold.

        GENERATOR_LEGACY (GE CFD22A/B, per GEK-34124E): single fixed 10% slope starting
        from zero restraint current — no flat pickup zone, no breakpoints. Note that
        i_rest_pu here is the SMALLER of the two terminal currents (see evaluate_protection),
        not an average/sum, per the manual's product-restraint principle.

        GENERATOR (GE G60 numerical, per instruction manual):
            Ir <= Break1            -> Threshold = Pickup                        (flat zone)
            Break1 < Ir <= Break2   -> Threshold = Pickup + Slope1*(Ir - Break1)
            Ir > Break2             -> Threshold = Pickup + Slope1*(Break2-Break1)
                                                    + Slope2*(Ir - Break2)
        """
        if self.mode == "GENERATOR_LEGACY":
            return self.i_pickup + (self.s1 * i_rest_pu)

        if i_rest_pu <= self.break_1:
            return self.i_pickup
        elif i_rest_pu <= self.break_2:
            return self.i_pickup + self.s1 * (i_rest_pu - self.break_1)
        else:
            return self.i_pickup + self.s1 * (self.break_2 - self.break_1) + self.s2 * (i_rest_pu - self.break_2)

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
        # GENERATOR_LEGACY (GE CFD22A/B, per GEK-34124): this is a PRODUCT-RESTRAINT relay.
        # Operating torque is proportional to the square of the current difference; restraining
        # torque is proportional to the PRODUCT of the two terminal currents. The manual states
        # pickup balances "when the differential current is 10% of the smaller of the other two"
        # — so the restraint reference is the smaller of the two currents, not their average or
        # sum. This is fixed by the relay's physical design, not user-selectable, so the
        # IEEE/IEC convention toggle does not apply to this mode.
        if self.mode == "GENERATOR_LEGACY":
            i_rest_pu = min(abs(vec_T_pu), abs(vec_N_pu))
        elif self.convention == "IEEE":
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
            ["Rated Current (Pri)", f"{relay_obj.i_rated_pri:.2f} A", "Restraint Slope (GEK-34124E)", f"{relay_obj.s1*100:.1f} %"],
            ["Neutral CT Ratio", f"{relay_obj.ct_ratio_N:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Breakpoints / 2nd Slope / High-Set", "N/A - fixed by relay design"],
            ["Terminal CT Ratio", f"{relay_obj.ct_ratio_T:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Relay Type", "GE CFD22B4A (GEK-34124)"]
        ]
    else:  # GENERATOR (GE G60)
        has_unrestrained = relay_obj.i_unrestrained < 1e5
        params_data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Generator Rating", f"{relay_obj.mva_rated} MVA", "Pickup", f"{relay_obj.i_pickup:.3f} pu"],
            ["Rated Voltage", f"{relay_obj.kv_rated} kV", "Slope 1", f"{relay_obj.s1*100:.0f} %"],
            ["Rated Current (Pri)", f"{relay_obj.i_rated_pri:.2f} A", "Slope 2", f"{relay_obj.s2*100:.0f} %"],
            ["Neutral CT Ratio", f"{relay_obj.ct_ratio_N:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Break 1", f"{relay_obj.break_1:.2f} pu"],
            ["Terminal CT Ratio", f"{relay_obj.ct_ratio_T:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Break 2", f"{relay_obj.break_2:.2f} pu"],
            ["Relay Type", "GE G60 (Numerical)", "Unrestrained High-Set", f"{relay_obj.i_unrestrained:.2f} pu" if has_unrestrained else "Not enabled / unconfirmed"]
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
# 3. STREAMLIT WEB APP MAIN PANEL & SYSTEM MENU
# =====================================================================
# =====================================================================
# 2b. SLIDER + EXACT-VALUE INPUT HELPER
#     Every protection setting below is shown as a slider (for quick exploration) paired
#     with a plain number box (for typing the exact value from a setting sheet). The two
#     stay in sync in both directions.
# =====================================================================
def slider_with_exact_input(container, label, min_v, max_v, default, step, key, help_text=None):
    slider_key = f"{key}__slider"
    number_key = f"{key}__number"

    if key not in st.session_state:
        st.session_state[key] = default
        st.session_state[slider_key] = default
        st.session_state[number_key] = default

    def _on_slider_change():
        v = st.session_state[slider_key]
        st.session_state[key] = v
        st.session_state[number_key] = v

    def _on_number_change():
        v = st.session_state[number_key]
        v = min(max(v, min_v), max_v)  # clamp to the valid manual range
        st.session_state[key] = v
        st.session_state[slider_key] = v

    col_s, col_n = container.columns([2.4, 1])
    with col_s:
        st.slider(
            label, min_value=min_v, max_value=max_v, value=st.session_state[key],
            step=step, key=slider_key, on_change=_on_slider_change, help=help_text
        )
    with col_n:
        st.number_input(
            "Exact", min_value=min_v, max_value=max_v, value=st.session_state[key],
            step=step, key=number_key, on_change=_on_number_change, label_visibility="collapsed"
        )

    return st.session_state[key]


st.set_page_config(page_title="Generator Differential Relay Suite", layout="wide")

st.title("⚡ Enterprise Generator Differential Protection (87G) Suite")
st.caption("Active Phase Vector Analysis, GE G60 Dual-Breakpoint Curve Engine & Secondary Injection Testing")

# MAIN NAVIGATION MENU - PICK RELAY TYPE
st.markdown("### 🎛️ Generator Relay Type Select")
mode_selection = st.radio(
    "Choose Relay Implementation:",
    ["Generator Winding (87G) - GE G60 (Digital/Numerical)", "Generator Winding (87G) - Legacy Fixed-% (GE CFD22B4A)"],
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
        # Confirmed from real G60 setting sheet, Section "3.2 [87G] GENERATOR DIFFERENTIAL":
        #   Percent Diff. Pick Up: 0.06 pu | Percent Diff. Slope 1: 20%
        #   Percent Diff. Break 1: 1.15 pu | Percent Diff. Slope 2: 80% | Percent Diff. Break 2: 8 pu
        #   Line End Source: OUTG. SRC1 (Terminal side) / NETG. SRC2 (Neutral side)
        #   Percent Diff. Block: Off — no separate unrestrained/high-set element mentioned,
        #   so that stays disabled by default until confirmed otherwise.
        # CT ratio 24000:5 confirmed separately from the plant single-line diagram (87G7 zone).
        # Setting ranges/steps per GE G60 instruction manual:
        #   Pickup: 0.050-1.00 pu (step 0.01) | Slope1/Slope2: 1-100% (step 1)
        #   Break1: 1.00-1.50 pu (step 0.01) | Break2: 1.50-30.00 pu (step 0.01)
        #   Operate time: <3/4 cycle when I_diff > 5x Pickup (speed spec, not modeled numerically)
        "Gen Unit 7 - 846 MVA (G60, confirmed)": {"mva": 846.231, "kv": 23.0, "ct_n": 24000, "ct_t": 24000, "pickup": 0.06, "s1": 20, "break_1": 1.15, "s2": 80, "break_2": 8.00},
        "Gen Unit 8 - 846 MVA (G60, confirmed)": {"mva": 846.231, "kv": 23.0, "ct_n": 24000, "ct_t": 24000, "pickup": 0.06, "s1": 20, "break_1": 1.15, "s2": 80, "break_2": 8.00}
    },
    "GENERATOR_LEGACY": {
        # Real Paiton Units 7 & 8 generator differential data, from setting sheet
        # P101-17-1823.16-0001 Rev.5 and generator nameplate:
        #   kVA=846,231 / 23,000V / CT ratio 24000:5 / relay GE CFD22B4A (GEK-34124E)
        #   "Target and Seal-in" pickup set to 0.2 A secondary (factory default).
        # The 10% slope below IS confirmed by GEK-34124E's Principles of Operation section:
        # this relay is a product-restraint type whose operating/restraining torques balance
        # "when the differential current is 10% of the smaller of the other two, up to
        # approximately normal current" — this is fixed by internal design, not a field
        # setting. Above ~normal (rated) current, the manual notes the differential circuit
        # saturates, which INCREASES the effective margin beyond the flat 10% line (see
        # Figure 7) — that extra margin is not modeled here since it's shown only as a curve,
        # not a formula, in the manual.
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
i_unrestrained_value = None

if current_mode == "GENERATOR_LEGACY":
    st.sidebar.caption(
        "ℹ️ Per GEK-34124E: this relay (GE CFD22A/B, e.g. CFD22B4A) has only ONE field "
        "setting — the pickup. Everything else is fixed by the relay's internal "
        "product-restraint design, not adjustable on site."
    )
    target_amps = slider_with_exact_input(
        st.sidebar, "Target / Seal-in Pickup (Secondary Amps)", 0.1, 1.0, p_data["target_amps"], 0.05,
        key=f"{current_mode}__{selected_preset}__target_amps",
        help_text="Factory default is 0.2 A. Per GEK-34124E, it is NOT recommended to set below "
                   "0.1 A, and the rear contact may need up to ~0.25 A to close — verify the actual "
                   "closing current during commissioning."
    )
    slope_1 = slider_with_exact_input(
        st.sidebar, "Restraint Slope (%)", 5, 30, p_data["s1"], 1,
        key=f"{current_mode}__{selected_preset}__slope1",
        help_text="Confirmed by GEK-34124E's Principles of Operation: this relay balances when "
                   "the differential current is 10% of the SMALLER of the two terminal currents, "
                   "up to approximately rated current. This is fixed by the relay's internal "
                   "design, not a field setting — the slider exists here only to explore 'what if' "
                   "sensitivity; leave at 10% to match the actual hardware."
    )
    i_pickup = 0.0  # overridden inside the relay class from target_amps for this mode
    slope_2 = slope_1
    break_1, break_2 = 1e6, 1e6  # unused in legacy formula

else:  # GENERATOR - GE G60, ranges/steps per instruction manual
    st.sidebar.success(
        "✅ Pickup, Slope 1/2, and Break 1/2 below are confirmed from the actual G60 setting "
        "sheet, Section 3.2 [87G] Generator Differential. CT ratio confirmed from the plant "
        "single-line diagram. No separate unrestrained/high-set element was listed on that "
        "sheet, so it stays disabled below unless you confirm otherwise."
    )
    i_pickup = slider_with_exact_input(
        st.sidebar, "Pickup (pu)", 0.05, 1.00, p_data["pickup"], 0.01,
        key=f"{current_mode}__{selected_preset}__pickup",
        help_text="G60 manual range: 0.050 to 1.00 pu, step 0.01"
    )
    slope_1 = slider_with_exact_input(
        st.sidebar, "Slope 1 (%)", 1, 100, p_data["s1"], 1,
        key=f"{current_mode}__{selected_preset}__slope1",
        help_text="G60 manual range: 1 to 100%, step 1"
    )
    break_1 = slider_with_exact_input(
        st.sidebar, "Break 1 (pu)", 1.00, 1.50, p_data["break_1"], 0.01,
        key=f"{current_mode}__{selected_preset}__break1",
        help_text="G60 manual range: 1.00 to 1.50 pu, step 0.01. Restraint stays flat at Pickup below this point."
    )
    slope_2 = slider_with_exact_input(
        st.sidebar, "Slope 2 (%)", 1, 100, p_data["s2"], 1,
        key=f"{current_mode}__{selected_preset}__slope2",
        help_text="G60 manual range: 1 to 100%, step 1"
    )
    break_2 = slider_with_exact_input(
        st.sidebar, "Break 2 (pu)", 1.50, 30.00, p_data["break_2"], 0.01,
        key=f"{current_mode}__{selected_preset}__break2",
        help_text="G60 manual range: 1.50 to 30.00 pu, step 0.01. Slope 2 applies above this point."
    )

    st.sidebar.caption(
        "ℹ️ Per G60 manual: operate time is **<¾ cycle when I_diff > 5× Pickup**. This is a "
        "relay *speed* specification, not a separate trip threshold, so it isn't modeled "
        "numerically here."
    )

    enable_unrestrained = st.sidebar.checkbox(
        "Enable Unrestrained High-Set Element",
        value=False,
        help="Only enable this if your G60 manual confirms a separate unrestrained/high-set "
             "differential element with its own pickup setting. Left unconfirmed by default."
    )
    if enable_unrestrained:
        i_unrestrained_value = slider_with_exact_input(
            st.sidebar, "Unrestrained High-Set Pickup (pu)", 3.0, 30.0, 8.0, 0.5,
            key=f"{current_mode}__{selected_preset}__unrestrained"
        )

st.sidebar.header("3. Wiring & Convention")
if current_mode == "GENERATOR_LEGACY":
    st.sidebar.caption(
        "ℹ️ This relay has no harmonic restraint capability. It's also a **product-restraint** "
        "type (GEK-34124E) that always balances against the smaller of the two terminal "
        "currents — the IEEE/IEC toggle below doesn't apply to it and is ignored in this mode."
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
    i_pickup=i_pickup, slope_1=slope_1, slope_2=slope_2,
    break_1=break_1, break_2=break_2,
    i_unrestrained=i_unrestrained_value,
    convention=convention, ct_polarity=ct_polarity,
    target_amps=target_amps
)

# TABS CONFIG
tab1, tab2 = st.tabs(["📊 Live Vector Simulation", "🧰 Commissioning & Injection Tool"])


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

        # Summary Metrics Table
        table_rows = []
        for p in phases:
            e = evals[p]
            table_rows.append({
                "Phase": p,
                "I_op [pu]": f"{e['i_op_pu']:.3f}",
                "I_rest [pu]": f"{e['i_rest_pu']:.3f}",
                "Threshold [pu]": f"{e['i_threshold_pu']:.3f}",
                "Action Verdict": e["status"]
            })
        st.table(table_rows)

        # PDF Export Process
        pdf_bytes = generate_pdf_report(selected_preset, relay, evals, phases)
        st.download_button(
            label="📄 Export Certified Protection Audit Report",
            data=pdf_bytes,
            file_name=f"Generator_Differential_Protection_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )


    # INTERACTIVE PLOTLY GRAPHIC
    st.subheader("📈 Differential Slope Characteristic Curve")

    chart_units = st.radio(
        "Chart units", ["Per-Unit (pu)", "Secondary Amps (A)"], horizontal=True,
        help="Secondary Amps matches how commissioning test reports are usually plotted "
             "(e.g. GEK-34124 Figure 7). Conversion uses the Neutral-side rated secondary "
             "current as the base — accurate as long as both CTs share the same ratio, "
             "which they do for this unit (24000:5 on both sides)."
    )
    use_amps = chart_units == "Secondary Amps (A)"
    amps_base = relay.i_rated_sec_N  # both CTs share the same ratio for this unit

    has_unrestrained_element = relay.i_unrestrained < 1e5
    extra_range = (relay.break_2 + 1.0) if current_mode == "GENERATOR" else 0.0
    max_x_val = max(6.0, max(e["i_rest_pu"] for e in evals.values()) + 1.5, extra_range)
    x_axis_line = np.linspace(0, max_x_val, 400)
    y_axis_line = [relay.calculate_trip_threshold(x) for x in x_axis_line]

    x_plot = x_axis_line * amps_base if use_amps else x_axis_line
    y_plot = np.array(y_axis_line) * amps_base if use_amps else np.array(y_axis_line)
    unit_label = "A" if use_amps else "pu"

    fig = go.Figure()

    # Slope boundary
    fig.add_trace(go.Scatter(
        x=x_plot, y=y_plot, mode='lines', name='CAL.',
        line=dict(color='#2563EB', width=3)
    ))

    # High-set boundary — only meaningful when this relay actually has an unrestrained
    # element enabled and confirmed by the user.
    if has_unrestrained_element:
        hs_val = relay.i_unrestrained * amps_base if use_amps else relay.i_unrestrained
        fig.add_trace(go.Scatter(
            x=[0, max_x_val * amps_base if use_amps else max_x_val], y=[hs_val, hs_val],
            mode='lines', name='Unrestrained High-Set',
            line=dict(color='#DC2626', width=2, dash='dash')
        ))

    # Render dynamic operating points
    phase_colors = {"Phase A": "red", "Phase B": "green", "Phase C": "blue"}
    for p in phases:
        e = evals[p]
        px = e["i_rest_pu"] * amps_base if use_amps else e["i_rest_pu"]
        py = e["i_op_pu"] * amps_base if use_amps else e["i_op_pu"]
        fig.add_trace(go.Scatter(
            x=[px], y=[py],
            mode='markers+text', name=f"{p}",
            text=[f"{p}"], textposition="top center",
            marker=dict(size=14, color=phase_colors[p], symbol='x' if e["is_trip"] else 'circle'),
            hovertemplate=f"<b>{p}</b><br>I_rest: %{{x:.3f}} {unit_label}<br>I_op: %{{y:.3f}} {unit_label}<br>State: {e['status']}<extra></extra>"
        ))

    # Plot styling
    y_upper_pu = max(relay.i_unrestrained + 2.0, max(y_axis_line) + 1.0) if has_unrestrained_element else max(y_axis_line) + 1.0
    y_upper = y_upper_pu * amps_base if use_amps else y_upper_pu
    x_upper = max_x_val * amps_base if use_amps else max_x_val
    fig.update_layout(
        title="Differential Slope Characteristic Curve",
        xaxis_title=f"Restraint Current I_rest ({unit_label})",
        yaxis_title=f"Differential/Operating Current I_op ({unit_label})",
        xaxis=dict(range=[0, x_upper]),
        yaxis=dict(range=[0, y_upper]),
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Curve shape: {'GE G60 dual-breakpoint' if relay.mode == 'GENERATOR' else 'CFD22B4A single-slope'} "
        f"characteristic ({relay.mode})."
    )


# SECONDARY TESTING INJECTION WORKBENCH
with tab2:
    st.subheader("🧰 Commissioning & Secondary Current Injection Assistant")
    st.write(
        "Pick a target restraint current for each phase to calculate the exact secondary "
        "Amps to inject at your test set for that phase — this is your test plan, telling "
        "you what to dial in before you inject."
    )

    n_inj_label, t_inj_label = "Neutral Side", "Terminal Side"
    default_restraints = {"Phase A": 0.5, "Phase B": 2.5, "Phase C": 5.0}

    st.markdown("#### 🎯 Boundary Injection Calculator")
    phase_test_points = {}
    cols = st.columns(3)
    for p, col in zip(phases, cols):
        with col:
            st.markdown(f"**{p}**")
            r_val = slider_with_exact_input(
                st, f"{p} Target Restraint (pu)", 0.1, 30.0, default_restraints[p], 0.1,
                key=f"{current_mode}__{selected_preset}__commtest__{p}"
            )
            boundary_op = relay.calculate_trip_threshold(r_val)
            sec_N = (r_val + boundary_op / 2.0) * relay.i_rated_sec_N
            sec_T = (r_val - boundary_op / 2.0) * relay.i_rated_sec_T
            phase_test_points[p] = {"i_rest_pu": r_val, "i_op_pu": boundary_op, "sec_N": sec_N, "sec_T": sec_T}
            st.metric("Boundary I_op", f"{boundary_op:.3f} pu")
            st.caption(f"{n_inj_label} inject: **{sec_N:.3f} A**")
            st.caption(f"{t_inj_label} inject: **{sec_T:.3f} A**")

    # -------------------------------------------------------------
    # ADD TEST POINTS — real, measured results from actual injection testing.
    # These are what the chart below actually plots. Unlike the calculator
    # above (which always lands exactly on the curve by construction), these
    # can land anywhere — including off the curve — since that gap between
    # calculated and measured is the whole point of a verification test.
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📝 Add Test Points (Actual Measured Results)")
    st.caption(
        "Enter the restraint and differential currents actually read off your test set's "
        "ammeters during injection testing (in secondary Amps) — each one you add is "
        "plotted on the curve below, so you can see how real results compare to the "
        "calculated CAL. line."
    )

    if "manual_test_points" not in st.session_state:
        st.session_state.manual_test_points = []

    with st.form("add_test_point_form", clear_on_submit=True):
        tc1, tc2, tc3, tc4 = st.columns([1, 1, 1, 1.4])
        with tc1:
            tp_phase = st.selectbox("Phase", ["Phase A", "Phase B", "Phase C", "Other"])
        with tc2:
            tp_restraint = st.number_input("Restraint Current (A)", min_value=0.0, value=1.0, step=0.1)
        with tc3:
            tp_diff = st.number_input("Measured Diff. Current (A)", min_value=0.0, value=0.3, step=0.05)
        with tc4:
            tp_label = st.text_input("Label (optional)", value="")
        submitted = st.form_submit_button("➕ Add Test Point")
        if submitted:
            st.session_state.manual_test_points.append({
                "Phase": tp_phase,
                "Restraint (A)": round(tp_restraint, 3),
                "Measured Diff (A)": round(tp_diff, 3),
                "Label": tp_label
            })

    if st.session_state.manual_test_points:
        tp_df = pd.DataFrame(st.session_state.manual_test_points)
        st.dataframe(tp_df, use_container_width=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            remove_idx = st.number_input(
                "Row # to remove (0-indexed)", min_value=0,
                max_value=max(len(st.session_state.manual_test_points) - 1, 0),
                value=0, step=1
            )
            if st.button("🗑️ Remove Row"):
                st.session_state.manual_test_points.pop(int(remove_idx))
                st.rerun()
        with rc2:
            if st.button("🗑️ Clear All Test Points"):
                st.session_state.manual_test_points = []
                st.rerun()
    else:
        st.info("No test points added yet — add some above to see them plotted below.")

    # -------------------------------------------------------------
    # DIFFERENTIAL SLOPE CHARACTERISTIC CURVE — mirrors the commissioning test
    # report format: a smooth calculated ("CAL.") curve with your actual test
    # points overlaid on it. Has its own units toggle (pu / Secondary Amps),
    # independent of the Live Vector Simulation tab's chart.
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📈 Differential Slope Characteristic Curve")

    comm_chart_units = st.radio(
        "Chart units", ["Per-Unit (pu)", "Secondary Amps (A)"], horizontal=True,
        key="comm_chart_units",
        help="Secondary Amps matches how commissioning test reports are usually plotted "
             "(e.g. GEK-34124 Figure 7). Conversion uses the Neutral-side rated secondary "
             "current as the base — accurate as long as both CTs share the same ratio, "
             "which they do for this unit (24000:5 on both sides)."
    )
    use_amps_comm = comm_chart_units == "Secondary Amps (A)"
    unit_label_comm = "A" if use_amps_comm else "pu"

    manual_restraints_pu = [tp["Restraint (A)"] / amps_base for tp in st.session_state.manual_test_points]
    all_restraints_pu = [pt["i_rest_pu"] for pt in phase_test_points.values()] + manual_restraints_pu
    max_restraint = max(all_restraints_pu) if all_restraints_pu else 6.0

    curve_x_pu = np.linspace(0, max_restraint * 1.2 + 0.5, 300)
    curve_y_pu = [relay.calculate_trip_threshold(x) for x in curve_x_pu]
    curve_x = curve_x_pu * amps_base if use_amps_comm else curve_x_pu
    curve_y = np.array(curve_y_pu) * amps_base if use_amps_comm else np.array(curve_y_pu)

    sweep_fig = go.Figure()
    sweep_fig.add_trace(go.Scatter(
        x=curve_x, y=curve_y, mode="lines", name="CAL.",
        line=dict(color="#2E8B57", width=3)
    ))

    tp_marker_colors = {"Phase A": "#D63384", "Phase B": "#6C757D", "Phase C": "#1E3A8A", "Other": "#F59E0B"}
    tp_marker_symbols = {"Phase A": "square", "Phase B": "triangle-up", "Phase C": "square", "Other": "diamond"}

    for tp in st.session_state.manual_test_points:
        r_amps = tp["Restraint (A)"]
        d_amps = tp["Measured Diff (A)"]
        px = r_amps if use_amps_comm else r_amps / amps_base
        py = d_amps if use_amps_comm else d_amps / amps_base
        trace_name = tp["Phase"] + (f' ({tp["Label"]})' if tp["Label"] else "")
        sweep_fig.add_trace(go.Scatter(
            x=[px], y=[py], mode="markers", name=trace_name,
            marker=dict(size=13, color=tp_marker_colors.get(tp["Phase"], "#F59E0B"),
                        symbol=tp_marker_symbols.get(tp["Phase"], "diamond")),
            hovertemplate=f"<b>{tp['Phase']}</b><br>Restraint: %{{x:.3f}} {unit_label_comm}<br>Measured Diff: %{{y:.3f}} {unit_label_comm}<extra></extra>"
        ))

    sweep_fig.update_layout(
        title="Differential Slope Characteristic Curve",
        xaxis_title=f"Restraint Current ({unit_label_comm})",
        yaxis_title=f"Diff. Current ({unit_label_comm})",
        template="plotly_white",
        height=450
    )

    png_filename = f"87G_Differential_Slope_Curve_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    st.plotly_chart(
        sweep_fig, use_container_width=True,
        config={"toImageButtonOptions": {"format": "png", "filename": png_filename, "scale": 3}}
    )
    st.caption(
        "📷 To save this chart as an image: hover over the top-right of the chart and "
        "click the camera icon — it downloads a PNG directly from your browser, no extra "
        "software needed."
    )

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
        if current_mode == "GENERATOR":
            default_end = float(relay.break_2) + 2.0
        else:
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
