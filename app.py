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
# 1. CORE MULTI-EQUIPMENT DIFFERENTIAL RELAY ENGINE (87G / 87T / 87L)
# =====================================================================
class AdvancedDifferentialRelay:
    def __init__(self, mode, mva_rated, kv_rated_pri, kv_rated_sec=None,
                 ct_ratio_N=1.0, ct_ratio_T=1.0, ct_secondary_rating=5.0,
                 i_pickup=0.15, slope_1=15.0, i_breakpoint=1.0, slope_2=50.0, i_unrestrained=8.0,
                 harmonic_block_threshold=15.0, harmonic_5th_threshold=35.0, 
                 convention="IEEE", ct_polarity="OPPOSITE",
                 vector_group="Yy0", line_length_km=0.0, charging_current_a_per_km=0.0,
                 target_amps=None):
        self.mode = mode.upper() # 'GENERATOR', 'GENERATOR_LEGACY', 'TRANSFORMER', 'LINE'
        self.mva_rated = mva_rated
        self.kv_rated_pri = kv_rated_pri
        self.kv_rated_sec = kv_rated_sec if kv_rated_sec else kv_rated_pri
        # ct_ratio_N / ct_ratio_T are entered as CT nameplate PRIMARY current (e.g. the "2000"
        # in a "2000:5" CT), matching how CTs are actually specified in the field.
        # ct_secondary_rating is the CT's rated secondary current (1 A or 5 A), which the
        # earlier version of this model silently assumed was baked into the ratio already.
        # The TRUE turns ratio used for all scaling is primary_rating / secondary_rating.
        self.ct_ratio_N = ct_ratio_N  # In Line mode, this represents End 1 (Local) CT primary rating
        self.ct_ratio_T = ct_ratio_T  # In Line mode, this represents End 2 (Remote) CT primary rating
        self.ct_secondary_rating = ct_secondary_rating
        self.effective_ratio_N = (ct_ratio_N / ct_secondary_rating) if ct_secondary_rating > 0 else ct_ratio_N
        self.effective_ratio_T = (ct_ratio_T / ct_secondary_rating) if ct_secondary_rating > 0 else ct_ratio_T
        self.i_pickup = i_pickup
        self.s1 = slope_1 / 100.0
        self.i_bp = i_breakpoint
        self.s2 = slope_2 / 100.0
        self.i_unrestrained = i_unrestrained
        self.harmonic_block_threshold = harmonic_block_threshold
        self.harmonic_5th_threshold = harmonic_5th_threshold
        self.convention = convention.upper()
        self.ct_polarity = ct_polarity
        self.vector_group = vector_group
        self.line_length_km = line_length_km
        self.charging_current_a_per_km = charging_current_a_per_km
        self.target_amps = target_amps

        # 1. Base Currents Calculations
        self.i_rated_pri_H = (mva_rated * 1000.0) / (math.sqrt(3) * self.kv_rated_pri) if self.kv_rated_pri > 0 else 1.0
        self.i_rated_pri_L = (mva_rated * 1000.0) / (math.sqrt(3) * self.kv_rated_sec) if self.kv_rated_sec > 0 else 1.0

        # Secondary ratings on both terminals — now correctly divides by the TRUE ratio
        # (primary rating / secondary rating), not the raw nameplate primary rating alone.
        self.i_rated_sec_N = self.i_rated_pri_H / self.effective_ratio_N if self.effective_ratio_N > 0 else 1.0
        self.i_rated_sec_T = self.i_rated_pri_L / self.effective_ratio_T if self.effective_ratio_T > 0 else 1.0

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

    def evaluate_protection(self, i_primary_N, angle_N_deg, i_primary_T, angle_T_deg, 
                            harmonic_2nd_pct=0.0, harmonic_5th_pct=0.0):
        """
        In Generator/Transformer Mode:
            N = Neutral Side (or Primary/HV), T = Terminal Side (or Secondary/LV)
        In GENERATOR_LEGACY Mode (e.g. GE CFD22B4A):
            Same N/T meaning as Generator mode; single fixed-% slope, no breakpoint/87U/harmonics.
        In Line Mode:
            N = End 1 (Local), T = End 2 (Remote)
        """
        # Step 1: Scale primary currents into secondary terms using the TRUE CT ratio
        i_N_sec_mag = i_primary_N / self.effective_ratio_N if self.effective_ratio_N > 0 else 0.0
        i_T_sec_mag = i_primary_T / self.effective_ratio_T if self.effective_ratio_T > 0 else 0.0

        # Step 2: Convert secondary values into per-unit base settings
        i_N_pu_mag = i_N_sec_mag / self.i_rated_sec_N if self.i_rated_sec_N > 0 else 0.0
        i_T_pu_mag = i_T_sec_mag / self.i_rated_sec_T if self.i_rated_sec_T > 0 else 0.0

        # Step 3: Vector Group Phase Shift Compensation (For Transformers)
        compensated_angle_T_deg = angle_T_deg
        if self.mode == "TRANSFORMER":
            if self.vector_group == "Dyn11":
                # Dyn11 has secondary currents leading by 30 degrees compared to primary.
                # To align vectors, shift secondary angle backwards by 30 degrees.
                compensated_angle_T_deg -= 30.0
            elif self.vector_group == "Dyn1":
                # Dyn1 lags by 30 degrees. Compensate by adding 30 degrees.
                compensated_angle_T_deg += 30.0

        # Step 4: Complex Phasors calculation
        rad_N = math.radians(angle_N_deg)
        rad_T = math.radians(compensated_angle_T_deg)
        
        vec_N_pu = cmath.rect(i_N_pu_mag, rad_N)
        vec_T_pu = cmath.rect(i_T_pu_mag, rad_T)

        # Step 5: Vector Differential Operating Current (I_op)
        if self.ct_polarity == "SAME":
            # CT polarities pointing in same direction through protected zone
            vec_op = vec_T_pu + vec_N_pu
        else:
            # Traditional differential CT facing inward
            vec_op = vec_T_pu - vec_N_pu

        # Step 6: Line Capacitive Charging Current Compensation (For Lines)
        if self.mode == "LINE" and self.line_length_km > 0 and self.charging_current_a_per_km > 0:
            total_charging_amps = self.charging_current_a_per_km * self.line_length_km
            # Convert to secondary and then to p.u. (referenced to End 1 Local Base)
            charging_sec = total_charging_amps / self.effective_ratio_N if self.effective_ratio_N > 0 else 0.0
            charging_pu = charging_sec / self.i_rated_sec_N
            
            # Charging current acts as a continuous reactive fake differential current (+90 deg shift)
            vec_charging_pu = cmath.rect(charging_pu, math.radians(90.0))
            
            # Compensate vector difference by subtracting charging current vector
            vec_op = vec_op - vec_charging_pu

        i_op_pu = abs(vec_op)

        # Step 7: Restraining Current Calculation
        if self.convention == "IEEE":
            i_rest_pu = (abs(vec_T_pu) + abs(vec_N_pu)) / 2.0
        else:
            i_rest_pu = abs(vec_T_pu) + abs(vec_N_pu)

        i_threshold_pu = self.calculate_trip_threshold(i_rest_pu)

        # Step 8: Harmonic Restraint check
        # 2nd/5th harmonic blocking is a TRANSFORMER-ONLY concept: it exists to distinguish
        # magnetizing inrush (rich in 2nd harmonic) and overexcitation (rich in 5th harmonic)
        # from genuine internal faults. Generators do not have a magnetic core that produces
        # inrush the way a transformer does, so gating a generator's trip decision on these
        # harmonics is not physically justified and would only mask real internal faults.
        # Generator differential (87G) schemes instead rely on CT saturation detection /
        # supervision, which is a different mechanism (see CT saturation modeling).
        harmonic_2nd_blocked = (self.mode == "TRANSFORMER") and (harmonic_2nd_pct >= self.harmonic_block_threshold)
        harmonic_5th_blocked = (self.mode == "TRANSFORMER") and (harmonic_5th_pct >= self.harmonic_5th_threshold)
        is_blocked = harmonic_2nd_blocked or harmonic_5th_blocked

        # Step 9: Main Tripping Decision Engine
        is_unrestrained_trip = i_op_pu >= self.i_unrestrained
        is_restrained_trip = (i_op_pu > i_threshold_pu) and not is_blocked
        is_trip = is_unrestrained_trip or is_restrained_trip

        # Output Text Building
        status_text = "SAFE"
        if is_unrestrained_trip:
            status_text = "UNRESTRAINED TRIP"
        elif is_restrained_trip:
            status_text = "SLOPE TRIP"
        elif is_blocked and (i_op_pu > i_threshold_pu):
            if harmonic_2nd_blocked and harmonic_5th_blocked:
                status_text = "BLOCKED (2nd & 5th Harmonics)"
            elif harmonic_2nd_blocked:
                status_text = "BLOCKED (Inrush / 2nd Harmonic)"
            else:
                status_text = "BLOCKED (Overexcitation / 5th Harmonic)"

        return {
            "i_op_pu": i_op_pu,
            "i_rest_pu": i_rest_pu,
            "i_threshold_pu": i_threshold_pu,
            "is_trip": is_trip,
            "is_unrestrained": is_unrestrained_trip,
            "harmonic_blocked": is_blocked,
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

    # Clearer heading without emojis to guarantee rendering compatibility
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor("#1E3A8A"))
    story.append(Paragraph(f"Differential Protection System Evaluation Report - {relay_obj.mode} Mode", title_style))
    story.append(Spacer(1, 10))

    meta_text = f"<b>Date/Time:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <b>Configuration:</b> {unit_name}"
    story.append(Paragraph(meta_text, styles['Normal']))
    story.append(Spacer(1, 15))

    # Ratings & Settings Table
    story.append(Paragraph("<b>1. Technical System Parameters</b>", styles['Heading2']))
    
    if relay_obj.mode == "TRANSFORMER":
        params_data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Transformer Rating", f"{relay_obj.mva_rated} MVA", "Minimum Pickup", f"{relay_obj.i_pickup} pu"],
            ["Rated Voltage HV", f"{relay_obj.kv_rated_pri} kV", "Slope 1", f"{relay_obj.s1*100:.1f} %"],
            ["Rated Voltage LV", f"{relay_obj.kv_rated_sec} kV", "Breakpoint", f"{relay_obj.i_bp} pu"],
            ["Vector Group", f"{relay_obj.vector_group}", "Slope 2", f"{relay_obj.s2*100:.1f} %"],
            ["HV / LV CT Ratios", f"{relay_obj.ct_ratio_N:.0f}:{relay_obj.ct_secondary_rating:.0f} / {relay_obj.ct_ratio_T:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Unrestrained (87U)", f"{relay_obj.i_unrestrained} pu"]
        ]
    elif relay_obj.mode == "LINE":
        params_data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["System Power Rating", f"{relay_obj.mva_rated} MVA", "Minimum Pickup", f"{relay_obj.i_pickup} pu"],
            ["System Voltage", f"{relay_obj.kv_rated_pri} kV", "Slope 1", f"{relay_obj.s1*100:.1f} %"],
            ["Line Length", f"{relay_obj.line_length_km} km", "Breakpoint", f"{relay_obj.i_bp} pu"],
            ["Charging Current rate", f"{relay_obj.charging_current_a_per_km} A/km", "Slope 2", f"{relay_obj.s2*100:.1f} %"],
            ["Local / Remote CT", f"{relay_obj.ct_ratio_N:.0f}:{relay_obj.ct_secondary_rating:.0f} / {relay_obj.ct_ratio_T:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Unrestrained (87U)", f"{relay_obj.i_unrestrained} pu"]
        ]
    elif relay_obj.mode == "GENERATOR_LEGACY":
        params_data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Generator Rating", f"{relay_obj.mva_rated} MVA", "Target/Seal-in Pickup", f"{relay_obj.target_amps} A sec." if relay_obj.target_amps is not None else "N/A"],
            ["Rated Voltage", f"{relay_obj.kv_rated_pri} kV", "Equivalent Pickup", f"{relay_obj.i_pickup:.3f} pu"],
            ["Rated Current (Pri)", f"{relay_obj.i_rated_pri_H:.2f} A", "Restraint Slope (assumed)", f"{relay_obj.s1*100:.1f} %"],
            ["Neutral CT Ratio", f"{relay_obj.ct_ratio_N:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Breakpoint / 2nd Slope / 87U", "N/A - fixed by relay design"],
            ["Terminal CT Ratio", f"{relay_obj.ct_ratio_T:.0f}:{relay_obj.ct_secondary_rating:.0f}", "Relay Type", "GE CFD22B4A (GEK-34124)"]
        ]
    else:  # GENERATOR
        params_data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Generator Rating", f"{relay_obj.mva_rated} MVA", "Minimum Pickup", f"{relay_obj.i_pickup} pu"],
            ["Rated Voltage", f"{relay_obj.kv_rated_pri} kV", "Slope 1", f"{relay_obj.s1*100:.1f} %"],
            ["Rated Current (Pri)", f"{relay_obj.i_rated_pri_H:.2f} A", "Breakpoint", f"{relay_obj.i_bp} pu"],
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
st.set_page_config(page_title="Differential Relay Suite", layout="wide")

st.title("⚡ Enterprise Multi-Equipment Differential Protection Suite")
st.caption("Active Phase Vector Analysis, Complex Charging Current, Vector Group Phase Matching & Harmonic Block Control")

# MAIN NAVIGATION MENU - PICK EQUIPMENT
st.markdown("### 🎛️ Equipment Protection Type Select")
mode_selection = st.radio(
    "Choose Protected System Element:",
    ["Generator Winding (87G) - Modern Numerical", "Generator Winding (87G) - Legacy Fixed-% (GE CFD22B4A)",
     "Power Transformer (87T)", "High Voltage Transmission Line (87L)"],
    horizontal=True
)

# Convert selection to internal mode
if "Legacy" in mode_selection:
    current_mode = "GENERATOR_LEGACY"
elif "Generator" in mode_selection:
    current_mode = "GENERATOR"
elif "Transformer" in mode_selection:
    current_mode = "TRANSFORMER"
else:
    current_mode = "LINE"

# PRESET PROFILE MANAGEMENT
PRESETS = {
    "GENERATOR": {
        "Gen Unit 7 - 846 MVA": {"mva": 846.231, "kv_pri": 23.0, "kv_sec": 23.0, "ct_n": 20000, "ct_t": 20000, "pickup": 0.10, "s1": 15, "bp": 1.5, "s2": 60, "u87": 6.0},
        "Gen Unit 8 - 846 MVA": {"mva": 846.231, "kv_pri": 23.0, "kv_sec": 23.0, "ct_n": 20000, "ct_t": 20000, "pickup": 0.10, "s1": 15, "bp": 1.5, "s2": 60, "u87": 6.0}
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
        "Paiton Unit 7 - CFD22B4A (846 MVA)": {"mva": 846.231, "kv_pri": 23.0, "kv_sec": 23.0, "ct_n": 24000, "ct_t": 24000, "target_amps": 0.2, "s1": 10},
        "Paiton Unit 8 - CFD22B4A (846 MVA)": {"mva": 846.231, "kv_pri": 23.0, "kv_sec": 23.0, "ct_n": 24000, "ct_t": 24000, "target_amps": 0.2, "s1": 10}
    },
    "TRANSFORMER": {
        "Main Step-Up 873 MVA": {"mva": 873.6, "kv_pri": 23.0, "kv_sec": 500.0, "ct_n": 25000, "ct_t": 1200, "pickup": 0.20, "s1": 25, "bp": 1.2, "s2": 60, "u87": 8.0},
        "Auxiliary Unit 112 MVA": {"mva": 112.0, "kv_pri": 23.0, "kv_sec": 13.8, "ct_n": 3000, "ct_t": 5000, "pickup": 0.25, "s1": 30, "bp": 1.0, "s2": 70, "u87": 10.0}
    },
    "LINE": {
        "500kV Line": {"mva": 400.0, "kv_pri": 550.0, "kv_sec": 550.0, "ct_n": 2000, "ct_t": 2000, "pickup": 0.20, "s1": 20, "bp": 1.5, "s2": 50, "u87": 6.0}
    }
}

current_mode_presets = PRESETS[current_mode]
st.sidebar.header("📋 Equipment Presets")
selected_preset = st.sidebar.selectbox("Load Standard Profile", list(current_mode_presets.keys()))
p_data = current_mode_presets[selected_preset]


# DYNAMIC SIDEBAR CONTROLS BY EQUIPMENT TYPE
st.sidebar.header("1. Electrical Asset Spec")
mva = st.sidebar.number_input("Rating Capacity (MVA)", value=p_data["mva"], step=10.0)

if current_mode == "TRANSFORMER":
    kv_pri = st.sidebar.number_input("Primary Winding (kV)", value=p_data["kv_pri"], step=1.0)
    kv_sec = st.sidebar.number_input("Secondary Winding (kV)", value=p_data["kv_sec"], step=1.0)
    ct_ratio_N = st.sidebar.number_input("Primary Side CT Rating (Primary A, e.g. 2000 in '2000:5')", value=p_data["ct_n"])
    ct_ratio_T = st.sidebar.number_input("Secondary Side CT Rating (Primary A)", value=p_data["ct_t"])
    vector_group = st.sidebar.selectbox("Vector Transformer Group Shift", ["Yy0", "Dyn11", "Dyn1"], help="Compensates for delta-star physical vector shifts")
else:
    kv_pri = st.sidebar.number_input("System Rated Voltage (kV)", value=p_data["kv_pri"], step=1.0)
    kv_sec = kv_pri
    vector_group = "Yy0"
    
    if current_mode == "LINE":
        ct_ratio_N = st.sidebar.number_input("Local Terminal (End 1) CT Rating (Primary A)", value=p_data["ct_n"])
        ct_ratio_T = st.sidebar.number_input("Remote Terminal (End 2) CT Rating (Primary A)", value=p_data["ct_t"])
    else: # GENERATOR
        ct_ratio_N = st.sidebar.number_input("Neutral Side CT Rating (Primary A, e.g. 20000 in '20000:5')", value=p_data["ct_n"])
        ct_ratio_T = st.sidebar.number_input("Terminal Side CT Rating (Primary A)", value=p_data["ct_t"])

ct_secondary_rating = st.sidebar.selectbox(
    "CT Secondary Rating (A)", [1.0, 5.0], index=1,
    help="The rated secondary current stamped on the CT nameplate (e.g. the '5' in '2000:5'). "
         "This is applied to both CTs and determines the true turns ratio used in all "
         "per-unit scaling — entering only the primary rating without this was a labelling bug."
)
st.sidebar.caption(
    f"Effective ratio → Neutral/End1: **{ct_ratio_N:.0f} : {ct_secondary_rating:.0f}** "
    f"(= {ct_ratio_N/ct_secondary_rating:.1f}:1)  |  "
    f"Terminal/End2: **{ct_ratio_T:.0f} : {ct_secondary_rating:.0f}** "
    f"(= {ct_ratio_T/ct_secondary_rating:.1f}:1)"
)

if current_mode == "LINE":
    st.sidebar.header("🗺️ Line Geometry & Transmission")
    line_len = st.sidebar.number_input("Line Length (km)", value=50.0, step=10.0)
    charging_curr = st.sidebar.number_input("Charging Current rate (A/km)", value=0.15, step=0.05, help="Capacitive charging current parameter")
else:
    line_len = 0.0
    charging_curr = 0.0

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

st.sidebar.header("3. Blocking Harmonics & Wiring")
if current_mode == "TRANSFORMER":
    harmonic_block_thresh = st.sidebar.slider("2nd Harmonic Limit (%)", 10, 30, 15, 1, help="Blocks on Transformer Inrush current")
    harmonic_5th_thresh = st.sidebar.slider("5th Harmonic Limit (%)", 20, 50, 35, 1, help="Blocks on Transformer Overexcitation")
else:
    harmonic_block_thresh = 15.0
    harmonic_5th_thresh = 35.0
    if current_mode == "GENERATOR":
        st.sidebar.caption(
            "ℹ️ 2nd/5th harmonic blocking is not applicable to generators — "
            "generators don't produce magnetizing inrush the way transformer "
            "cores do, so this element is disabled in Generator mode."
        )
    elif current_mode == "GENERATOR_LEGACY":
        st.sidebar.caption(
            "ℹ️ This relay has no harmonic restraint capability — it's a fixed "
            "percentage-differential design with a single pickup setting only."
        )

col_conv, col_pol = st.sidebar.columns(2)
with col_conv:
    convention = st.radio("Restraint Standard", ["IEEE", "IEC"], help="IEEE: Average current. IEC: Arithmetic sum.")
with col_pol:
    ct_polarity = st.radio("Polarity Reference", ["OPPOSITE", "SAME"], help="OPPOSITE: standard facing inwards. SAME: facing identical directions.")

# Create main relay object
relay = AdvancedDifferentialRelay(
    mode=current_mode, mva_rated=mva, kv_rated_pri=kv_pri, kv_rated_sec=kv_sec,
    ct_ratio_N=ct_ratio_N, ct_ratio_T=ct_ratio_T, ct_secondary_rating=ct_secondary_rating,
    i_pickup=i_pickup, slope_1=slope_1, i_breakpoint=i_bp, slope_2=slope_2, i_unrestrained=i_unrestrained,
    harmonic_block_threshold=harmonic_block_thresh, harmonic_5th_threshold=harmonic_5th_thresh,
    convention=convention, ct_polarity=ct_polarity,
    vector_group=vector_group, line_length_km=line_len, charging_current_a_per_km=charging_curr,
    target_amps=target_amps
)

# TABS CONFIG
tab1, tab2 = st.tabs(["📊 Live Vector Simulation", "🧰 Commissioning & Injection Tool"])


with tab1:
    col_inputs, col_results = st.columns([1.2, 1.0])

    with col_inputs:
        st.subheader("Primary (System) Operating Phase Inputs")
        st.caption(
            "Enter the actual PRIMARY-side current in Amps (e.g. generator load current or "
            "fault current at the machine terminals) — the app converts this through the CT "
            "ratio and rated base automatically. You do not need to divide by the CT ratio "
            "yourself. For the actual 0–5 A (or 0–1 A) secondary current you'd inject into "
            "the physical relay during testing, see the Commissioning & Injection Tool tab."
        )
        
        # Display derived system values
        if current_mode == "TRANSFORMER":
            st.info(f"Nominal Rated Primary Current: **{relay.i_rated_pri_H:.1f} A** | Secondary: **{relay.i_rated_pri_L:.1f} A**")
        elif current_mode == "LINE":
            st.info(f"Nominal Line Rated Current: **{relay.i_rated_pri_H:.1f} A**")
        else:
            st.info(f"Generator Nominal Rated Current: **{relay.i_rated_pri_H:.1f} A**")

        phases = ["Phase A", "Phase B", "Phase C"]

        # Side labels must match the actual physical meaning of N/T per equipment type —
        # Generator: both CTs sit on the SAME winding at the same voltage (neutral end vs
        # terminal end), so "Primary/Secondary" (a transformer voltage-ratio concept) is wrong.
        if current_mode in ("GENERATOR", "GENERATOR_LEGACY"):
            n_side_label, t_side_label = "Neutral Side (End 1)", "Terminal Side (End 2)"
        elif current_mode == "TRANSFORMER":
            n_side_label, t_side_label = "Primary (HV)", "Secondary (LV)"
        else:  # LINE
            n_side_label, t_side_label = "Local (End 1)", "Remote (End 2)"
        inputs = {}

        # Capture Phase inputs in tabs/expanders
        for idx, phase in enumerate(phases):
            with st.expander(f"📌 {phase} Settings", expanded=(phase == "Phase A")):
                if current_mode in ("GENERATOR", "GENERATOR_LEGACY"):
                    c1, c2 = st.columns(2)
                else:
                    c1, c2, c3 = st.columns(3)
                
                # Default values for anti-parallel current flow under healthy conditions
                def_val_N = relay.i_rated_pri_H if phase == "Phase A" else 0.0
                def_val_T = relay.i_rated_pri_L if phase == "Phase A" else 0.0
                def_ang_N = -120.0 * idx
                # Under opposite CT polarity, normal load will show terminal side shifted by 180 deg
                def_ang_T = def_ang_N + 180.0 if ct_polarity == "OPPOSITE" else def_ang_N
                
                with c1:
                    i_N = st.number_input(f"{n_side_label} Primary Amps [A]", value=def_val_N, key=f"N_i_{phase}")
                    a_N = st.number_input(f"{n_side_label} Angle (°)", value=def_ang_N, key=f"N_a_{phase}")
                with c2:
                    i_T = st.number_input(f"{t_side_label} Primary Amps [A]", value=def_val_T, key=f"T_i_{phase}")
                    a_T = st.number_input(f"{t_side_label} Angle (°)", value=def_ang_T, key=f"T_a_{phase}")
                if current_mode in ("GENERATOR", "GENERATOR_LEGACY"):
                    h2 = 0.0
                    h5 = 0.0
                else:
                    with c3:
                        h2 = st.number_input(f"2nd Harmonic (%)", value=0.0, key=f"H2_{phase}")
                        h5 = st.number_input(f"5th Harmonic (%)", value=0.0, key=f"H5_{phase}")

                inputs[phase] = {"i_N": i_N, "a_N": a_N, "i_T": i_T, "a_T": a_T, "h2": h2, "h5": h5}

        # Calculate live state evaluation
        evals = {p: relay.evaluate_protection(
            inputs[p]["i_N"], inputs[p]["a_N"], 
            inputs[p]["i_T"], inputs[p]["a_T"], 
            inputs[p]["h2"], inputs[p]["h5"]
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
            file_name=f"Differential_Protection_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )


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
    has_unrestrained_element = relay.i_unrestrained < 1e5
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

    if current_mode in ("GENERATOR", "GENERATOR_LEGACY"):
        n_inj_label, t_inj_label = "Neutral Side", "Terminal Side"
    elif current_mode == "TRANSFORMER":
        n_inj_label, t_inj_label = "Primary Winding", "Secondary Winding"
    else:
        n_inj_label, t_inj_label = "Local (End 1)", "Remote (End 2)"

    c_sec_a, c_sec_b = st.columns(2)
    with c_sec_a:
        st.info(f"**{n_inj_label} Secondary Injection Current ($I_N$):**\n# {sec_N_injection:.3f} Amps AC")
    with c_sec_b:
        st.info(f"**{t_inj_label} Secondary Injection Current ($I_T$):**\n# {sec_T_injection:.3f} Amps AC")
