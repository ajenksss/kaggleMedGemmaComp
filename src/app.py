import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import plotly.graph_objects as go

# Import our Layers
from mock_stream import VitalStream
from layer_1_tda import TopologicalSensor
from layer_2_pinn import HemodynamicPINN
from layer_3_kan import PhysicsInformedKAN
import importlib
import layer_4_agent
importlib.reload(layer_4_agent) # FORCE RELOAD to fix stale cache
from layer_4_agent import MedGemmaAgent

st.set_page_config(page_title="MedGemma Triage Copilot", layout="wide")

if "tpt_system" not in st.session_state:
    st.session_state.tpt_system = {
        "stream": VitalStream(csv_path="data/mock_vitals_v2.csv"),
        "tda": TopologicalSensor(),
        "pinn": HemodynamicPINN(),
        "kan": PhysicsInformedKAN(),
        "agent": MedGemmaAgent()
    }
    st.session_state.history = {
        "HR": [], "MAP": [], "Shape": [], "Risk": [], "RR": []
    }
    st.session_state.logs = []
    st.session_state.running = False

# Ensure Async State exists
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False
    st.session_state.analysis_result = None
    st.session_state.last_trigger_time = 0
    # Stability State
    st.session_state.last_alert_time = 0
    st.session_state.latched_decision = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎮 Controls")
    if st.button("Start Simulation"):
        st.session_state.running = True
    if st.button("Pause"):
        st.session_state.running = False
    
    st.markdown("---")
    st.header("🧠 Agent Config")
    model_id_input = st.text_input("Model ID", value="google/gemma-2b-it")
    
    if model_id_input != st.session_state.tpt_system["agent"].real_engine.model_id:
         st.session_state.tpt_system["agent"].real_engine.model_id = model_id_input
         st.session_state.tpt_system["agent"].real_engine.is_loaded = False 
         st.toast(f"Model switched to {model_id_input}")

    run_deep_analysis = st.button("Run Deep Analysis (Real Model)")
    enable_auto_analysis = st.toggle("⚡ Enable Auto-Copilot", value=True)

# --- HEADER (The Hook) ---
st.title("MedGemma: The Triage Copilot")
st.markdown("**Goal**: Resolve ambiguity between 'Normal Vitals' and 'Unstable Physiology'.")

# --- MEDGEMMA HERO SECTION (The "Meaning Bridge") ---
st.markdown("---")
# Use an empty placeholder that we can overwrite every loop iteration
copilot_placeholder = st.empty()

# --- METRICS ROW ---
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
metric_hr = kpi1.empty()
metric_map = kpi2.empty()
metric_spo2 = kpi3.empty()
metric_rr = kpi4.empty()
metric_risk = kpi5.empty()

# --- EVIDENCE SECTION (Collapsible "Math Flex") ---
st.markdown("### 🔍 Clinical Evidence Stack")
with st.expander("Show Topological & Physics Evidence", expanded=True):
    col_raw, col_eng = st.columns(2)
    with col_raw:
        st.subheader("Raw Vitals Stream")
        chart_vitals = st.empty()
    with col_eng:
        st.subheader("Topological Manifold (TDA)")
        st.caption("Visualizing physiological coherence. Explosion = Instability.")
        chart_topology = st.empty()
        chart_shape = st.empty()


# --- SIMULATION LOOP ---
if st.session_state.running:
    system = st.session_state.tpt_system
    
    # Process 1 step at a time for clarity
    for chunk_idx, chunk_data in enumerate(system["stream"].stream(chunk_size=1, delay=0.08)):
        
        # 0. Get Data
        row = chunk_data.iloc[0]
        vitals_vec = row[["HR", "MAP", "SpO2", "Temp", "RR"]].values
        
        # 1. TDA (Shape)
        shape_score = system["tda"].update(vitals_vec)
        
        # 2. PINN (Physics)
        phys_valid, phys_score, phys_reason = system["pinn"].validate(vitals_vec)
        
        # 3. KAN (Risk)
        risk_score, formula = system["kan"].predict_risk(shape_score, phys_score)
        
        # 4. Agent Decision
        
        # Snapshot string for the agent
        snapshot_str = f"HR {row['HR']:.0f}, MAP {row['MAP']:.0f}, RR {row['RR']:.0f}"
        shape_desc = f"Stable (Radius {shape_score:.2f})"
        if shape_score > 2.0:
             shape_desc = f"EXPLODING (Radius {shape_score:.2f}). Variance High."

        # Default: Simulation Result
        decision = system["agent"].evaluate(risk_score, phys_valid, formula, 
                                          run_real_inference=False, 
                                          shape_desc=shape_desc,
                                          vitals_snapshot=snapshot_str)
        
        # Triggers
        triggers = []
        if run_deep_analysis: triggers.append("Manual")
        if enable_auto_analysis and risk_score > 0.6: triggers.append("Auto-Risk") # Lower threshold for "Concern"
        
        # Background Thread (Real Inference)
        if triggers and not st.session_state.is_analyzing:
            if time.time() - st.session_state.get("last_trigger_time", 0) > 8:
                st.session_state.is_analyzing = True
                st.session_state.last_trigger_time = time.time()
                
                def background_task(chk_agent, r_score, p_valid, form, s_desc, v_snap):
                    res = chk_agent.evaluate(r_score, p_valid, form, 
                                           run_real_inference=True, 
                                           shape_desc=s_desc,
                                           vitals_snapshot=v_snap)
                    st.session_state.analysis_result = res
                    st.session_state.is_analyzing = False
                    
                t = threading.Thread(target=background_task, args=(system["agent"], risk_score, phys_valid, formula, shape_desc, snapshot_str))
                t.start()
                st.toast(f"🧠 Copilot Thinking... ({triggers[0]})")

        # Hysteresis / Stability Logic for UI
        curr_time = time.time()
        if st.session_state.analysis_result and (curr_time - st.session_state.last_trigger_time < 15):
            display_decision = st.session_state.analysis_result
        else:
             # Latched Logic
            if decision['risk_state'] in ["RED", "ORANGE"]:
                st.session_state.latched_decision = decision
                st.session_state.last_alert_time = curr_time
                display_decision = decision
            elif (curr_time - st.session_state.get("last_alert_time", 0) < 5.0) and st.session_state.latched_decision:
                 display_decision = st.session_state.latched_decision
            else:
                 display_decision = decision
            
        # --- UI UPDATE: COPILOT HERO ---
        # Clear and rebuild the container each frame to prevent stacking
        with copilot_placeholder.container(border=True):
            # 1. Status Banner
            state = display_decision.get('risk_state', 'GREEN')
            
            if st.session_state.is_analyzing:
                st.info("🧠 **MedGemma is analyzing patterns...** (Resolving Ambiguity)")
            else:
                if state == "RED":
                    st.error(f"🚨 **CRITICAL: {display_decision.get('conflict', 'Risk Detected')}**")
                elif state == "ORANGE":
                    st.warning(f"⚠️ **CONCERN: {display_decision.get('conflict', 'Instability')}**")
                elif state == "YELLOW":
                    st.warning(f"⚠️ **SENSOR: {display_decision.get('conflict', 'Artifact')}**")
                else:
                    st.success("✅ **Patient Stable** (Monitoring for latent shifts)")

            # 2. Structured Rationale Grid
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"**Rationale**: {display_decision.get('rationale', 'No active concerns.')}")
            with c2:
                st.caption(f"Source: {display_decision.get('inference_mode', 'Sim')}")

            # 3. Actionable Checks (The "Copilot" part)
            checks = display_decision.get('suggested_checks', 'Continue standard monitoring.')
            if state != "GREEN":
                st.info(f"**Suggested Clarifying Checks**: {checks}")

        
        # --- METRICS UPDATE ---
        metric_hr.metric("Heart Rate", f"{row['HR']:.0f} bpm")
        metric_map.metric("MAP", f"{row['MAP']:.0f} mmHg", delta_color="inverse")
        metric_spo2.metric("SpO2", f"{row['SpO2']:.0f}%")
        metric_rr.metric("Resp Rate", f"{row['RR']:.0f}", help="Tachypnea is often the first sign.")
        # Risk Metric uses the Copilot's State color now
        metric_risk.metric("Hemodynamic Risk", f"{risk_score*100:.0f}%", delta=state, delta_color="inverse")
                           
        # --- EVIDENCE CHARTS ---
        hist = st.session_state.history
        hist["HR"].append(row["HR"])
        hist["MAP"].append(row["MAP"])
        hist["Shape"].append(shape_score)
        hist["RR"].append(row["RR"])
        if len(hist["HR"]) > 100: 
            for k in hist: hist[k].pop(0)
            
        chart_vitals.line_chart(pd.DataFrame({"HR": hist["HR"], "MAP": hist["MAP"]}))
        
        # 3D Manifold (Point Cloud)
        cloud = system["tda"].point_cloud
        if len(cloud) > 5:
            cloud_arr = np.array(cloud)
            fig = go.Figure(data=[go.Scatter3d(
                x=cloud_arr[:, 0], y=cloud_arr[:, 1], z=cloud_arr[:, 2],
                mode='markers',
                marker=dict(size=4, color=list(range(len(cloud))), colorscale='Viridis', opacity=0.8)
            )])
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube'),
                height=250,
            )
            chart_topology.plotly_chart(fig, use_container_width=True, key=f"topo_{chunk_idx}")
            
        # Stop check
        if not st.session_state.running: break
else:
    st.info("Click 'Start Simulation' to enable the Triage Copilot.")
