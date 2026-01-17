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

st.set_page_config(page_title="MedGemma TPT Monitor", layout="wide")

st.warning(
    "Disclaimer: Prototype decision support demo only. Not medical advice. "
    "Do not use for real patient care. Clinician must verify."
)

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

# Ensure Async State exists (even if tpt_system already exists)
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
    st.header("🧠 MedGemma Agent")
    # Configurable Model ID (Resolves "Gemma vs MedGemma" critique)
    model_id_input = st.text_input("Model ID", value="google/gemma-2b-it", help="Use 'google/medgemma-2b' if you have access")
    
    # Update Agent if Model ID changes
    if model_id_input != st.session_state.tpt_system["agent"].real_engine.model_id:
         st.session_state.tpt_system["agent"].real_engine.model_id = model_id_input
         st.session_state.tpt_system["agent"].real_engine.is_loaded = False # Force reload
         st.toast(f"Model switched to {model_id_input}")

    run_deep_analysis = st.button("Run Deep Analysis (Real Model)")
    enable_auto_analysis = st.toggle("⚡ Enable Auto-Reaction (Real-Time)", value=False, help="Automatically triggers Deep Analysis when Sepsis Risk > 80%")
    
    st.markdown("---")
    st.markdown("**System Architecture**")
    st.markdown("1. **Data**: Mock MIMIC-III")
    st.markdown("2. **Sensing**: TDA (JL-Project + Witness)")
    st.markdown("3. **Physics**: PINN (Residual Check)")
    st.markdown("4. **Logic**: KAN (Neuro-symbolic)")
    st.markdown("5. **Agent**: MedGemma (Decision)")

# --- MAIN LAYOUT ---
st.title("MedGemma: Topological-Physical Twin (TPT)")

# Top Metrics Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
metric_hr = kpi1.empty()
metric_map = kpi2.empty()
metric_spo2 = kpi3.empty()
metric_risk = kpi5.empty()
metric_rr = kpi4.empty()

# MEDGEMMA COMMAND CENTER (High Visibility)
st.markdown("---")
st.subheader("🤖 MedGemma Decision Support")
ai_container = st.container(border=True)
with ai_container:
    ai_status = st.empty()
    ai_message = st.empty()

# Charts Row
st.markdown("---")
col_raw, col_eng = st.columns(2)
with col_raw:
    st.subheader("Raw Vitals (The 'Dumb' Monitor)")
    chart_vitals = st.empty()
    
with col_eng:
    st.subheader("Early Warning Signal (TDA)")
    st.caption("Measures 'Physiological Chaos'. Spikes > 2.0 predict crash.")
    
    # LIVE TOPOLOGY VISUALIZATION (The "Bouncing Ball")
    # We show the underlying point cloud to demonstrate the "Shape" directly
    chart_topology = st.empty()
    chart_shape = st.empty()
    
# Agent Log (Hidden, replaced by Command Center)
# st.subheader("🤖 MedGemma Clinical Orders")
# log_container = st.empty()

# --- SIMULATION LOOP ---
if st.session_state.running:
    system = st.session_state.tpt_system
    
    # Process 50 steps at a time for demo speed
    for chunk_idx, chunk_data in enumerate(system["stream"].stream(chunk_size=1, delay=0.05)):
        
        # 0. Get Data
        # [HR, MAP, SpO2, Temp, RR]
        row = chunk_data.iloc[0]
        vitals_vec = row[["HR", "MAP", "SpO2", "Temp", "RR"]].values
        
        # 1. TDA (Shape)
        shape_score = system["tda"].update(vitals_vec)
        
        # 2. PINN (Physics)
        phys_valid, phys_score, phys_reason = system["pinn"].validate(vitals_vec)
        
        # 3. KAN (Risk)
        risk_score, formula = system["kan"].predict_risk(shape_score, phys_score)
        
        # 4. Agent Decision (ASYNC NON-BLOCKING)
        
        # Default: Use Fast Simulation Result
        decision = system["agent"].evaluate(risk_score, phys_valid, formula, run_real_inference=False)
        
        # Check Triggers
        triggers = []
        if run_deep_analysis: triggers.append("Manual")
        if enable_auto_analysis and risk_score > 0.8: triggers.append("Auto-Sepsis")
        
        # Describe Shape for Agent
        shape_desc = f"Stable (Radius {shape_score:.2f})"
        if shape_score > 2.0:
             shape_desc = f"EXPLODING MANIFOLD (Radius {shape_score:.2f} >> 2.0). High Variance."
        
        # Start Background Thread if Triggered and Idle
        if triggers and not st.session_state.is_analyzing:
            # Debounce (prevent spamming threads)
            if time.time() - st.session_state.get("last_trigger_time", 0) > 5:
                st.session_state.is_analyzing = True
                st.session_state.last_trigger_time = time.time()
                
                def background_task(chk_agent, r_score, p_valid, form, s_desc):
                    # Heavy Compute
                    res = chk_agent.evaluate(r_score, p_valid, form, run_real_inference=True, shape_desc=s_desc)
                    st.session_state.analysis_result = res
                    st.session_state.is_analyzing = False
                    
                t = threading.Thread(target=background_task, args=(system["agent"], risk_score, phys_valid, formula, shape_desc))
                t.start()
                st.toast(f"🧠 Deep Analysis Started ({triggers[0]})...")

        # --- STABILIZER LOGIC (Hysteresis) ---
        # Solving "Jitter" / Alert Fatigue
        curr_time = time.time()
        
        # 1. Prefer Real Analysis if fresh ( < 10s old)
        if st.session_state.analysis_result and (curr_time - st.session_state.last_trigger_time < 10):
            display_decision = st.session_state.analysis_result
        else:
            # 2. Otherwise use Simulation, but LATCH it
            # If we have a high alert, hold it for 3 seconds to avoid flickering
            if decision['alert_level'] in ["RED", "ORANGE"]:
                st.session_state.latched_decision = decision
                st.session_state.last_alert_time = curr_time
                display_decision = decision
            
            # If we are effectively "Green" now, check if we are still in the hold period
            elif (curr_time - st.session_state.get("last_alert_time", 0) < 3.0) and st.session_state.latched_decision:
                 # Hold the old alert
                 display_decision = st.session_state.latched_decision
            else:
                 # Truly Green
                 display_decision = decision
            
        # Display Status
        if st.session_state.is_analyzing:
            ai_status.info("🧠 **MedGemma Thinking...** (Analysis in Progress)")
        
        # --- UPDATE UI ---
        
        
        # Metrics
        metric_hr.metric("Heart Rate", f"{row['HR']:.0f} bpm")
        metric_map.metric("MAP", f"{row['MAP']:.0f} mmHg", delta_color="inverse")
        metric_spo2.metric("SpO2", f"{row['SpO2']:.0f}%")
        metric_rr.metric("Resp Rate", f"{row['RR']:.0f} /min", help="Tachypnea is often the first sign of sepsis.")
        metric_risk.metric("Sepsis Risk", f"{risk_score*100:.0f}%", 
                           delta=f"{decision['alert_level']}", delta_color="inverse")
                           
        # History
        hist = st.session_state.history
        hist["HR"].append(row["HR"])
        hist["MAP"].append(row["MAP"])
        hist["Shape"].append(shape_score)
        hist["Risk"].append(risk_score)
        hist["RR"].append(row["RR"])
        
        # Trim history
        if len(hist["HR"]) > 100:
            for k in hist: hist[k].pop(0)
            
        # Charts
        chart_vitals.line_chart(pd.DataFrame({"HR": hist["HR"], "MAP": hist["MAP"]}))
        
        # TDA Chart: 1. Live Manifold (The "Ball")
        # Extract the point cloud from the TDA sensor
        # It is a list of 10D vectors. We take Dim 0 and Dim 1 for 2D projection.
        cloud = system["tda"].point_cloud
        if len(cloud) > 5:
            cloud_arr = np.array(cloud)
            
            # Interactive 3D Scatter (Plotly)
            # visualizes the "Physiological Manifold" in 3 dimensions
            # Healthy = Tight Ball. Sepsis = Exploded Cloud.
            fig = go.Figure(data=[go.Scatter3d(
                x=cloud_arr[:, 0],
                y=cloud_arr[:, 1],
                z=cloud_arr[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=list(range(len(cloud))), # Age gradient
                    colorscale='Viridis',
                    opacity=0.8
                )
            )])
            
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis=dict(showticklabels=False, title='Dim 1'),
                    yaxis=dict(showticklabels=False, title='Dim 2'),
                    zaxis=dict(showticklabels=False, title='Dim 3'),
                    aspectmode='cube'
                ),
                height=300,
            )
            
            chart_topology.plotly_chart(fig, use_container_width=True, key=f"topo_{chunk_idx}")
        
        # TDA Chart: 2. The Instability Score (The "Red Line")
        df_shape = pd.DataFrame({
            "Instability": hist["Shape"],
            "Critical Threshold": [2.0] * len(hist["Shape"])
        })
        chart_shape.line_chart(df_shape, color=["#ff4b4b", "#808080"])
        
        # AI Command Center Updates
        # Only show Alert if NOT analyzing (to prevent jitter/overwrite)
        if st.session_state.is_analyzing:
             # Just show thinking (matches logic above)
             pass 
        else:
            mode = display_decision.get("inference_mode", "SIMULATION")
            
            if display_decision['alert_level'] == "RED":
                ai_status.error("🚨 **CRITICAL SEPSIS ALERT** (Action Required)")
                ai_message.error(f"**MedGemma Recommendation:** {display_decision['recommendation']}\n\n*(Source: {mode})*")
            elif display_decision['alert_level'] == "ORANGE":
                ai_status.warning("⚠️ **Hemodynamic Instability Detected**")
                ai_message.warning(f"**MedGemma Recommendation:** {display_decision['recommendation']}")
            else:
                 ai_status.success("✅ Patient Stable (Monitoring)")
                 ai_message.info(f"MedGemma Monitoring... (Risk: {risk_score:.0%})")
        
        # Stop manually to prevent infinite freeze in some envs
        # In Streamlit Cloud, rerun handles this. Here we loop.
        if not st.session_state.running:
            break
            
else:
    st.info("Click 'Start Simulation' in the sidebar to begin.")
