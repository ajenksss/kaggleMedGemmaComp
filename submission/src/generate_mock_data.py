import pandas as pd
import numpy as np
import os

def generate_sepsis_scenario():
    """
    Generates a synthetic dataset mimicking a patient going into Septic Shock.
    
    Phases:
    1. Healthy (0-2000 steps): Stable vitals.
    2. Compensation/Pre-Sepsis (2000-4000 steps): HR rises, BP wobbles. *TDA Target*
    3. Decompensation/Shock (4000-6000 steps): Hypotension, Tachycardia.
    
    Vitals:
    - HR (Heart Rate)
    - MAP (Mean Arterial Pressure)
    - SpO2 (Oxygen Saturation)
    - Temp (Temperature)
    - RR (Respiratory Rate) *NEW*
    """
    
    # Time steps (e.g., 1 second intervals)
    steps = 6000
    t = np.arange(steps)
    
    # Baselines
    hr = np.ones(steps) * 75
    map_bp = np.ones(steps) * 90
    spo2 = np.ones(steps) * 98
    temp = np.ones(steps) * 37.0
    rr = np.ones(steps) * 16.0 # Normal RR
    
    # Noise (Physiological variability)
    np.random.seed(42)
    hr += np.random.normal(0, 2, steps)
    map_bp += np.random.normal(0, 3, steps)
    spo2 += np.random.normal(0, 0.5, steps)
    temp += np.random.normal(0, 0.1, steps)
    rr += np.random.normal(0, 1.0, steps)
    
    # --- PHASE 2: Pre-Sepsis (Drift) ---
    # HR slowly creeps up
    hr[2000:] += np.linspace(0, 30, 4000) # Ends at +30 (105 bpm)
    
    # MAP gets "wobbly" (variance increases) before dropping
    # This "shape change" is what TDA detects before the value drops!
    map_noise = np.random.normal(0, 10, 4000) * np.linspace(0, 1, 4000)
    map_bp[2000:] += map_noise
    
    # Temp starts rising
    temp[2000:] += np.linspace(0, 2.0, 4000) # Ends at 39C
    
    # RR drifts up (Tachypnea is often the FIRST sign)
    rr[2000:] += np.linspace(0, 8, 4000) # Ends at 24 bpm
    
    # --- PHASE 3: Shock (Crash) ---
    # HR spikes
    hr[4000:] += 15 
    
    # MAP crashes (Hypotension)
    map_bp[4000:] -= np.linspace(0, 40, 2000) # Ends at 50 mmHg
    
    # SpO2 drops
    spo2[4000:] -= np.linspace(0, 10, 2000) # Ends at 88%
    
    # RR spikes further
    rr[4000:] += np.linspace(0, 6, 2000) # Ends at 30 bpm

    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": t,
        "HR": hr,
        "MAP": map_bp,
        "SpO2": spo2,
        "Temp": temp,
        "RR": rr
    })
    
    # Rounding
    df = df.round(2)
    
    # Save
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/mock_vitals.csv", index=False)
    print(f"Generated {len(df)} mocked vital points to data/mock_vitals.csv")

if __name__ == "__main__":
    generate_sepsis_scenario()
