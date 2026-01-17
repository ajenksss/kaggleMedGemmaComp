import pandas as pd
import numpy as np
import json
import random

# Import our "Teacher" Models
from layer_1_tda import TopologicalSensor
from layer_2_pinn import HemodynamicPINN
from layer_3_kan import PhysicsInformedKAN

def generate_physics_distillation_data(n_samples=1000):
    """
    Generates a 'Synthetic Fine-Tuning Dataset' for MedGemma 2B.
    
    The Strategy: "Physics-Informed Distillation" (Sim-to-Real).
    We don't need real patient data to teach the model PHYSICS.
    We generate random (but plausible) vitals, run them through our PINN/KAN,
    and treat the PINN's output as the "Ground Truth Label."
    
    This teaches the LLM to 'simulate' the physics engine in its head.
    """
    
    # Init Teachers
    tda = TopologicalSensor()
    pinn = HemodynamicPINN()
    kan = PhysicsInformedKAN()
    
    dataset = []
    
    print(f"Generating {n_samples} synthetic training examples...")
    
    for i in range(n_samples):
        # 1. Generate Random Vitals
        # Mix of Healthy (Normal dist) and Shock (Uniform random)
        if random.random() > 0.5:
             # Healthy-ish
             hr = np.random.normal(75, 10)
             map_bp = np.random.normal(90, 10)
        else:
             # Chaos / Shock
             hr = np.random.uniform(40, 160)
             map_bp = np.random.uniform(30, 120)
             
        # Create vector
        vitals = np.array([hr, map_bp, 98, 37])
        
        # 2. Ask Teachers for Labels
        # We simulate shape score probability (higher for chaos inputs)
        shape_score = abs(hr - 75)/30 + abs(map_bp - 90)/30 
        
        valid, phys_score, reason = pinn.validate(vitals)
        risk, formula = kan.predict_risk(shape_score, phys_score)
        
        # 3. Formulate the "Chain of Thought" Label
        # This is what we want MedGemma to learn to output
        
        action = "MONITOR"
        if not valid: action = "CHECK_SENSOR"
        elif risk > 0.8: action = "PROTOCOL_SEPSIS"
        elif risk > 0.5: action = "PREPARE_VASOPRESSOR"
        
        # The PROMPT (Input)
        prompt = f"Analyze Vitals: HR={hr:.0f}, MAP={map_bp:.0f}. ShapeScore={shape_score:.1f}."
        
        # The COMPLETION (Output)
        completion = {
            "Physics_Check": "Valid" if valid else "Invalid",
            "Risk_Calculation": f"{risk:.0%}",
            "Reasoning": reason if not valid else formula,
            "Recommended_Action": action
        }
        
        entry = {
            "instruction": "You are MedGemma TPT, a physics-guided clinical assistant. Analyze the hemodynamics.",
            "input": prompt,
            "output": json.dumps(completion)
        }
        dataset.append(entry)

    # Save as JSONL (Standard Format for Fine-Tuning)
    with open("data/medgemma_physics_distillation.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print("Dataset Generated: data/medgemma_physics_distillation.jsonl")
    print("Example Entry:")
    print(json.dumps(dataset[0], indent=2))

if __name__ == "__main__":
    generate_physics_distillation_data()
