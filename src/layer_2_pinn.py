import numpy as np

class HemodynamicPINN:
    def __init__(self):
        """
        Layer 2: Physics-Informed Neural Network (PINN) - The 'Legislative' Layer.
        
        Physics Laws (Hemodynamics):
        1. MAP = CO * SVR  (Mean Arterial Pressure = Cardiac Output * Resistance)
        2. CO = HR * SV    (Cardiac Output = Heart Rate * Stroke Volume)
        
        Constraints:
        - SVR (Systemic Vascular Resistance) has physical limits (700 - 1600 dynes).
        - Extremely low SVR (< 500) implies Vasodilation (Septic Shock).
        - Extremely high SVR (> 3000) implies Vasoconstriction.
        - Impossible SVR (< 0) -> Physics Violation (Sensor Error).
        """
        # Estimates for Stroke Volume (SV) based on averages (ml/beat)
        # In a real system, SV is estimated from Pulse Pressure (PP = SBP - DBP)
        self.estimated_sv = 70.0 

    def validate(self, vitals):
        """
        Checks if the vitals obey the Laws of Physics.
        Returns: (is_valid, physics_score, reason)
        """
        # Unpack
        # vitals: [HR, MAP, SpO2, Temp]
        if isinstance(vitals, np.ndarray):
            vitals = vitals.flatten()
            
        hr = vitals[0]
        map_val = vitals[1]
        
        # 0. Sanity Check (The "Hard" Filter)
        if map_val > 300 or map_val < 10:
            return False, 0.0, "Impossible BP"
        if hr > 300 or hr < 10:
            return False, 0.0, "Impossible HR"

        # 1. Calculate Latent Variables (The "Hidden" Physics)
        # CO (L/min) = (HR * SV) / 1000
        co = (hr * self.estimated_sv) / 1000.0
        
        # SVR (dynes/sec/cm-5) = 80 * MAP / CO
        if co == 0: 
            return False, 0.0, "Zero Cardiac Output"
            
        svr = 80 * map_val / co
        
        # 2. Residual Analysis (The "Soft" Filter)
        # "Normal" SVR is 800-1200.
        # "Septic" SVR is often < 800 (Vasodilation).
        # "Impossible" SVR is < 100 or > 5000 (likely sensor noise).
        
        if svr < 100 or svr > 5000:
            # Huge Residual -> Physics Violation
            return False, 0.0, f"Physics Violation (SVR={svr:.0f} impossible)"
            
        # 3. Gated Mixture / Severity Scoring
        # If SVR is valid but LOW, it supports the "Shock" hypothesis.
        # We return a 'Physics Severity Score' (1.0 = Healthy, 0.0 = Shock state confirmed by physics)
        
        # Normalize SVR to a 0-1 score where 1 is healthy (1000) and 0 is dangerous (600)
        # Sigmoid-like mapping
        severity = np.clip((svr - 400) / 600, 0, 1)
        
        return True, severity, f"Valid (SVR={svr:.0f})"

if __name__ == "__main__":
    pinn = HemodynamicPINN()
    
    # Test Data
    tests = [
        ("Healthy", [75, 90, 98, 37]),  # Normal HR, Normal MAP
        ("Compensating", [100, 90, 98, 38]), # High HR, Normal MAP (Validation: SVR should drop)
        ("Septic Shock", [120, 50, 90, 39]), # High HR, Low MAP (Validation: SVR should be CRITICAL)
        ("Sensor Noise", [75, 350, 0, 0])    # Impossible BP
    ]
    
    print("Running PINN Validation...")
    for label, data in tests:
        valid, score, reason = pinn.validate(np.array(data))
        print(f"[{label}] -> Valid: {valid}, Health Score: {score:.2f}, Note: {reason}")
