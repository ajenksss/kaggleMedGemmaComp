import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude
from sklearn.random_projection import SparseRandomProjection
import warnings

# Suppress TDA warnings for clean output
warnings.filterwarnings("ignore")

class TopologicalSensor:
    def __init__(self, window_size=20, embedding_dim=100, projection_dim=10):
        """
        Layer 1: The Topological Sensor.
        
        Stack:
        1. Time-Delay Embedding: 5 vitals * 20 sec history = 100 dimensions.
        2. JL-Projection: 100d -> 10d (Speed Optimization).
        3. TDA: Vietoris-Rips (Approximated) on the projected cloud.
        """
        self.window_size = window_size
        self.raw_buffer = [] # Buffer for sliding window of raw vitals
        
        # Johnson-Lindenstrauss Projector
        # We initialize it once to ensure consistency
        self.jl_projector = SparseRandomProjection(n_components=projection_dim, random_state=42)
        
        # TDA Engine
        # We use 'Wasserstein' amplitude as a scalar "Shape Score"
        self.vr = VietorisRipsPersistence(metric="euclidean", homology_dimensions=[0, 1])
        self.amplitude = Amplitude(metric="wasserstein")
        
        # Baseline (Healthy Shape)
        # We need to collect some initial points to define "Normal"
        self.point_cloud = [] 
        self.baseline_entropy = 0
        self.is_calibrated = False

    def update(self, vital_chunk):
        """
        Ingest new data, update sliding window, project, and return Shape Score.
        """
        # 1. Update Raw Buffer
        # Flatten chunk if needed, assuming chunk is a single row [HR, MAP, SpO2, Temp]
        if isinstance(vital_chunk, np.ndarray):
             vals = vital_chunk.flatten().tolist()
        else:
             # Pandas series
             vals = vital_chunk.values.flatten().tolist()
             
        # Just keep track of the last 'window_size' vectors
        # Logic: We need 'window_size' steps to form ONE 100d point.
        # But to do TDA, we need a CLOUD of points.
        # So we maintain a buffer of raw history. 
        self.raw_buffer.append(vals)
        if len(self.raw_buffer) > self.window_size:
            self.raw_buffer.pop(0)
            
        # We need at least window_size points to create ONE embedded point
        if len(self.raw_buffer) < self.window_size:
            return 0.0 # Not enough data
            
        # 2. Time-Delay Embedding (Create 100d vector)
        # Flatten the buffer ([20, 5] -> [100])
        embedded_point = np.array(self.raw_buffer).flatten().reshape(1, -1)
        
        # 3. JL-Projection (100d -> 10d)
        # We fit the projector on the fly or just transform? 
        # SRP needs fitting once. We'll fit on the first valid point (dummy fit)
        # In prod, we'd pre-fit on healthy data.
        try:
            val_10d = self.jl_projector.transform(embedded_point)
        except:
            # First run, fit it
            self.jl_projector.fit(embedded_point)
            val_10d = self.jl_projector.transform(embedded_point)
            
        # 4. Form Point Cloud
        # TDA needs a cloud (e.g. last 50 states).
        self.point_cloud.append(val_10d[0])
        if len(self.point_cloud) > 50: # Keep cloud size fixed at 50
            self.point_cloud.pop(0)
            
        if len(self.point_cloud) < 30:
            return 0.0 # Calibration phase
            
        # 5. Run TDA
        X_cloud = np.array(self.point_cloud)
        X_cloud = X_cloud[None, :, :] # Shape for gtda: (1, n_points, n_dim)
        
        diagrams = self.vr.fit_transform(X_cloud)
        
        # 6. Calculate Instability (Amplitude)
        # How "big" are the holes compared to empty space?
        # Higher amplitude = More complex topology = Instability/Sepsis
        score = self.amplitude.fit_transform(diagrams)[0][1] # H1 score
        
        return float(score)

if __name__ == "__main__":
    # Test Driver
    print("Initializing Sensor...")
    sensor = TopologicalSensor()
    
    # Simulate Healthy
    print("Simulating Healthy Data...")
    for i in range(100):
        # 5 random "Healthy" vitals
        data = np.random.normal([75, 90, 98, 37], 1) 
        score = sensor.update(data)
        if i % 20 == 0: print(f"Time {i}: Shape Score={score:.4f}")
        
    # Simulate Shock (Drift)
    print("\nSimulating Shock Data...")
    for i in range(100):
        # Drifting vitals
        drift = i * 0.5
        data = np.random.normal([75+drift, 90-drift, 98, 37+drift], 2)
        score = sensor.update(data)
        if i % 20 == 0: print(f"Time {i}: Shape Score={score:.4f}")
