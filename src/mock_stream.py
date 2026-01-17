import pandas as pd
import time
import numpy as np

class VitalStream:
    def __init__(self, csv_path="data/mock_vitals.csv"):
        self.csv_path = csv_path
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Stream loaded: {len(self.df)} points.")
        except FileNotFoundError:
            print("Error: Mock data not found. Run generate_mock_data.py first.")
            self.df = pd.DataFrame()
        
        self.index = 0
        
    def stream(self, chunk_size=1, delay=0.0):
        """
        Yields data points as if they are coming from a live monitor.
        """
        while self.index < len(self.df):
            chunk = self.df.iloc[self.index : self.index + chunk_size]
            self.index += chunk_size
            
            # Simulate network/sensor delay if needed
            if delay > 0:
                time.sleep(delay)
                
            yield chunk

    def get_shared_buffer(self):
        """
        Simulates the 'SharedArrayBuffer' for Zero-Copy TDA.
        In a real WASM implementation, this would point to a raw memory block
        shared between the JS Main Thread and the WASM Worker.
        """
        if self.index >= len(self.df):
            return None
            
        # Mocking a memory view of the next 100 points
        next_chunk = self.df.iloc[self.index : self.index + 100].values
        # Force contiguous array in memory
        buffer_view = np.ascontiguousarray(next_chunk, dtype=np.float32)
        return buffer_view

if __name__ == "__main__":
    # Test the stream
    streamer = VitalStream()
    print("Testing Stream...")
    for i, chunk in enumerate(streamer.stream(chunk_size=10)):
        if i > 2: break
        print(f"Chunk {i}: \n{chunk}")
        
    print("\nTesting Zero-Copy Buffer...")
    buf = streamer.get_shared_buffer()
    print(f"Buffer Shape: {buf.shape}, Type: {buf.dtype}")
