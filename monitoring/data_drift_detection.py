import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import joblib
import os

class DataDriftDetector:
    def __init__(self, reference_data=None):
        self.reference_data = reference_data
    
    def set_reference_data(self, data):
        self.reference_data = data
        print("Reference data set for drift detection")
    
    def detect_drift(self, current_data, threshold=0.05):
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        drift_report = {}
        
        # For image data, we'll check basic statistics
        ref_mean = np.mean(self.reference_data, axis=(0,1,2))
        curr_mean = np.mean(current_data, axis=(0,1,2))
        
        ref_std = np.std(self.reference_data, axis=(0,1,2))
        curr_std = np.std(current_data, axis=(0,1,2))
        
        # Check for significant changes in statistics
        mean_drift = np.abs(ref_mean - curr_mean) / (ref_std + 1e-8)
        std_drift = np.abs(ref_std - curr_std) / (ref_std + 1e-8)
        
        drift_detected = np.any(mean_drift > threshold) or np.any(std_drift > threshold)
        
        drift_report = {
            'drift_detected': drift_detected,
            'mean_drift': mean_drift.tolist(),
            'std_drift': std_drift.tolist(),
            'threshold': threshold
        }
        
        return drift_report

def main():
    from src.data.data_pipeline import DataPipeline
    
    # Load reference data
    pipeline = DataPipeline()
    (x_train, y_train), (x_test, y_test) = pipeline.run_pipeline()
    
    # Initialize drift detector with training data as reference
    drift_detector = DataDriftDetector()
    drift_detector.set_reference_data(x_train)
    
    # Save drift detector
    joblib.dump(drift_detector, 'monitoring/drift_detector.joblib')
    print("Drift detector saved with reference data")

if __name__ == "__main__":
    main()
