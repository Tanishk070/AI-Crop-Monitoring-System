import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

def generate_health_report(index_name, data_map):
    """
    Performs a detailed numerical analysis on a given vegetation index map.
    """
    print(f"\n--- Detailed Report for {index_name} ---")

    # 1. Overall Statistics
    mean_val = np.mean(data_map)
    min_val = np.min(data_map)
    max_val = np.max(data_map)
    std_dev = np.std(data_map)
    print(f"Overall Stats:")
    print(f"  - Average (Health Score): {mean_val:.4f}")
    print(f"  - Min / Max Values: {min_val:.4f} / {max_val:.4f}")
    print(f"  - Standard Deviation (Uniformity): {std_dev:.4f}")

    # 2. Health Category Distribution
    total_pixels = data_map.size
    stressed_pixels = np.sum(data_map < 0.3)
    moderate_pixels = np.sum((data_map >= 0.3) & (data_map < 0.6))
    healthy_pixels = np.sum(data_map >= 0.6)
    
    print("\nHealth Category Distribution:")
    print(f"  - Stressed (< 0.3): {(stressed_pixels / total_pixels) * 100:.2f}%")
    print(f"  - Moderate (0.3 - 0.6): {(moderate_pixels / total_pixels) * 100:.2f}%")
    print(f"  - Healthy (> 0.6): {(healthy_pixels / total_pixels) * 100:.2f}%")
    
    # 3. Simulated Zone Analysis (4x4 Grid)
    print("\nSimulated Zone Analysis (Average Health per Zone):")
    rows, cols = data_map.shape
    row_step, col_step = rows // 4, cols // 4
    zone_alerts = []
    
    print("-" * 37)
    for i in range(4):
        row_str = "|"
        for j in range(4):
            zone = data_map[i*row_step:(i+1)*row_step, j*col_step:(j+1)*col_step]
            zone_mean = np.mean(zone)
            row_str += f" {zone_mean:.3f} |"
            if zone_mean < mean_val - std_dev: # Alert if a zone is significantly below average
                zone_alerts.append(f"Zone ({i+1},{j+1})")
        print(row_str)
        print("-" * 37)
    
    if zone_alerts:
        print("\n⚠️ Potential Problem Zones (significantly below average):")
        print(f"  - {', '.join(zone_alerts)}")

def run_full_analysis():
    # --- 1. CONFIGURATION ---
    mat_file_path = "C:/Programming/SIH_2025/AI-Crop-Monitoring-System/matlab/Vegetation_Indices/vegetation_indices.mat"

    # --- 2. LOAD THE DATA ---
    print(f"Loading data from: {mat_file_path}")
    mat_data = loadmat(mat_file_path)
    
    # --- 3. GENERATE REPORT FOR EACH INDEX ---
    indices_to_analyze = ['ndvi', 'savi', 'pri']
    for index in indices_to_analyze:
        if index in mat_data:
            generate_health_report(index.upper(), mat_data[index])
        else:
            print(f"\n- WARNING: '{index}' data not found in .mat file.")

if __name__ == "__main__":
    run_full_analysis()