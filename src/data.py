"""
Robust data pipeline with partial coverage extraction for simulation.
"""
import numpy as np
import os

def load_robust_segment():
    data_path = 'data/processed/usage_1min.npy'
    if not os.path.exists(data_path):
        print('Processed data not found.')
        return None
        
    data = np.load(data_path) # [F=5, N=12, T]
    
    # Fill missing values per feature/machine using rolling/median imputation for simulation stability
    for f in range(data.shape[0]):
        for n in range(data.shape[1]):
            series = data[f, n]
            mask = np.isnan(series)
            if mask.all():
                series[:] = 0.0
            else:
                med = np.nanmedian(series)
                series[mask] = med
                data[f, n] = series
                
    print('Robust data loaded and imputed successfully. Shape:', data.shape)
    return data

if __name__ == '__main__':
    load_robust_segment()
