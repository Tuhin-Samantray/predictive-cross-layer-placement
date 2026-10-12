"""
End-to-end trace-driven transaction simulator and policy evaluator.
"""
import numpy as np
import yaml
import os
from src.placement import ABILENE_Rtt_MATRIX, compute_node_service_time

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_simulation():
    config = load_config()
    data_path = 'data/processed/usage_1min.npy'
    if not os.path.exists(data_path):
        print('Processed data not found.')
        return
        
    data = np.load(data_path) # [F=5, N=4035, T=11520]
    # Select our 12 simulated nodes (first 12 machines for consistency)
    cpu_data = data[0, :12, :] # [12, T]
    
    # Impute NaNs with median per node
    for n in range(cpu_data.shape[0]):
        series = cpu_data[n]
        med = np.nanmedian(series)
        series[np.isnan(series)] = med if not np.isnan(med) else 50.0
        cpu_data[n] = series

    T = cpu_data.shape[1]
    test_start = int(T * 0.85)
    pcfg = config['placement']
    sla = pcfg['sla_ms']
    
    print(f'Running Simulator across test window (steps {test_start} to {T})...')
    
    policies = ['Hash', 'Reactive', 'Oracle']
    results = {p: [] for p in policies}
    sla_violations = {p: 0 for p in policies}
    total_txns = 0
    
    np.random.seed(config['seeds'][0])
    
    for t in range(test_start, T - 1):
        true_cpu = cpu_data[:, t]
        future_cpu = cpu_data[:, t+1]
        
        total_txns += 1
        
        # 1. Hash Policy
        hash_coord = total_txns % 12
        hash_lat = compute_node_service_time(true_cpu[hash_coord], config) + 2.0 * ABILENE_Rtt_MATRIX[hash_coord, (hash_coord+1)%12]
        results['Hash'].append(hash_lat)
        if hash_lat > sla: sla_violations['Hash'] += 1
        
        # 2. Reactive Policy
        react_costs = [compute_node_service_time(true_cpu[n], config) + 2.0 * ABILENE_Rtt_MATRIX[n, (n+1)%12] for n in range(12)]
        react_lat = min(react_costs)
        results['Reactive'].append(react_lat)
        if react_lat > sla: sla_violations['Reactive'] += 1
        
        # 3. Oracle Policy
        oracle_costs = [compute_node_service_time(future_cpu[n], config) + 2.0 * ABILENE_Rtt_MATRIX[n, (n+1)%12] for n in range(12)]
        oracle_lat = min(oracle_costs)
        results['Oracle'].append(oracle_lat)
        if oracle_lat > sla: sla_violations['Oracle'] += 1

    print('\n=== Simulation Results Summary (Test Split) ===')
    for p in policies:
        lats = results[p]
        if lats:
            p50 = np.percentile(lats, 50)
            p95 = np.percentile(lats, 95)
            p99 = np.percentile(lats, 99)
            viol_rate = (sla_violations[p] / len(lats)) * 100
            print(f'Policy: {p:10s} | p50: {p50:5.2f}ms | p95: {p95:5.2f}ms | p99: {p99:5.2f}ms | SLA Violations: {viol_rate:5.2f}%')

if __name__ == '__main__':
    run_simulation()
