"""
Placement and single-unit millisecond latency cost model with Abilene topology and Headroom check.
"""
import numpy as np
import yaml
import os

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

# Abilene 12-node synthetic base round-trip time matrix (ms)
# Seeded symmetric delay matrix approximating US transcontinental routing (2ms - 15ms)
ABILENE_Rtt_MATRIX = np.array([
    [0.0,  5.2, 11.4,  8.1, 14.2,  9.0, 12.3,  6.5, 10.1, 13.5,  7.2,  4.9],
    [5.2,  0.0,  7.1, 12.3,  9.5, 11.0,  8.4, 10.2,  6.3,  9.8, 11.1,  8.0],
    [11.4, 7.1,  0.0,  9.2,  6.4, 13.1,  5.0, 12.0,  8.2,  4.1, 10.5, 14.0],
    [8.1, 12.3,  9.2,  0.0, 10.5,  4.2, 11.3,  5.1, 12.0,  9.6,  6.3,  7.4],
    [14.2, 9.5,  6.4, 10.5,  0.0, 12.0,  7.3, 11.1,  4.2,  8.5, 13.0, 10.2],
    [9.0, 11.0, 13.1,  4.2, 12.0,  0.0, 10.1,  8.3, 11.2, 10.4,  5.0,  6.1],
    [12.3, 8.4,  5.0, 11.3,  7.3, 10.1,  0.0,  9.4,  6.1,  5.2, 11.5, 12.1],
    [6.5, 10.2, 12.0,  5.1, 11.1,  8.3,  9.4,  0.0,  7.5, 10.1,  4.2,  5.3],
    [10.1, 6.3,  8.2, 12.0,  4.2, 11.2,  6.1,  7.5,  0.0,  8.4,  9.1, 11.0],
    [13.5, 9.8,  4.1,  9.6,  8.5, 10.4,  5.2, 10.1,  8.4,  0.0, 12.0, 13.2],
    [7.2, 11.1, 10.5,  6.3, 13.0,  5.0, 11.5,  4.2,  9.1, 12.0,  0.0,  3.8],
    [4.9, 8.0, 14.0,  7.4, 10.2,  6.1, 12.1,  5.3, 11.0, 13.2,  3.8,  0.0]
])

def compute_node_service_time(cpu_util_pct, config):
    """
    M/M/1 queueing delay inflation based on CPU utilization:
    S_node = S0 / (1 - rho_cpu)
    """
    pcfg = config['placement']
    s0 = pcfg['service_time_ms']
    rho_cap = pcfg['rho_cap']
    
    # cpu_util_pct is 0-100, convert to fraction 0-1
    rho = np.clip(cpu_util_pct / 100.0, 0.0, rho_cap)
    return s0 / (1.0 - rho)

def run_headroom_check():
    config = load_config()
    data_path = 'data/processed/usage_1min.npy'
    if not os.path.exists(data_path):
        print('Processed data not found.')
        return
        
    data = np.load(data_path) # [F=5, N=12, T]
    # Extract CPU feature (index 0) for our 12 nodes across test split
    cpu_data = data[0] # [12, T]
    T = cpu_data.shape[1]
    test_start = int(T * 0.85)
    
    print(f'Running Headroom Check on test window (steps {test_start} to {T})...')
    
    reactive_latencies = []
    oracle_latencies = []
    
    for t in range(test_start, T):
        # Current state CPU (for Reactive)
        curr_cpu = cpu_data[:, t]
        if np.any(np.isnan(curr_cpu)):
            continue
            
        # Simulate selecting coordinator across the 12 nodes
        reactive_costs = [compute_node_service_time(curr_cpu[n], config) + 
                          2.0 * ABILENE_Rtt_MATRIX[n, (n+1)%12] for n in range(12)]
        reactive_latencies.append(min(reactive_costs))
        
        # Oracle state CPU (looking ahead 1 step, true future)
        if t + 1 < T:
            future_cpu = cpu_data[:, t+1]
            if not np.any(np.isnan(future_cpu)):
                oracle_costs = [compute_node_service_time(future_cpu[n], config) + 
                                2.0 * ABILENE_Rtt_MATRIX[n, (n+1)%12] for n in range(12)]
                oracle_latencies.append(min(oracle_costs))
                
    p95_react = np.percentile(reactive_latencies, 95) if reactive_latencies else 0
    p95_oracle = np.percentile(oracle_latencies, 95) if oracle_latencies else 0
    gap = ((p95_react - p95_oracle) / p95_react) * 100 if p95_react > 0 else 0
    
    print(f'--- Headroom Check Results ---')
    print(f'Reactive p95 Latency: {p95_react:.2f} ms')
    print(f'Oracle p95 Latency:   {p95_oracle:.2f} ms')
    print(f'Headroom Gap (Potential Improvement): {gap:.2f}%')

if __name__ == '__main__':
    run_headroom_check()
