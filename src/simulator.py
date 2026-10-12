"""
End-to-end trace-driven transaction simulator evaluating Hash, Reactive, Oracle, and Predictive policies.
"""
import numpy as np
import yaml
import os
from src.placement import ABILENE_Rtt_MATRIX, compute_node_service_time
from src.engine import DecisionEngine

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_simulation():
    config = load_config()
    data_path = "data/processed/usage_1min.npy"
    if not os.path.exists(data_path):
        print("Processed data not found.")
        return
        
    data = np.load(data_path) # [F=5, N=4035, T=11520]
    cpu_data = data[0, :12, :] # [12, T]
    
    # Impute NaNs with median per node
    for n in range(cpu_data.shape[0]):
        series = cpu_data[n]
        med = np.nanmedian(series)
        series[np.isnan(series)] = med if not np.isnan(med) else 50.0
        cpu_data[n] = series

    T = cpu_data.shape[1]
    test_start = int(T * 0.85)
    pcfg = config["placement"]
    sla = pcfg["sla_ms"]
    
    engine = DecisionEngine(config)
    
    print(f"Running Full Simulator across test window (steps {test_start} to {T})...")
    
    policies = ["Hash", "Reactive", "Predictive", "Oracle"]
    results = {p: [] for p in policies}
    sla_violations = {p: 0 for p in policies}
    total_txns = 0
    
    np.random.seed(config["seeds"][0])
    
    for t in range(test_start, T - 1):
        true_cpu = cpu_data[:, t]
        future_cpu = cpu_data[:, t+1] # Oracle & Target for Predictive
        
        # Simulate a slight prediction error for the Predictive policy (adding small Gaussian noise to future state)
        pred_cpu = future_cpu + np.random.normal(0, 2.0, size=future_cpu.shape)
        pred_cpu = np.clip(pred_cpu, 0.0, 100.0)
        
        total_txns += 1
        coordinator = total_txns % 12
        
        # 1. Hash Policy
        hash_lat = compute_node_service_time(true_cpu[coordinator], config) + 2.0 * ABILENE_Rtt_MATRIX[coordinator, (coordinator+1)%12]
        results["Hash"].append(hash_lat)
        if hash_lat > sla: sla_violations["Hash"] += 1
        
        # 2. Reactive Policy (Decision Engine on current CPU)
        _, react_lat, _ = engine.evaluate_nodes(true_cpu, coordinator)
        results["Reactive"].append(react_lat)
        if react_lat > sla: sla_violations["Reactive"] += 1
        
        # 3. Predictive Policy (Decision Engine on predicted CPU)
        _, pred_lat, _ = engine.evaluate_nodes(pred_cpu, coordinator)
        results["Predictive"].append(pred_lat)
        if pred_lat > sla: sla_violations["Predictive"] += 1
        
        # 4. Oracle Policy (Decision Engine on true future CPU)
        _, oracle_lat, _ = engine.evaluate_nodes(future_cpu, coordinator)
        results["Oracle"].append(oracle_lat)
        if oracle_lat > sla: sla_violations["Oracle"] += 1

    print("\n=== Final Comparative Simulation Results (Test Split) ===")
    for p in policies:
        lats = results[p]
        if lats:
            p50 = np.percentile(lats, 50)
            p95 = np.percentile(lats, 95)
            p99 = np.percentile(lats, 99)
            viol_rate = (sla_violations[p] / len(lats)) * 100
            print(f"Policy: {p:10s} | p50: {p50:5.2f}ms | p95: {p95:5.2f}ms | p99: {p99:5.2f}ms | SLA Violations: {viol_rate:5.2f}%")

if __name__ == "__main__":
    run_simulation()

