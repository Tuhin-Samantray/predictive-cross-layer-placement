"""
Decision Engine: Evaluates transaction placement using predicted cross-layer metrics.
"""
import numpy as np
from src.placement import ABILENE_Rtt_MATRIX, compute_node_service_time

class DecisionEngine:
    def __init__(self, config):
        self.config = config
        self.sla = config["placement"]["sla_ms"]

    def evaluate_nodes(self, predicted_cpu_loads, coordinator_node):
        """
        Scores every node based on predicted CPU load and network RTT.
        Cost = 2PC Commit Latency + M/M/1 Queueing Delay
        """
        n_nodes = len(predicted_cpu_loads)
        costs = []
        
        for n in range(n_nodes):
            service_time = compute_node_service_time(predicted_cpu_loads[n], self.config)
            rtt_cost = 2.0 * ABILENE_Rtt_MATRIX[coordinator_node, n]
            total_cost = service_time + rtt_cost
            costs.append(total_cost)
            
        best_node = int(np.argmin(costs))
        min_cost = costs[best_node]
        
        return best_node, min_cost, costs

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    engine = DecisionEngine(config)
    dummy_cpu = np.array([30.0, 85.0, 20.0, 45.0, 60.0, 25.0, 40.0, 50.0, 35.0, 55.0, 20.0, 15.0])
    best, cost, all_costs = engine.evaluate_nodes(dummy_cpu, coordinator_node=0)
    print(f"Decision Engine Test -> Best Node: {best}, Cost: {cost:.2f}ms")

