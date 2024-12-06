import random
import math
from collections import defaultdict, deque

def generate_sink_source_graph(n, r, upperCap, upperCost, output_file):
    """
    Generate a directed graph with random nodes and edges, ensuring non-negative costs.
    """
    nodes = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(n)]
    edges = []

    for i in range(n):
        for j in range(n):
            if i != j:
                distance = math.sqrt((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)
                if distance <= r:
                    rand = random.random()
                    if rand < 0.3:
                        # Forward edge with positive cost
                        capacity = random.randint(1, upperCap)
                        cost = random.randint(1, upperCost)
                        edges.append((i + 1, j + 1, capacity, cost))
                    elif rand < 0.6:
                        # Reverse edge with positive cost (handled as residual later)
                        capacity = random.randint(1, upperCap)
                        cost = random.randint(1, upperCost)
                        edges.append((j + 1, i + 1, capacity, cost))
                    # else: no edge

    with open(output_file, "w") as file:
        for edge in edges:
            file.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]}\n")

    print(f"Graph saved to {output_file} with {len(edges)} edges.")


def load_graph_from_file(file_path):
    """
    Load a graph from a file in EDGES format.

    Parameters:
    - file_path: Path to the graph file.

    Returns:
    - edges: List of edges as (source, destination, capacity, cost).
    """
    edges = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 4:
                continue  # Skip malformed lines
            u, v, cap, cost = map(int, parts)
            edges.append((u, v, cap, cost))
    return edges

def find_largest_connected_component(edges, n):
    """
    Find the largest connected component (LCC) in the graph.
    Treat edges as undirected for connectivity purposes.

    Parameters:
    - edges: List of edges as (source, destination, capacity, cost).
    - n: Number of nodes in the graph.

    Returns:
    - lcc: List of nodes in the largest connected component.
    """
    graph = defaultdict(list)
    for u, v, _, _ in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    largest_component = []

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in range(1, n + 1):  # Nodes are 1-indexed
        if node not in visited:
            component = []
            dfs(node, component)
            if len(component) > len(largest_component):
                largest_component = component

    return largest_component

def select_source_and_sink(lcc, edges):
    """
    Select s and t from the LCC. s is arbitrary (first in LCC), 
    t is the farthest node from s in terms of BFS hops.

    Parameters:
    - lcc: List of nodes in the largest connected component.
    - edges: List of edges as (source, destination, capacity, cost).

    Returns:
    - s: Source node.
    - t: Sink node.
    - max_path_length: Length of the longest path from s to t.
    """
    graph = defaultdict(list)
    for u, v, _, _ in edges:
        if u in lcc and v in lcc:
            graph[u].append(v)

    def bfs(start):
        distances = {}
        queue = deque([start])
        distances[start] = 0
        furthest_node = start
        max_distance = 0

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
                    if distances[neighbor] > max_distance:
                        max_distance = distances[neighbor]
                        furthest_node = neighbor
        return furthest_node, max_distance

    s = lcc[0]  
    t, max_distance = bfs(s)
    return s, t, max_distance

def prepare_residual_graph(edges, n):
    """
    Prepare the residual graph with capacities and correct costs.
    Reverse edges have negative costs.
    """
    residual_graph = defaultdict(dict)
    edge_costs = defaultdict(dict)

    for u, v, capacity, cost in edges:
        # Forward edge
        residual_graph[u][v] = residual_graph[u].get(v, 0) + capacity
        edge_costs[u][v] = cost

        # Reverse edge with zero initial capacity and negative cost
        residual_graph[v][u] = residual_graph[v].get(u, 0)
        edge_costs[v][u] = -cost

    # Ensure all nodes are present in the residual graph
    for node in range(1, n + 1):
        residual_graph[node]  # Initializes empty dict if not present

    return residual_graph, edge_costs


def ford_fulkerson(residual_graph, s, t):
    """
    Compute the maximum flow in a graph using the Ford-Fulkerson algorithm.

    Parameters:
    - residual_graph: Adjacency list of the residual graph with capacities.
    - s: Source node.
    - t: Sink node.

    Returns:
    - max_flow: The maximum flow from `s` to `t`.
    """
    max_flow = 0
    iteration = 0

    while True:
        iteration += 1
        parent = {}
        visited = set([s])
        queue = deque([s])

        found_path = False
        while queue and not found_path:
            node = queue.popleft()
            for neighbor, cap in residual_graph[node].items():
                if neighbor not in visited and cap > 0:
                    parent[neighbor] = node
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if neighbor == t:
                        found_path = True
                        break

        if not found_path:
            print(f"Ford-Fulkerson: No more augmenting paths found after {iteration-1} iterations.")
            break

        # Find bottleneck
        path_flow = float('inf')
        v = t
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, residual_graph[u][v])
            v = u

        # Update residual capacities
        v = t
        while v != s:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = u

        max_flow += path_flow
        print(f"Ford-Fulkerson Iteration {iteration}: Augmented flow by {path_flow}, Total flow now {max_flow}.")

    return max_flow

def bellman_ford(residual_graph, edge_costs, s, t):
    """
    Bellman-Ford for shortest path in terms of costs. 
    distance[node]: cost to reach node
    parent[node]: predecessor for shortest path

    Parameters:
    - residual_graph: Adjacency list of residual capacities.
    - edge_costs: Adjacency list of edge costs.
    - s: Source node.
    - t: Sink node.

    Returns:
    - parent: Parent dictionary to reconstruct path.
    - path_cost: Total cost of the path.
    """
    distance = {node: float('inf') for node in residual_graph}
    parent = {}
    distance[s] = 0

    print(f"Starting Bellman-Ford from source {s} to sink {t}.")

    for iteration in range(len(residual_graph) - 1):
        print(f"Bellman-Ford Iteration {iteration + 1}: Relaxing edges.")
        updated = False
        for u in residual_graph:
            for v in residual_graph[u]:
                if residual_graph[u][v] > 0:  # There's capacity
                    if distance[u] + edge_costs[u][v] < distance[v]:
                        distance[v] = distance[u] + edge_costs[u][v]
                        parent[v] = u
                        updated = True
                        print(f"Updated: distance[{v}] = {distance[v]}, parent[{v}] = {u}")
        if not updated:
            print("No updates in this iteration. Early stopping.")
            break

    print("Checking for negative cost cycles.")
    for u in residual_graph:
        for v in residual_graph[u]:
            if residual_graph[u][v] > 0 and distance[u] + edge_costs[u][v] < distance[v]:
                print(f"Negative cycle detected: edge {u} -> {v} with cost {edge_costs[u][v]}")
                raise ValueError("Graph contains a negative cost cycle.")

    print(f"Bellman-Ford completed. Distance to {t}: {distance[t]}")
    return parent, distance[t]

def successive_shortest_paths(residual_graph, edge_costs, s, t, d, n):
    """
    Successive Shortest Paths algorithm for min-cost flow.
    """
    total_flow = 0
    total_cost = 0
    paths = 0
    path_lengths = []

    print(f"Starting Successive Shortest Paths algorithm for demand: {d}")

    while d > 0:
        paths += 1
        parent = {}
        distance = {node: float('inf') for node in residual_graph}
        distance[s] = 0
        in_queue = {node: False for node in residual_graph}
        queue = deque([s])
        in_queue[s] = True

        # Bellman-Ford to find shortest path
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for v in residual_graph[u]:
                if residual_graph[u][v] > 0 and distance[u] + edge_costs[u][v] < distance[v]:
                    distance[v] = distance[u] + edge_costs[u][v]
                    parent[v] = u
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        if t not in parent:
            print("No augmenting path found. Terminating algorithm.")
            break

        # Reconstruct path
        path = []
        v = t
        while v != s:
            u = parent[v]
            path.append((u, v))
            v = u
        path_length = len(path)
        path_lengths.append(path_length)
        print(f"Found path with cost {distance[t]} and length {path_length}.")

        # Find bottleneck
        path_flow = min(residual_graph[u][v] for u, v in path)
        path_flow = min(path_flow, d)
        print(f"Bottleneck capacity on path: {path_flow}")

        # Update residual graph
        for u, v in path:
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow

        # Update total flow and cost
        total_flow += path_flow
        total_cost += distance[t] * path_flow
        d -= path_flow
        print(f"Iteration complete. Total flow: {total_flow}, Total cost: {total_cost}, Remaining demand: {d}")

    if d > 0:
        print("Unable to satisfy full demand. Flow algorithm terminated with unmet demand.")

    # Calculate metrics
    ML = sum(path_lengths) / len(path_lengths) if path_lengths else 0
    MPL = (ML / max(path_lengths)) if path_lengths else 0

    print(f"Successive Shortest Paths algorithm complete. Total flow: {total_flow}, Total cost: {total_cost}")
    print(f"Number of paths: {paths}, Mean length (ML): {ML}, Mean proportional length (MPL): {MPL}")

    return total_flow, total_cost, paths, ML, MPL


if __name__ == "__main__":
    scenarios = [
        {"graph": "graph1.txt", "n": 100, "r": 0.2, "upperCap": 8, "upperCost": 5},
        {"graph": "graph2.txt", "n": 200, "r": 0.2, "upperCap": 8, "upperCost": 5},
        {"graph": "graph3.txt", "n": 100, "r": 0.3, "upperCap": 8, "upperCost": 5},
        {"graph": "graph4.txt", "n": 200, "r": 0.3, "upperCap": 8, "upperCost": 5},
        {"graph": "graph5.txt", "n": 100, "r": 0.2, "upperCap": 64, "upperCost": 20},
        {"graph": "graph6.txt", "n": 200, "r": 0.2, "upperCap": 64, "upperCost": 20},
        {"graph": "graph7.txt", "n": 100, "r": 0.3, "upperCap": 64, "upperCost": 20},
        {"graph": "graph8.txt", "n": 200, "r": 0.3, "upperCap": 64, "upperCost": 20},
    ]

    # Store results for Table 1 and Table 2
    table1_data = []
    table2_data = []

    print("\nGenerating Table 1: Random Graph Characteristics...\n")
    for scenario in scenarios:
        print(f"Processing {scenario['graph']}...")
        # Load graph
        edges = load_graph_from_file(scenario["graph"])

        # Find LCC
        lcc = find_largest_connected_component(edges, scenario["n"])
        num_nodes_lcc = len(lcc)
        print(f"Largest Connected Component has {num_nodes_lcc} nodes.")

        # Select source and sink
        s, t, max_path_length = select_source_and_sink(lcc, edges)
        print(f"Selected source: {s}, sink: {t}, max path length: {max_path_length}")

        # Prepare residual graph and compute max flow
        residual_graph, _ = prepare_residual_graph(edges, scenario["n"])
        print("Computing maximum flow using Ford-Fulkerson...")
        fmax = ford_fulkerson(residual_graph, s, t)
        print(f"Maximum flow (fmax): {fmax}")

        # Compute LCC statistics
        out_degrees = defaultdict(int)
        in_degrees = defaultdict(int)
        num_edges_lcc = 0

        for u, v, _, _ in edges:
            if u in lcc and v in lcc:
                out_degrees[u] += 1
                in_degrees[v] += 1
                num_edges_lcc += 1

        max_out_degree = max(out_degrees.values(), default=0)
        max_in_degree = max(in_degrees.values(), default=0)
        avg_degree = num_edges_lcc / num_nodes_lcc if num_nodes_lcc > 0 else 0

        # Append results to Table 1
        table1_data.append({
            "Graph": scenario["graph"],
            "n": scenario["n"],
            "r": scenario["r"],
            "upperCap": scenario["upperCap"],
            "upperCost": scenario["upperCost"],
            "fmax": fmax,
            "|VLCC|": num_nodes_lcc,
            "Δout(LCC)": max_out_degree,
            "Δin(LCC)": max_in_degree,
            "k(LCC)": round(avg_degree, 4)
        })
        print(f"Completed processing {scenario['graph']}.\n")

    print("Finished generating Table 1.\n")

    print("\nTable 1: Random Graph Characteristics")
    header = f"{'Graph':<12} {'n':<8} {'r':<8} {'upperCap':<10} {'upperCost':<10} " \
             f"{'fmax':<10} {'|VLCC|':<10} {'Δout(LCC)':<12} {'Δin(LCC)':<12} {'k(LCC)':<10}"
    print(header)
    print("-" * len(header))
    for row in table1_data:
        print(f"{row['Graph']:<12} {row['n']:<8} {row['r']:<8} {row['upperCap']:<10} "
              f"{row['upperCost']:<10} {row['fmax']:<10} {row['|VLCC|']:<10} "
              f"{row['Δout(LCC)']:<12} {row['Δin(LCC)']:<12} {row['k(LCC)']:<10}")

    print("\nGenerating Table 2: Results for Minimum-Cost Flow Algorithms...\n")
    for scenario in scenarios:
        print(f"Processing {scenario['graph']} for Table 2...")
        # Load graph
        edges = load_graph_from_file(scenario["graph"])

        # Find LCC
        lcc = find_largest_connected_component(edges, scenario["n"])
        num_nodes_lcc = len(lcc)

        # Select source and sink
        s, t, max_path_length = select_source_and_sink(lcc, edges)

        # Prepare residual graph and compute max flow
        residual_graph, edge_costs = prepare_residual_graph(edges, scenario["n"])
        print("Computing maximum flow using Ford-Fulkerson...")
        fmax = ford_fulkerson(residual_graph.copy(), s, t)  # Use a copy to preserve original residual_graph
        d = int(0.95 * fmax)
        print(f"Computed fmax: {fmax}, setting demand d to 0.95 * fmax = {d}")

        # Run SSP
        flow, cost, paths, ML, MPL = successive_shortest_paths(residual_graph, edge_costs, s, t, d, scenario["n"])

        # Append SSP results to Table 2
        table2_data.append({
            "Algorithm": "SSP",
            "Graph": scenario["graph"],
            "f": flow,
            "MC": cost,
            "paths": paths,
            "ML": round(ML, 4),
            "MPL": round(MPL, 4),
        })
        print(f"Completed SSP for {scenario['graph']}.\n")

    print("Finished generating Table 2.\n")

    print("\nTable 2: Results for Minimum-Cost Flow Algorithms")
    header = f"{'Algorithm':<10} {'Graph':<12} {'f':<8} {'MC':<10} {'paths':<8} {'ML':<10} {'MPL':<10}"
    print(header)
    print("-" * len(header))
    for row in table2_data:
        print(f"{row['Algorithm']:<10} {row['Graph']:<12} {row['f']:<8} {row['MC']:<10} "
              f"{row['paths']:<8} {row['ML']:<10} {row['MPL']:<10}")
