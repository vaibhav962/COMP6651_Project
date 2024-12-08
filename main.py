import logging
import random
import math
from collections import defaultdict, deque
import os

logging.basicConfig(
    filename='graph_analysis.log',  
    filemode='w', 
    level=logging.INFO,            
    format='%(message)s'           
)

def generate_graph_with_sink_source(n_nodes, radius, max_capacity, max_cost, output_file):
    """
    Generate a directed graph with random nodes and edges.

    Parameters:
        n_nodes (int): Number of nodes in the graph.
        radius (float): Maximum distance between nodes to connect them.
        max_capacity (int): Maximum capacity of an edge.
        max_cost (int): Maximum cost of an edge.
        output_file (str): File to save the generated graph.

    Returns:
        None
    """
    node_coordinates = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(n_nodes)]
    edge_list = []
    edge_set = set()

    for source in range(n_nodes):
        for target in range(n_nodes):
            if source != target:  # No self-loops
                distance = math.sqrt((node_coordinates[source][0] - node_coordinates[target][0])**2 + 
                                     (node_coordinates[source][1] - node_coordinates[target][1])**2)
                if distance <= radius:
                    rand = random.random()
                    if rand < 0.3:
                        if (source + 1, target + 1) not in edge_set and (target + 1, source + 1) not in edge_set:
                            capacity = random.randint(1, max_capacity)
                            cost = random.randint(1, max_cost)
                            edge_list.append((source + 1, target + 1, capacity, cost))
                            edge_set.add((source + 1, target + 1))
                    elif rand < 0.6:
                        if (target + 1, source + 1) not in edge_set and (source + 1, target + 1) not in edge_set:
                            capacity = random.randint(1, max_capacity)
                            cost = random.randint(1, max_cost)
                            edge_list.append((target + 1, source + 1, capacity, cost))
                            edge_set.add((target + 1, source + 1))

    with open(output_file, "w") as file:
        for edge in edge_list:
            file.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]}\n")

    logging.info(f"Graph saved to {output_file} with {len(edge_list)} edges.")

def load_graph_from_file(file_path):
    """
    Load a graph from a file in EDGES format.

    Parameters:
        file_path (str): Path to the graph file.

    Returns:
        list: A list of edges represented as tuples (source, target, capacity, cost).
    """
    edge_list = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            source, target, capacity, cost = map(int, parts)
            edge_list.append((source, target, capacity, cost))
    logging.info(f"Loaded graph from {file_path} with {len(edge_list)} edges.")
    return edge_list

def find_largest_connected_component(edges, n_nodes):
    """
    Find the largest connected component (LCC) in the graph.

    Parameters:
        edges (list): List of edges in the graph.
        n_nodes (int): Number of nodes in the graph.

    Returns:
        list: List of nodes in the largest connected component.
    """
    adjacency_list = defaultdict(list)
    for source, target, _, _ in edges:
        adjacency_list[source].append(target)
        adjacency_list[target].append(source)

    visited_nodes = set()
    largest_component = []

    def depth_first_search(node, component):
        visited_nodes.add(node)
        component.append(node)
        for neighbor in adjacency_list[node]:
            if neighbor not in visited_nodes:
                depth_first_search(neighbor, component)

    for node in range(1, n_nodes + 1):
        if node not in visited_nodes:
            component = []
            depth_first_search(node, component)
            if len(component) > len(largest_component):
                largest_component = component

    lcc_representation = " -> ".join(map(str, largest_component))
    logging.info(f"Largest Connected Component (LCC): {lcc_representation}")
    return largest_component

def select_source_and_sink(lcc, edges):
    """
    Select source and sink nodes from the Largest Connected Component (LCC).

    Parameters:
        lcc (list): List of nodes in the largest connected component.
        edges (list): List of edges in the graph.

    Returns:
        tuple: (source, sink, max_path_length) where source and sink are nodes, and max_path_length is the distance between them.
    """
    adjacency_list = defaultdict(list)
    for source, target, _, _ in edges:
        if source in lcc and target in lcc:
            adjacency_list[source].append(target)

    def bfs_to_find_furthest_node(start_node):
        distances = {}
        queue = deque([start_node])
        distances[start_node] = 0
        furthest_node = start_node
        max_distance = 0

        while queue:
            current_node = queue.popleft()
            for neighbor in adjacency_list[current_node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
                    if distances[neighbor] > max_distance:
                        max_distance = distances[neighbor]
                        furthest_node = neighbor
        return furthest_node, max_distance

    source = lcc[0]
    sink, max_path_length = bfs_to_find_furthest_node(source)
    return source, sink, max_path_length

def prepare_residual_graph(edges):
    """
    Prepare the residual graph with capacities and costs.

    Parameters:
        edges (list): List of edges in the graph.

    Returns:
        tuple: (residual_graph, edge_cost_map), where both are dictionaries.
    """
    residual_graph = defaultdict(dict)
    edge_cost_map = defaultdict(dict)

    for source, target, capacity, cost in edges:
        residual_graph[source][target] = residual_graph[source].get(target, 0) + capacity
        edge_cost_map[source][target] = cost

        residual_graph[target][source] = residual_graph[target].get(source, 0)
        edge_cost_map[target][source] = -cost

    logging.info(f"Residual graph generated")
    return residual_graph, edge_cost_map

def ford_fulkerson_max_flow(residual_graph, source, sink):
    """
    Compute the maximum flow in a graph using the Ford-Fulkerson algorithm.

    Parameters:
        residual_graph (dict): Residual graph with capacities.
        source (int): Source node.
        sink (int): Sink node.

    Returns:
        int: Maximum flow from source to sink.
    """
    max_flow = 0
    iteration_count = 0

    while True:
        iteration_count += 1
        parent_map = {}
        visited_nodes = set([source])
        queue = deque([source])

        found_path = False
        while queue and not found_path:
            current_node = queue.popleft()
            for neighbor, capacity in residual_graph[current_node].items():
                if neighbor not in visited_nodes and capacity > 0:
                    parent_map[neighbor] = current_node
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
                    if neighbor == sink:
                        found_path = True
                        break

        if not found_path:
            logging.info(f"Ford-Fulkerson: No more augmenting paths found after {iteration_count - 1} iterations.")
            break

        # Find bottleneck capacity
        augmenting_path = []
        bottleneck_flow = float('inf')
        current_node = sink
        while current_node != source:
            augmenting_path.append(current_node)
            previous_node = parent_map[current_node]
            bottleneck_flow = min(bottleneck_flow, residual_graph[previous_node][current_node])
            current_node = previous_node
        augmenting_path.append(source)
        augmenting_path.reverse()

        # Update residual capacities
        current_node = sink
        while current_node != source:
            previous_node = parent_map[current_node]
            residual_graph[previous_node][current_node] -= bottleneck_flow
            residual_graph[current_node][previous_node] += bottleneck_flow
            current_node = previous_node

        max_flow += bottleneck_flow
        augmenting_path_str = " -> ".join(map(str, augmenting_path))
        logging.info(f"Ford-Fulkerson Iteration {iteration_count}: Augmented flow by {bottleneck_flow}, "
                     f"Total flow now {max_flow}, Augmenting Path: {augmenting_path_str}")

    return max_flow

def successive_shortest_paths(edges, source, sink, required_flow):
    """
    Compute the minimum-cost flow using the Successive Shortest Paths algorithm.

    Parameters:
        edges (list): List of edges in the graph.
        source (int): Source node.
        sink (int): Sink node.
        required_flow (int): Required flow to achieve.

    Returns:
        tuple: (total_cost, total_flow, path_count, avg_path_length, avg_proportional_length)
        - total_cost: Minimum cost of achieving the flow.
        - total_flow: Total flow achieved.
        - path_count: Number of augmenting paths used.
        - avg_path_length: Average length of the paths.
        - avg_proportional_length: Average proportional path length.
    """
    residual_graph, edge_cost_map = prepare_residual_graph(edges)

    total_flow = 0
    total_cost = 0
    path_count = 0
    path_lengths = []

    def log_residual_graph():
        logging.debug("Current Residual Graph:")
        for source_node in residual_graph:
            for target_node in residual_graph[source_node]:
                if residual_graph[source_node][target_node] > 0:
                    logging.debug(f"  Edge {source_node} -> {target_node}: capacity = {residual_graph[source_node][target_node]}, cost = {edge_cost_map[source_node][target_node]}")

    def find_shortest_path_bellman_ford(start_node, end_node):
        """
        Find the shortest path in terms of cost using Bellman-Ford.

        Parameters:
            start_node (int): Start node for the path.
            end_node (int): End node for the path.

        Returns:
            tuple: (path, cost) where path is the shortest path and cost is its total cost.
        """
        logging.info("Running Bellman-Ford for shortest path calculation...")
        distances = {node: float('inf') for node in residual_graph}
        parents = {node: None for node in residual_graph}
        distances[start_node] = 0

        # Relaxation for V-1 iterations
        for iteration in range(len(residual_graph) - 1):
            logging.debug(f"Iteration {iteration + 1} of Bellman-Ford relaxation...")
            for source_node in residual_graph:
                for target_node in residual_graph[source_node]:
                    if residual_graph[source_node][target_node] > 0:  # Only consider edges with positive capacity
                        new_distance = distances[source_node] + edge_cost_map[source_node][target_node]
                        if new_distance < distances[target_node]:
                            distances[target_node] = new_distance
                            parents[target_node] = source_node

        # Reconstruct path
        path = []
        current_node = end_node
        while current_node and current_node != start_node:
            path.append(current_node)
            current_node = parents[current_node]

        if current_node == start_node:
            path.append(start_node)
            path.reverse()
            logging.info(f"Shortest path found: {path} with cost {distances[end_node]}")
            return path, distances[end_node]

        logging.info("No augmenting path found.")
        return None, float('inf')  # No path found

    def push_flow(path, flow_to_push):
        """
        Push a specified flow along a given path.

        Parameters:
            path (list): Path along which to push the flow.
            flow_to_push (int): Amount of flow to push.

        Returns:
            None
        """
        logging.info(f"Pushing flow along path: {path} with flow {flow_to_push}")
        for i in range(len(path) - 1):
            source_node, target_node = path[i], path[i + 1]
            residual_graph[source_node][target_node] -= flow_to_push
            residual_graph[target_node][source_node] += flow_to_push

        log_residual_graph()

    while required_flow > 0:
        path, path_cost = find_shortest_path_bellman_ford(source, sink)
        if not path:
            break  # No more augmenting paths available

        max_path_flow = min(residual_graph[source_node][target_node] for source_node, target_node in zip(path, path[1:]))
        flow_to_push = min(max_path_flow, required_flow)

        push_flow(path, flow_to_push)
        total_flow += flow_to_push
        total_cost += flow_to_push * path_cost
        path_count += 1
        path_lengths.append(len(path) - 1)  # Number of edges in the path

        required_flow -= flow_to_push

        logging.info(f"After iteration {path_count}: Total flow: {total_flow}, Total cost: {total_cost}, "
                     f"Remaining flow: {required_flow}")

    if required_flow > 0:
        logging.warning("Failed to achieve the desired flow.")
        return None, -1, None, None, None  # Failure to achieve the desired flow

    avg_path_length = sum(path_lengths) / path_count if path_count > 0 else 0
    max_path_length = max(path_lengths, default=1)
    avg_proportional_length = sum(length / max_path_length for length in path_lengths) / path_count

    logging.info("Successive Shortest Paths completed successfully.")
    logging.info(f"Total flow achieved: {total_flow}")
    logging.info(f"Minimum cost of flow: {total_cost}")
    logging.info(f"Number of augmenting paths used: {path_count}")
    logging.info(f"Average path length: {avg_path_length}")
    logging.info(f"Average proportional path length: {avg_proportional_length}")

    return total_cost, total_flow, path_count, avg_path_length, avg_proportional_length

if __name__ == "__main__":
    scenarios = [
        {"graph": "graph1.txt", "n_nodes": 20, "radius": 0.2, "max_capacity": 6, "max_cost": 5},
        {"graph": "graph2.txt", "n_nodes": 200, "radius": 0.2, "max_capacity": 8, "max_cost": 5},
        {"graph": "graph3.txt", "n_nodes": 100, "radius": 0.3, "max_capacity": 8, "max_cost": 5},
        {"graph": "graph4.txt", "n_nodes": 200, "radius": 0.3, "max_capacity": 8, "max_cost": 5},
        {"graph": "graph5.txt", "n_nodes": 100, "radius": 0.2, "max_capacity": 64, "max_cost": 20},
        {"graph": "graph6.txt", "n_nodes": 200, "radius": 0.2, "max_capacity": 64, "max_cost": 20},
        {"graph": "graph7.txt", "n_nodes": 100, "radius": 0.3, "max_capacity": 64, "max_cost": 20},
        {"graph": "graph8.txt", "n_nodes": 200, "radius": 0.3, "max_capacity": 64, "max_cost": 20},
    ]

    # Step 1: Generate graphs only if not already present
    logging.info("Checking and Generating Graph Files...")
    for scenario in scenarios:
        graph_file = scenario["graph"]
        if not os.path.exists(graph_file):
            logging.info(f"Generating graph: {graph_file} (n={scenario['n_nodes']}, r={scenario['radius']})")
            generate_graph_with_sink_source(
                n_nodes=scenario["n_nodes"],
                radius=scenario["radius"],
                max_capacity=scenario["max_capacity"],
                max_cost=scenario["max_cost"],
                output_file=graph_file
            )
        else:
            logging.info(f"Graph file {graph_file} already exists. Skipping generation.")
    logging.info("Graph file check and generation completed.")

    # Store results for Table 1 and Table 2
    table1_data = []
    table2_data = []

    logging.info("\n<---- Processing graphs ---->\n")
    for scenario in scenarios:
        logging.info(f"Processing {scenario['graph']}...")
        print(f"Processing {scenario['graph']}...")
        # Load graph
        edges = load_graph_from_file(scenario["graph"])

        # Find LCC
        largest_component = find_largest_connected_component(edges, scenario["n_nodes"])
        num_nodes_lcc = len(largest_component)
        logging.info(f"Largest Connected Component has {num_nodes_lcc} nodes.")

        # Select source and sink
        source, sink, max_path_length = select_source_and_sink(largest_component, edges)
        logging.info(f"Selected source: {source}, sink: {sink}, max path length between source and sink: {max_path_length}")

        # Prepare residual graph and compute max flow
        residual_graph, edge_cost_map = prepare_residual_graph(edges)

        logging.info("Computing maximum flow using Ford-Fulkerson...")
        max_flow = ford_fulkerson_max_flow(residual_graph, source, sink)
        logging.info(f"Maximum flow (fmax): {max_flow}")

        # Compute LCC statistics
        out_degrees = defaultdict(int)
        in_degrees = defaultdict(int)
        sum_degrees = 0

        for source_node, target_node, _, _ in edges:
            if source_node in largest_component and target_node in largest_component:
                out_degrees[source_node] += 1
                in_degrees[target_node] += 1

        for node in largest_component:
            degree = out_degrees[node] + in_degrees[node]
            sum_degrees += degree

        max_out_degree = max(out_degrees.values(), default=0)
        max_in_degree = max(in_degrees.values(), default=0)
        avg_degree = sum_degrees / num_nodes_lcc if num_nodes_lcc > 0 else 0

        # Append results to Table 1
        table1_data.append({
            "Graph": scenario["graph"],
            "n_nodes": scenario["n_nodes"],
            "radius": scenario["radius"],
            "max_capacity": scenario["max_capacity"],
            "max_cost": scenario["max_cost"],
            "max_flow": max_flow,
            "|LCC_nodes|": num_nodes_lcc,
            "max_out_degree": max_out_degree,
            "max_in_degree": max_in_degree,
            "avg_degree": round(avg_degree, 4)
        })

        # Minimum-cost flow algorithms
        desired_flow = math.floor(0.95 * max_flow)
        logging.info(f"\nRunning minimum-cost flow algorithms for demand d = {desired_flow}...")

        logging.info(f"\n*** Successive Shortest Paths Algorithm ***\n")

        # Successive Shortest Paths Algorithm
        min_cost, achieved_flow, num_paths, avg_length, avg_proportional_length = successive_shortest_paths(edges, source, sink, desired_flow)
        table2_data.append({
            "Algorithm": "SSP",
            "Graph": scenario["graph"],
            "achieved_flow": achieved_flow,
            "min_cost": min_cost,
            "num_paths": num_paths,
            "avg_length": round(avg_length, 4),
            "avg_proportional_length": round(avg_proportional_length, 4),
        })

        logging.info(f"Completed processing {scenario['graph']}.")

    # Print Table 1
    print("\nTable 1: Random Graph Characteristics")
    header = f"{'Graph':<12} {'n_nodes':<10} {'radius':<8} {'max_capacity':<12} {'max_cost':<10} " \
             f"{'max_flow':<10} {'|LCC_nodes|':<12} {'max_out_degree':<15} {'max_in_degree':<15} {'avg_degree':<12}"
    print(header)
    print("-" * len(header))
    for row in table1_data:
        print(f"{row['Graph']:<12} {row['n_nodes']:<10} {row['radius']:<8} {row['max_capacity']:<12} "
              f"{row['max_cost']:<10} {row['max_flow']:<10} {row['|LCC_nodes|']:<12} "
              f"{row['max_out_degree']:<15} {row['max_in_degree']:<15} {row['avg_degree']:<12}")

    # Print Table 2
    print("\nTable 2: Results for Minimum-Cost Flow Algorithms")
    header = f"{'Algorithm':<10} {'Graph':<12} {'achieved_flow':<15} {'min_cost':<10} {'num_paths':<12} {'avg_length':<12} {'avg_proportional_length':<20}"
    print(header)
    print("-" * len(header))
    for row in table2_data:
        print(f"{row['Algorithm']:<10} {row['Graph']:<12} {row['achieved_flow']:<15} {row['min_cost']:<10} "
              f"{row['num_paths']:<12} {row['avg_length']:<12} {row['avg_proportional_length']:<20}")
