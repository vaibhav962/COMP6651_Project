import random
import math
import os

def generate_sink_source_graph(n, r, upper_cap, upper_cost, file_name):
    """
    Generate a random weighted directed Euclidean source-sink graph.

    Parameters:
        n (int): Number of vertices.
        r (float): Maximum distance between nodes sharing an edge.
        upper_cap (int): Maximum capacity value for edges.
        upper_cost (int): Maximum cost value for edges.
        file_name (str): File name to save the generated graph in EDGES format.

    Returns:
        None
    """
    vertices = [{"id": i, "x": random.uniform(0, 1), "y": random.uniform(0, 1)} for i in range(n)]
    edges = []

    for u in vertices:
        for v in vertices:
            if u["id"] != v["id"]:
                distance = math.sqrt((u["x"] - v["x"]) ** 2 + (u["y"] - v["y"]) ** 2)
                if distance <= r:
                    rand = random.uniform(0, 1)
                    if rand < 0.3:
                        if not any(e for e in edges if e["from"] == u["id"] and e["to"] == v["id"]):
                            edges.append({"from": u["id"], "to": v["id"],
                                          "capacity": random.randint(1, upper_cap),
                                          "cost": random.randint(1, upper_cost)})
                    elif rand < 0.6:
                        if not any(e for e in edges if e["from"] == v["id"] and e["to"] == u["id"]):
                            edges.append({"from": v["id"], "to": u["id"],
                                          "capacity": random.randint(1, upper_cap),
                                          "cost": random.randint(1, upper_cost)})

    os.makedirs("graphs", exist_ok=True)
    with open(os.path.join("graphs", file_name), "w") as f:
        for edge in edges:
            f.write(f"{edge['from']} {edge['to']} {edge['capacity']} {edge['cost']}\n")

    print(f"Graph with {n} vertices and {len(edges)} edges saved to graphs/{file_name}.")

# Generate graphs for the 8 simulations
simulations = [
    (100, 0.2, 8, 5),
    (200, 0.2, 8, 5),
    (100, 0.3, 8, 5),
    (200, 0.3, 8, 5),
    (100, 0.2, 64, 20),
    (200, 0.2, 64, 20),
    (100, 0.3, 64, 20),
    (200, 0.3, 64, 2)
]

for i, (n, r, upper_cap, upper_cost) in enumerate(simulations, start=1):
    file_name = f"graph{i}.txt"
    generate_sink_source_graph(n, r, upper_cap, upper_cost, file_name)
