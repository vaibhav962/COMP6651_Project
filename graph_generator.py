import random
import math

def generate_sink_source_graph(n, r, upper_cap, upper_cost, file_name):
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

    with open(file_name, "w") as f:
        for edge in edges:
            f.write(f"{edge['from']} {edge['to']} {edge['capacity']} {edge['cost']}\n")

    print(f"Graph with {n} vertices and {len(edges)} edges saved to {file_name}.")

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
