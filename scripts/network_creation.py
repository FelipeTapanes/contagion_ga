import pandas as pd

K = 4
ga_matrix = pd.read_csv("data/ga_matrix_seconds.csv", index_col=0)

towns = ga_matrix.index.tolist()

edges = set()

for t in towns:
    row = ga_matrix.loc[t].drop(t)
    nearest = row.nsmallest(K)

    for n in nearest.index:
        edges.add(frozenset([t, n]))

edges_list = []

for link in edges:
    a, b = tuple(link)
    weight = min(ga_matrix.loc[a, b], ga_matrix.loc[b, a])
    edges_list.append((a, b, weight))

edges_df = pd.DataFrame(edges_list, columns=["node_a", "node_b", "seconds"])
edges_df["minutes"] = edges_df["seconds"] / 60.0

edges_df.to_csv("data/edges_k4.csv", index=False)
