'''
Docstring for scripts.network_creation
Reads the CSV file with the travel times between county towns and creates a new 
CSV file with the edges of the graph, where each edge is a pair of county towns 
and the weight is the travel time in seconds and minutes. Only the K nearest neighbors 
for each town are included in the graph to keep it sparse.
'''
import pandas as pd
K = 4

# Load the travel times between county towns from the CSV file
ga_matrix = pd.read_csv("data/ga_matrix_seconds.csv", index_col=0)

# Get the list of county towns from the index of the DataFrame
towns = ga_matrix.index.tolist()

# Create a set to store the edges of the graph
edges = set()

# For each town, find the K nearest neighbors based on travel time and add the edges to the set
for t in towns:
    row = ga_matrix.loc[t].drop(t)
    nearest = row.nsmallest(K)

    for n in nearest.index:
        edges.add(frozenset([t, n]))

# Convert the set of edges to a list of tuples and create a new DataFrame with the edges and their weights
edges_list = []
for link in edges:
    a, b = tuple(link)
    weight = min(ga_matrix.loc[a, b], ga_matrix.loc[b, a])
    edges_list.append((a, b, weight))

# Save the edges to a new CSV file with the travel times in seconds and another CSV file with the travel times in minutes
edges_df = pd.DataFrame(edges_list, columns=["node_a", "node_b", "seconds"])
edges_df["minutes"] = edges_df["seconds"] / 60.0
edges_df.to_csv("data/edges_k4.csv", index=False)
