import numpy as np

def floyd_warshall(graph, print_steps=False):
    """
    Implements the Floyd-Warshall algorithm to find the shortest paths
    between all pairs of vertices in a weighted graph.

    :param graph: A 2D numpy array representing the adjacency matrix of the graph.
                  If there is no edge between two vertices, the corresponding entry should be a large number, for example, 999.
                  A 2D numpy array representing the predecessor martrix, which denotes the intermediate vertices in the shortest paths.
    :return: A 2D numpy array containing the shortest path distances between all pairs of vertices.
             A 2D numpy array containing the predecessor matrix. 1-indexed.
    """

    num_vertices = graph.shape[0]
    dist = np.copy(graph)
    Pr = np.array([np.arange(1,graph.shape[0]+1,1) for _ in range(graph.shape[0])], dtype=int)

    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    Pr[i][j]   = k+1  # Store the predecessor vertex (1-indexed)
        if print_steps:
            print(f"Intermediate vertex {k}:")
            print(dist)

    return dist, Pr

# Example usage:
if __name__ == "__main__":
    # Create a graph represented as an adjacency matrix
    inf = 999
    graph = np.array([[0, 4, -3, inf],
                      [-3,0,-7,inf],
                      [inf,10,0,3],
                      [5,6,6,0]], dtype=int)
    
    shortest_paths, Pr = floyd_warshall(graph, print_steps=True)
    print("Shortest path distances between all pairs of vertices:")
    print(shortest_paths)   
    print("Predecessor matrix:")
    print(Pr)

