# Graph Traversal

Graphs are made up of vertices (nodes) and edges (connections between
nodes), and traversal algorithms visit vertices in a systematic order. The
two fundamental traversal strategies are breadth-first search (BFS) and
depth-first search (DFS), and the choice between them depends on what the
problem actually needs.

## Breadth-First Search

BFS explores a graph level by level: it visits all neighbors of the start
node before moving on to their neighbors, using a queue (FIFO) to track
which node to visit next. This level-by-level exploration means BFS finds
the shortest path between two nodes in an unweighted graph, measured in
number of edges - the first time BFS reaches the target node is guaranteed
to be via a shortest path. BFS runs in O(V + E) time, where V is the number
of vertices and E is the number of edges, since each vertex and edge is
processed at most once.

## Depth-First Search

DFS explores as far as possible along each branch before backtracking,
using a stack (either explicit or via recursion) to track the path back.
DFS is the natural choice for problems like detecting cycles, topological
sorting of a directed acyclic graph (DAG), and finding connected components,
where the goal is to explore structure rather than find shortest paths. DFS
also runs in O(V + E) time, matching BFS, but its memory usage pattern
differs: DFS's stack depth is bounded by the longest path in the graph,
while BFS's queue can hold an entire "frontier" of nodes at once, which can
be much larger in a wide, shallow graph.

## Dijkstra's Algorithm

When edges have weights and you need the shortest path by total weight
rather than edge count, BFS no longer applies directly. Dijkstra's algorithm
generalizes BFS to weighted graphs (with non-negative weights) by replacing
the FIFO queue with a priority queue ordered by current best-known distance,
always expanding the closest unvisited node next. Using a binary heap for
the priority queue gives Dijkstra's algorithm O((V + E) log V) time
complexity. If the graph can have negative edge weights, Dijkstra's
algorithm is no longer correct, and the Bellman-Ford algorithm - which runs
in O(V * E) time but correctly handles negative weights and can detect
negative cycles - must be used instead.

## Choosing Between BFS and DFS

As a rule of thumb: use BFS when you need the shortest path in an
unweighted graph, or when you need to explore the graph in order of
distance from the start (e.g. finding all nodes within k steps). Use DFS
when you need to explore full paths, detect cycles, perform a topological
sort, or when memory is a concern in a wide graph, since DFS's stack-based
memory footprint is typically smaller than BFS's queue-based one in graphs
with high branching factor.
