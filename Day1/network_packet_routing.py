"""
Problem 6: Network Packet Routing
Goal: Find lowest-cost route from router A to router G
Algorithms: UCS, A*, Greedy
"""

import heapq
from typing import List, Tuple, Optional

class NetworkRouter:
    """
    Network routing with latency costs.
    Routers are nodes, connections are weighted edges.
    """
    
    def __init__(self):
        # Network graph: router -> [(neighbor, latency)]
        self.network = {
            'A': [('B', 4), ('C', 2)],
            'B': [('A', 4), ('C', 1), ('D', 5)],
            'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
            'D': [('B', 5), ('C', 8), ('E', 2), ('F', 6)],
            'E': [('C', 10), ('D', 2), ('F', 3)],
            'F': [('D', 6), ('E', 3), ('G', 1)],
            'G': [('F', 1)]
        }
    
    def get_neighbors(self, router: str) -> List[Tuple[str, int]]:
        """Return list of (neighbor, latency) for given router"""
        return self.network.get(router, [])
    
    def heuristic(self, router: str, goal: str = 'G') -> float:
        """
        Heuristic for A* and Greedy.
        In real networks, this could be geographic distance or estimated latency.
        Here we use precomputed approximate distances to goal.
        """
        # Approximate minimum remaining latency to G
        approx_dist = {'A': 10, 'B': 7, 'C': 8, 'D': 4, 'E': 3, 'F': 1, 'G': 0}
        return approx_dist.get(router, float('inf'))
    
    def ucs(self, start: str, goal: str) -> Tuple[Optional[List[str]], float]:
        """
        Uniform Cost Search (Dijkstra's algorithm):
        - Finds path with minimum total latency
        - Guarantees optimal solution
        - Does not use heuristic
        - Equivalent to Dijkstra's algorithm
        """
        # Priority queue: (total_latency, current_router, path)
        pq = [(0, start, [start])]
        visited = {}  # router -> best_latency_so_far
        
        while pq:
            latency, router, path = heapq.heappop(pq)
            
            # Skip if we already found a better path
            if router in visited and visited[router] <= latency:
                continue
            visited[router] = latency
            
            if router == goal:
                return path, latency
            
            for neighbor, link_latency in self.get_neighbors(router):
                new_latency = latency + link_latency
                if neighbor not in visited or visited[neighbor] > new_latency:
                    heapq.heappush(pq, (new_latency, neighbor, path + [neighbor]))
        
        return None, float('inf')
    
    def a_star(self, start: str, goal: str = 'G') -> Tuple[Optional[List[str]], float]:
        """
        A* Search for routing:
        - f(n) = g(n) + h(n)
        - g(n) = actual latency from start to n
        - h(n) = estimated remaining latency to goal
        - More efficient than UCS when heuristic is good
        - Guarantees optimal path if heuristic is admissible
        """
        pq = [(self.heuristic(start, goal), 0, start, [start])]
        visited = {}  # router -> best g_score
        
        while pq:
            f_score, g_score, router, path = heapq.heappop(pq)
            
            if router in visited and visited[router] <= g_score:
                continue
            visited[router] = g_score
            
            if router == goal:
                return path, g_score
            
            for neighbor, latency in self.get_neighbors(router):
                new_g = g_score + latency
                new_f = new_g + self.heuristic(neighbor, goal)
                if neighbor not in visited or visited[neighbor] > new_g:
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
        
        return None, float('inf')
    
    def greedy(self, start: str, goal: str = 'G') -> Tuple[Optional[List[str]], float]:
        """
        Greedy Best-First Search for routing:
        - Uses only heuristic h(n)
        - Expands node closest to goal
        - Faster but NOT optimal
        - May take longer paths or fail to find goal
        """
        pq = [(self.heuristic(start, goal), start, [start], 0)]
        visited = set()
        
        while pq:
            _, router, path, latency = heapq.heappop(pq)
            
            if router in visited:
                continue
            visited.add(router)
            
            if router == goal:
                return path, latency
            
            for neighbor, link_latency in self.get_neighbors(router):
                if neighbor not in visited:
                    heapq.heappush(pq, (self.heuristic(neighbor, goal), neighbor,
                                       path + [neighbor], latency + link_latency))
        
        return None, float('inf')
    
    def print_network(self):
        """Print the network topology"""
        print("Network Topology (router: [(neighbor, latency)]):")
        for router, neighbors in self.network.items():
            print(f"  {router}: {neighbors}")


# ==================== Main Execution ====================
if __name__ == "__main__":
    network = NetworkRouter()
    start = 'A'
    goal = 'G'
    
    print("=" * 60)
    print(f"Problem 6: Network Packet Routing from {start} to {goal}")
    print("=" * 60)
    
    network.print_network()
    
    # UCS
    print("\n--- UCS (Uniform Cost Search) ---")
    path, latency = network.ucs(start, goal)
    if path:
        print(f"Optimal route: {' -> '.join(path)}")
        print(f"Total latency: {latency} ms")
    else:
        print("No route found")
    
    # A*
    print("\n--- A* Search ---")
    path, latency = network.a_star(start, goal)
    if path:
        print(f"Route: {' -> '.join(path)}")
        print(f"Total latency: {latency} ms")
    else:
        print("No route found")
    
    # Greedy
    print("\n--- Greedy Best-First Search ---")
    path, latency = network.greedy(start, goal)
    if path:
        print(f"Route: {' -> '.join(path)}")
        print(f"Total latency: {latency} ms")
        print("Note: Greedy may not find optimal path!")
    else:
        print("No route found")
    
    # Compare all routes
    print("\n" + "=" * 60)
    print("Comparison of Algorithms:")
    print("=" * 60)
    
    _, ucs_latency = network.ucs(start, goal)
    _, a_star_latency = network.a_star(start, goal)
    _, greedy_latency = network.greedy(start, goal)
    
    print(f"UCS:    {ucs_latency} ms (optimal)")
    print(f"A*:     {a_star_latency} ms (optimal)")
    print(f"Greedy: {greedy_latency} ms (may not be optimal)")
    
    # Alternative route test
    print("\n" + "-" * 40)
    print("Testing different start-goal pairs:")
    print("-" * 40)
    
    test_pairs = [('A', 'F'), ('B', 'G'), ('C', 'F')]
    for s, g in test_pairs:
        path, latency = network.a_star(s, g)
        if path:
            print(f"{s} -> {g}: {' -> '.join(path)} (latency: {latency} ms)")