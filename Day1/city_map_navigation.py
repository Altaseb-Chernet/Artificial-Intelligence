"""
Problem 1: City Map Navigation (Route Planning)
Goal: Find shortest/fastest path from Arad to Bucharest
Algorithms: BFS, UCS, A*, Greedy, Bidirectional Search
"""

from collections import deque
import heapq
from typing import List, Tuple, Optional, Dict

# Romanian map with cities and road distances
ROMANIA_MAP = {
    'Arad': [('Zerind', 75), ('Sibiu', 140), ('Timisoara', 118)],
    'Zerind': [('Arad', 75), ('Oradea', 71)],
    'Oradea': [('Zerind', 71), ('Sibiu', 151)],
    'Sibiu': [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu Vilcea', 80)],
    'Timisoara': [('Arad', 118), ('Lugoj', 111)],
    'Lugoj': [('Timisoara', 111), ('Mehadia', 70)],
    'Mehadia': [('Lugoj', 70), ('Drobeta', 75)],
    'Drobeta': [('Mehadia', 75), ('Craiova', 120)],
    'Craiova': [('Drobeta', 120), ('Rimnicu Vilcea', 146), ('Pitesti', 138)],
    'Rimnicu Vilcea': [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)],
    'Fagaras': [('Sibiu', 99), ('Bucharest', 211)],
    'Pitesti': [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)],
    'Bucharest': [('Fagaras', 211), ('Pitesti', 101), ('Giurgiu', 90), ('Urziceni', 85)],
    'Giurgiu': [('Bucharest', 90)],
    'Urziceni': [('Bucharest', 85), ('Hirsova', 98), ('Vaslui', 142)],
    'Hirsova': [('Urziceni', 98), ('Eforie', 86)],
    'Eforie': [('Hirsova', 86)],
    'Vaslui': [('Urziceni', 142), ('Iasi', 92)],
    'Iasi': [('Vaslui', 92), ('Neamt', 87)],
    'Neamt': [('Iasi', 87)]
}

# Straight-line distances to Bucharest (heuristic for A* and Greedy)
STRAIGHT_LINE_TO_BUCHAREST = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242,
    'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151,
    'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
    'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193, 'Sibiu': 253,
    'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
}

def get_neighbors(city: str) -> List[Tuple[str, int]]:
    """Return list of (neighbor_city, distance) for given city"""
    return ROMANIA_MAP.get(city, [])

def heuristic(city: str, goal: str = 'Bucharest') -> float:
    """Straight-line distance heuristic for A* and Greedy"""
    return STRAIGHT_LINE_TO_BUCHAREST.get(city, float('inf'))

# ==================== BFS ====================
def bfs_city_search(start: str, goal: str) -> Optional[List[str]]:
    """
    BFS: Explores level by level.
    - Guarantees shortest path in terms of NUMBER OF CITIES (not distance)
    - Uses queue (FIFO)
    - Time: O(V + E), Space: O(V)
    """
    queue = deque([(start, [start])])  # (current_city, path_so_far)
    visited = {start}
    
    while queue:
        city, path = queue.popleft()
        
        if city == goal:
            return path
        
        for neighbor, _ in get_neighbors(city):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# ==================== UCS ====================
def ucs_city_search(start: str, goal: str) -> Tuple[Optional[List[str]], float]:
    """
    Uniform Cost Search: Finds path with minimum TOTAL DISTANCE.
    - Uses priority queue (min-heap) ordered by cumulative cost
    - Guarantees optimal path for weighted graphs
    - Equivalent to Dijkstra's algorithm
    """
    pq = [(0, start, [start])]  # (total_cost, current_city, path)
    visited = {}  # city -> best_cost_so_far
    
    while pq:
        cost, city, path = heapq.heappop(pq)
        
        # Skip if we already found a better path to this city
        if city in visited and visited[city] <= cost:
            continue
        visited[city] = cost
        
        if city == goal:
            return path, cost
        
        for neighbor, dist in get_neighbors(city):
            new_cost = cost + dist
            if neighbor not in visited or visited[neighbor] > new_cost:
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
    
    return None, float('inf')

# ==================== A* ====================
def a_star_city_search(start: str, goal: str = 'Bucharest') -> Tuple[Optional[List[str]], float]:
    """
    A* Search: Uses f(n) = g(n) + h(n)
    - g(n) = actual cost from start to n
    - h(n) = straight-line distance to goal (heuristic)
    - More efficient than UCS when heuristic is good and admissible
    """
    def h(city: str) -> float:
        return heuristic(city, goal)
    
    pq = [(h(start), 0, start, [start])]  # (f_score, g_score, city, path)
    visited = {}
    
    while pq:
        f_score, g_score, city, path = heapq.heappop(pq)
        
        if city in visited and visited[city] <= g_score:
            continue
        visited[city] = g_score
        
        if city == goal:
            return path, g_score
        
        for neighbor, dist in get_neighbors(city):
            new_g = g_score + dist
            new_f = new_g + h(neighbor)
            if neighbor not in visited or visited[neighbor] > new_g:
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
    
    return None, float('inf')

# ==================== Greedy ====================
def greedy_city_search(start: str, goal: str = 'Bucharest') -> Tuple[Optional[List[str]], float]:
    """
    Greedy Best-First Search: Uses only h(n)
    - Explores nodes closest to goal according to heuristic
    - NOT optimal, can get stuck in local minima
    - Faster but may not find shortest path
    """
    def h(city: str) -> float:
        return heuristic(city, goal)
    
    pq = [(h(start), start, [start], 0)]  # (heuristic, city, path, cost)
    visited = set()
    
    while pq:
        _, city, path, cost = heapq.heappop(pq)
        
        if city in visited:
            continue
        visited.add(city)
        
        if city == goal:
            return path, cost
        
        for neighbor, dist in get_neighbors(city):
            if neighbor not in visited:
                heapq.heappush(pq, (h(neighbor), neighbor, path + [neighbor], cost + dist))
    
    return None, float('inf')

# ==================== Bidirectional Search ====================
def bidirectional_city_search(start: str, goal: str) -> Optional[List[str]]:
    """
    Bidirectional Search: Searches from start AND goal simultaneously
    - Two BFS frontiers meet in the middle
    - Much faster: O(b^(d/2)) instead of O(b^d)
    - Requires ability to reverse graph (undirected here)
    """
    if start == goal:
        return [start]
    
    # Forward search from start
    forward_queue = deque([start])
    forward_parent = {start: None}
    
    # Backward search from goal
    backward_queue = deque([goal])
    backward_parent = {goal: None}
    
    while forward_queue and backward_queue:
        # Expand forward frontier
        for _ in range(len(forward_queue)):
            city = forward_queue.popleft()
            
            # Check if forward and backward frontiers meet
            if city in backward_parent:
                # Build complete path
                path = []
                curr = city
                while curr is not None:
                    path.append(curr)
                    curr = forward_parent[curr]
                path.reverse()
                curr = backward_parent[city]
                while curr is not None:
                    path.append(curr)
                    curr = backward_parent[curr]
                return path
            
            for neighbor, _ in get_neighbors(city):
                if neighbor not in forward_parent:
                    forward_parent[neighbor] = city
                    forward_queue.append(neighbor)
        
        # Expand backward frontier
        for _ in range(len(backward_queue)):
            city = backward_queue.popleft()
            
            if city in forward_parent:
                path = []
                curr = city
                while curr is not None:
                    path.append(curr)
                    curr = forward_parent[curr]
                path.reverse()
                curr = backward_parent[city]
                while curr is not None:
                    path.append(curr)
                    curr = backward_parent[curr]
                return path
            
            for neighbor, _ in get_neighbors(city):
                if neighbor not in backward_parent:
                    backward_parent[neighbor] = city
                    backward_queue.append(neighbor)
    
    return None

# ==================== Main Execution ====================
if __name__ == "__main__":
    START = 'Arad'
    GOAL = 'Bucharest'
    
    print("=" * 60)
    print(f"Problem 1: City Map Navigation from {START} to {GOAL}")
    print("=" * 60)
    
    # BFS
    path = bfs_city_search(START, GOAL)
    print(f"\nBFS (shortest by #cities): {path}")
    print(f"  Path length: {len(path) if path else 0} cities")
    
    # UCS
    path, cost = ucs_city_search(START, GOAL)
    print(f"\nUCS (shortest by distance): {path}")
    print(f"  Total distance: {cost} km")
    
    # A*
    path, cost = a_star_city_search(START, GOAL)
    print(f"\nA* (with heuristic): {path}")
    print(f"  Total distance: {cost} km")
    
    # Greedy
    path, cost = greedy_city_search(START, GOAL)
    print(f"\nGreedy (heuristic only): {path}")
    print(f"  Total distance: {cost} km")
    
    # Bidirectional
    path = bidirectional_city_search(START, GOAL)
    print(f"\nBidirectional BFS: {path}")
    print(f"  Path length: {len(path) if path else 0} cities")