"""
Problem 2: Robot Pathfinding (Grid Navigation)
Goal: Move from (0,0) to (4,4) avoiding obstacles
Algorithms: BFS, DFS, A*, Greedy
"""

from collections import deque
import heapq
from typing import List, Tuple, Optional, Set

class RobotPathfinding:
    """
    2D grid navigation for a robot in a warehouse.
    0 = empty cell, 1 = obstacle
    """
    
    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not an obstacle"""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 4-directional neighbors (up, down, left, right)"""
        r, c = pos
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        return [n for n in neighbors if self.is_valid(n)]
    
    def heuristic(self, pos: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible for grid)"""
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
    
    def bfs(self) -> Optional[List[Tuple[int, int]]]:
        """
        BFS: Guarantees shortest path on unweighted grid.
        Uses queue (FIFO). Each move costs 1.
        """
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        
        while queue:
            pos, path = queue.popleft()
            
            if pos == self.goal:
                return path
            
            for neighbor in self.get_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None
    
    def dfs(self, max_depth: int = 100) -> Optional[List[Tuple[int, int]]]:
        """
        DFS: Explores deep paths first.
        - Does NOT guarantee shortest path
        - Can get stuck in deep branches
        - Memory efficient (O(depth))
        """
        def dfs_recursive(pos: Tuple[int, int], path: List[Tuple[int, int]], 
                         visited: Set[Tuple[int, int]]):
            # Depth limit to avoid infinite recursion
            if len(path) > max_depth:
                return None
            
            if pos == self.goal:
                return path
            
            for neighbor in self.get_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    result = dfs_recursive(neighbor, path + [neighbor], visited)
                    if result:
                        return result
                    visited.remove(neighbor)  # Backtrack
            return None
        
        return dfs_recursive(self.start, [self.start], {self.start})
    
    def a_star(self) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """
        A*: f(n) = g(n) + h(n)
        - h(n) = Manhattan distance to goal
        - Guarantees optimal path
        - More efficient than BFS for large grids
        """
        pq = [(self.heuristic(self.start), 0, self.start, [self.start])]
        visited = {}  # position -> best g_score
        
        while pq:
            f_score, g_score, pos, path = heapq.heappop(pq)
            
            if pos in visited and visited[pos] <= g_score:
                continue
            visited[pos] = g_score
            
            if pos == self.goal:
                return path, g_score
            
            for neighbor in self.get_neighbors(pos):
                new_g = g_score + 1
                new_f = new_g + self.heuristic(neighbor)
                if neighbor not in visited or visited[neighbor] > new_g:
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
        
        return None, float('inf')
    
    def greedy(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Greedy Best-First: Uses only heuristic h(n)
        - Faster but NOT optimal
        - Can take longer paths or get stuck
        """
        pq = [(self.heuristic(self.start), self.start, [self.start], 0)]
        visited = set()
        
        while pq:
            _, pos, path, cost = heapq.heappop(pq)
            
            if pos in visited:
                continue
            visited.add(pos)
            
            if pos == self.goal:
                return path, cost
            
            for neighbor in self.get_neighbors(pos):
                if neighbor not in visited:
                    heapq.heappush(pq, (self.heuristic(neighbor), neighbor, 
                                       path + [neighbor], cost + 1))
        
        return None, 0
    
    def print_path(self, path: List[Tuple[int, int]]):
        """Visualize the path on the grid"""
        if not path:
            print("No path found!")
            return
        
        grid_copy = [row[:] for row in self.grid]
        for r, c in path:
            if (r, c) != self.start and (r, c) != self.goal:
                grid_copy[r][c] = '.'
        
        # Mark start and goal
        sr, sc = self.start
        gr, gc = self.goal
        grid_copy[sr][sc] = 'S'
        grid_copy[gr][gc] = 'G'
        
        print("\nGrid with path (S=start, G=goal, .=path):")
        for row in grid_copy:
            print(' '.join(str(cell) for cell in row))


# ==================== Main Execution ====================
if __name__ == "__main__":
    # 5x5 grid: 0=empty, 1=obstacle
    # Start at (0,0), Goal at (4,4)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (4, 4)
    
    robot = RobotPathfinding(grid, start, goal)
    
    print("=" * 60)
    print(f"Problem 2: Robot Pathfinding from {start} to {goal}")
    print("=" * 60)
    print("Grid (0=empty, 1=obstacle):")
    for row in grid:
        print(row)
    
    # BFS
    path = robot.bfs()
    print(f"\nBFS Path: {path}")
    print(f"  Path length: {len(path) if path else 0} steps")
    
    # DFS
    path = robot.dfs(max_depth=50)
    print(f"\nDFS Path: {path}")
    print(f"  Path length: {len(path) if path else 0} steps (may not be shortest)")
    
    # A*
    path, cost = robot.a_star()
    print(f"\nA* Path: {path}")
    print(f"  Path length: {cost} steps")
    robot.print_path(path)
    
    # Greedy
    path, cost = robot.greedy()
    print(f"\nGreedy Path: {path}")
    print(f"  Path length: {cost} steps")