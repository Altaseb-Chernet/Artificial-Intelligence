"""
Problem 7: Game AI Pathfinding (Maze Navigation)
Goal: Move from (1,1) to (7,7) while avoiding walls
Algorithms: DFS, A*, Greedy
"""

import heapq
import math
from typing import List, Tuple, Optional, Set

class GameMaze:
    """
    Maze navigation for game character.
    0 = open path, 1 = wall
    """
    
    def __init__(self, maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.rows = len(maze)
        self.cols = len(maze[0]) if self.rows > 0 else 0
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall"""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.maze[r][c] == 0
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 4-directional neighbors"""
        r, c = pos
        # Can add diagonal movement by uncommenting:
        # neighbors = [(r+1,c), (r-1,c), (r,c+1), (r,c-1), (r+1,c+1), (r-1,c-1), (r+1,c-1), (r-1,c+1)]
        neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
        return [n for n in neighbors if self.is_valid(n)]
    
    def heuristic(self, pos: Tuple[int, int]) -> float:
        """
        Euclidean distance heuristic (admissible).
        For grid, we could also use Manhattan or Chebyshev.
        """
        return math.sqrt((pos[0] - self.goal[0])**2 + (pos[1] - self.goal[1])**2)
    
    def manhattan_heuristic(self, pos: Tuple[int, int]) -> int:
        """Manhattan distance (faster to compute)"""
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
    
    def dfs(self, max_depth: int = 100) -> Optional[List[Tuple[int, int]]]:
        """
        DFS for game pathfinding:
        - Explores deep paths first
        - May find a path but NOT guaranteed shortest
        - Memory efficient: O(depth)
        - Can get stuck in dead ends
        """
        def dfs_recursive(pos: Tuple[int, int], path: List[Tuple[int, int]], 
                         visited: Set[Tuple[int, int]]):
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
    
    def a_star(self, use_manhattan: bool = False) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """
        A* for game pathfinding:
        - Finds optimal shortest path
        - Uses heuristic to guide search toward goal
        - Most common in game AI
        """
        h_func = self.manhattan_heuristic if use_manhattan else self.heuristic
        
        # Priority queue: (f_score, g_score, position, path)
        pq = [(h_func(self.start), 0, self.start, [self.start])]
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
                new_f = new_g + h_func(neighbor)
                if neighbor not in visited or visited[neighbor] > new_g:
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
        
        return None, float('inf')
    
    def greedy(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Greedy Best-First for game pathfinding:
        - Only uses heuristic (distance to goal)
        - Faster but may not find optimal path
        - Can get stuck in local minima
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
    
    def print_maze(self, path: List[Tuple[int, int]] = None):
        """Visualize maze with path"""
        maze_copy = [row[:] for row in self.maze]
        
        if path:
            for r, c in path:
                if (r, c) != self.start and (r, c) != self.goal:
                    maze_copy[r][c] = '.'
        
        # Mark start and goal
        sr, sc = self.start
        gr, gc = self.goal
        maze_copy[sr][sc] = 'S'
        maze_copy[gr][gc] = 'G'
        
        print("\nMaze (0=path, 1=wall, S=start, G=goal, .=path):")
        for row in maze_copy:
            print(' '.join(str(cell) for cell in row))


# ==================== Main Execution ====================
if __name__ == "__main__":
    # 9x9 maze: 0=path, 1=wall
    # Start at (1,1), Goal at (7,7)
    maze = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    
    start = (1, 1)
    goal = (7, 7)
    
    game = GameMaze(maze, start, goal)
    
    print("=" * 60)
    print(f"Problem 7: Game AI Pathfinding from {start} to {goal}")
    print("=" * 60)
    
    game.print_maze()
    
    # DFS
    print("\n--- DFS (Depth-First Search) ---")
    path = game.dfs(max_depth=50)
    if path:
        print(f"Path found with {len(path)-1} steps")
        print(f"Path: {path[:5]}... (showing first 5 steps)")
        game.print_maze(path)
    else:
        print("No path found (DFS may fail in complex mazes)")
    
    # A* with Euclidean heuristic
    print("\n--- A* with Euclidean Heuristic ---")
    path, cost = game.a_star(use_manhattan=False)
    if path:
        print(f"Optimal path with {cost} steps")
        print(f"Path: {path[:5]}... (showing first 5 steps)")
        game.print_maze(path)
    else:
        print("No path found")
    
    # A* with Manhattan heuristic
    print("\n--- A* with Manhattan Heuristic ---")
    path, cost = game.a_star(use_manhattan=True)
    if path:
        print(f"Optimal path with {cost} steps")
    
    # Greedy
    print("\n--- Greedy Best-First Search ---")
    path, cost = game.greedy()
    if path:
        print(f"Path found with {cost} steps (may not be optimal)")
        print(f"Path: {path[:5]}...")
        game.print_maze(path)
    else:
        print("No path found")
    
    # Algorithm comparison
    print("\n" + "=" * 60)
    print("Algorithm Comparison:")
    print("=" * 60)
    
    # Find optimal length using A*
    optimal_path, optimal_cost = game.a_star()
    
    dfs_path = game.dfs()
    greedy_path, greedy_cost = game.greedy()
    
    print(f"Optimal path length: {optimal_cost} steps (A*)")
    if dfs_path:
        print(f"DFS path length: {len(dfs_path)-1} steps")
    else:
        print("DFS: No path found")
    if greedy_path:
        print(f"Greedy path length: {greedy_cost} steps")
        if greedy_cost > optimal_cost:
            print(f"  Greedy took {greedy_cost - optimal_cost} extra steps")
    else:
        print("Greedy: No path found")