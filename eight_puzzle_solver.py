"""
Problem 3: 8-Puzzle Solver (Sliding Tile Puzzle)
Goal: Rearrange tiles from scrambled state to goal configuration
Algorithms: BFS, IDDFS, A* (with misplaced tiles & Manhattan heuristics)
"""

from collections import deque
import heapq
from typing import List, Tuple, Optional, Set

class EightPuzzle:
    """
    8-puzzle: 3x3 grid with tiles 1-8 and one empty space (0)
    Goal state: [1,2,3,4,5,6,7,8,0]
    """
    
    def __init__(self, initial: List[int], goal: List[int] = None):
        self.initial = tuple(initial)
        self.goal = tuple(goal) if goal else (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.size = 3
    
    def get_blank_pos(self, state: Tuple[int, ...]) -> int:
        """Return index of blank space (0)"""
        return state.index(0)
    
    def get_neighbors(self, state: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """
        Generate possible moves by sliding a tile into the blank space.
        Returns list of new states.
        """
        blank = self.get_blank_pos(state)
        row, col = blank // 3, blank % 3
        neighbors = []
        
        # Possible moves: up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_blank = new_row * 3 + new_col
                state_list = list(state)
                # Swap blank with adjacent tile
                state_list[blank], state_list[new_blank] = state_list[new_blank], state_list[blank]
                neighbors.append(tuple(state_list))
        
        return neighbors
    
    def misplaced_tiles_heuristic(self, state: Tuple[int, ...]) -> int:
        """
        Heuristic 1: Count number of tiles in wrong position.
        Excludes blank space (0).
        Admissible? Yes, each misplaced tile needs at least 1 move.
        """
        count = 0
        for i in range(9):
            if state[i] != 0 and state[i] != self.goal[i]:
                count += 1
        return count
    
    def manhattan_heuristic(self, state: Tuple[int, ...]) -> int:
        """
        Heuristic 2: Sum of Manhattan distances of each tile to its goal position.
        More informed than misplaced tiles.
        Admissible? Yes, Manhattan distance is lower bound.
        """
        distance = 0
        for i in range(9):
            if state[i] != 0:
                current_row, current_col = i // 3, i % 3
                goal_pos = self.goal.index(state[i])
                goal_row, goal_col = goal_pos // 3, goal_pos % 3
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        return distance
    
    def bfs(self) -> Tuple[Optional[List[Tuple[int, ...]]], int]:
        """
        BFS: Guarantees minimum number of moves.
        - State space: 9!/2 = 181,440 states
        - Memory intensive
        """
        queue = deque([(self.initial, [self.initial])])
        visited = {self.initial}
        nodes_explored = 0
        
        while queue:
            state, path = queue.popleft()
            nodes_explored += 1
            
            if state == self.goal:
                return path, nodes_explored
            
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None, nodes_explored
    
    def iddfs(self, max_depth: int = 30) -> Tuple[Optional[List[Tuple[int, ...]]], int]:
        """
        Iterative Deepening DFS:
        - Combines DFS memory efficiency with BFS completeness
        - Repeatedly runs DFS with increasing depth limits
        - Memory: O(depth) instead of O(states)
        """
        def dls(state: Tuple[int, ...], depth: int, path: List[Tuple[int, ...]], 
                visited: Set[Tuple[int, ...]]) -> Optional[List[Tuple[int, ...]]]:
            """Depth-Limited DFS"""
            if state == self.goal:
                return path
            if depth <= 0:
                return None
            
            for neighbor in self.get_neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    result = dls(neighbor, depth - 1, path + [neighbor], visited)
                    if result:
                        return result
                    visited.remove(neighbor)
            return None
        
        nodes_explored = 0
        for depth in range(max_depth):
            visited = {self.initial}
            result = dls(self.initial, depth, [self.initial], visited)
            nodes_explored += len(visited)
            if result:
                return result, nodes_explored
        
        return None, nodes_explored
    
    def a_star(self, heuristic: str = 'manhattan') -> Tuple[Optional[List[Tuple[int, ...]]], int]:
        """
        A* Search: f(n) = g(n) + h(n)
        - g(n) = number of moves so far
        - h(n) = heuristic (misplaced tiles or Manhattan)
        - Optimal and more efficient than BFS
        """
        # Select heuristic function
        if heuristic == 'manhattan':
            h_func = self.manhattan_heuristic
        else:
            h_func = self.misplaced_tiles_heuristic
        
        # Priority queue: (f_score, g_score, state, path)
        pq = [(h_func(self.initial), 0, self.initial, [self.initial])]
        visited = {}  # state -> best g_score
        nodes_explored = 0
        
        while pq:
            f_score, g_score, state, path = heapq.heappop(pq)
            nodes_explored += 1
            
            # Skip if we found a better path to this state
            if state in visited and visited[state] <= g_score:
                continue
            visited[state] = g_score
            
            if state == self.goal:
                return path, nodes_explored
            
            for neighbor in self.get_neighbors(state):
                new_g = g_score + 1
                new_f = new_g + h_func(neighbor)
                if neighbor not in visited or visited[neighbor] > new_g:
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
        
        return None, nodes_explored
    
    def print_state(self, state: Tuple[int, ...]):
        """Print 3x3 grid representation of a state"""
        for i in range(0, 9, 3):
            row = state[i:i+3]
            print(' '.join(str(x) if x != 0 else '_' for x in row))


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Initial scrambled state (0 represents empty space)
    initial = [1, 2, 3, 4, 0, 5, 6, 7, 8]
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    
    puzzle = EightPuzzle(initial, goal)
    
    print("=" * 60)
    print("Problem 3: 8-Puzzle Solver")
    print("=" * 60)
    
    print("\nInitial State:")
    puzzle.print_state(puzzle.initial)
    print("\nGoal State:")
    puzzle.print_state(puzzle.goal)
    
    # BFS
    print("\n--- BFS ---")
    path, nodes = puzzle.bfs()
    if path:
        print(f"Solution found in {len(path)-1} moves")
        print(f"Nodes explored: {nodes}")
    else:
        print("No solution found")
    
    # IDDFS
    print("\n--- IDDFS ---")
    path, nodes = puzzle.iddfs(max_depth=25)
    if path:
        print(f"Solution found in {len(path)-1} moves")
        print(f"Nodes explored: {nodes}")
    else:
        print("No solution found within depth limit")
    
    # A* with Manhattan heuristic
    print("\n--- A* with Manhattan Heuristic ---")
    path, nodes = puzzle.a_star(heuristic='manhattan')
    if path:
        print(f"Solution found in {len(path)-1} moves")
        print(f"Nodes explored: {nodes}")
        
        print("\nSolution path (first few states):")
        for i, state in enumerate(path[:5]):
            print(f"\nStep {i}:")
            puzzle.print_state(state)
        if len(path) > 5:
            print(f"\n... and {len(path)-5} more steps")
    
    # A* with misplaced tiles heuristic
    print("\n--- A* with Misplaced Tiles Heuristic ---")
    path, nodes = puzzle.a_star(heuristic='misplaced')
    if path:
        print(f"Solution found in {len(path)-1} moves")
        print(f"Nodes explored: {nodes}")