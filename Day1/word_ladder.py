"""
Problem 8: Word Ladder (Word Transformation Game)
Goal: Transform 'hit' to 'cog' using valid dictionary words
Algorithms: BFS, A*, Bidirectional Search
"""

from collections import deque
import heapq
from typing import List, Tuple, Optional, Set

class WordLadder:
    """
    Word transformation by changing one letter at a time.
    Each intermediate word must be in dictionary.
    """
    
    def __init__(self, dictionary: Set[str]):
        self.dictionary = dictionary
    
    def get_neighbors(self, word: str) -> List[str]:
        """
        Generate all valid words by changing one letter at a time.
        Time: O(26 * word_length) per word
        """
        neighbors = []
        word_list = list(word)
        
        for i in range(len(word)):
            original_char = word_list[i]
            # Try all 26 letters
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == original_char:
                    continue
                word_list[i] = c
                new_word = ''.join(word_list)
                if new_word in self.dictionary:
                    neighbors.append(new_word)
            word_list[i] = original_char  # Restore
        
        return neighbors
    
    def heuristic(self, word: str, goal: str) -> int:
        """
        Heuristic for A*: number of differing characters.
        Admissible? Yes, each move can fix at most one difference.
        """
        return sum(1 for a, b in zip(word, goal) if a != b)
    
    def bfs(self, start: str, goal: str) -> Tuple[Optional[List[str]], int]:
        """
        BFS for word ladder:
        - Guarantees shortest transformation (minimum steps)
        - Explores words level by level
        - Most common approach for word ladder
        """
        if start == goal:
            return [start], 1
        
        queue = deque([(start, [start])])
        visited = {start}
        nodes_explored = 0
        
        while queue:
            word, path = queue.popleft()
            nodes_explored += 1
            
            for neighbor in self.get_neighbors(word):
                if neighbor == goal:
                    return path + [neighbor], nodes_explored + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None, nodes_explored
    
    def a_star(self, start: str, goal: str) -> Tuple[Optional[List[str]], int]:
        """
        A* for word ladder:
        - f(n) = g(n) + h(n)
        - h(n) = number of differing characters
        - More efficient than BFS for large dictionaries
        """
        pq = [(self.heuristic(start, goal), 0, start, [start])]
        visited = {}  # word -> best g_score
        nodes_explored = 0
        
        while pq:
            f_score, g_score, word, path = heapq.heappop(pq)
            nodes_explored += 1
            
            if word in visited and visited[word] <= g_score:
                continue
            visited[word] = g_score
            
            if word == goal:
                return path, nodes_explored
            
            for neighbor in self.get_neighbors(word):
                new_g = g_score + 1
                new_f = new_g + self.heuristic(neighbor, goal)
                if neighbor not in visited or visited[neighbor] > new_g:
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
        
        return None, nodes_explored
    
    def bidirectional_search(self, start: str, goal: str) -> Tuple[Optional[List[str]], int]:
        """
        Bidirectional Search for word ladder:
        - Searches from start and goal simultaneously
        - Much faster: O(b^(d/2)) vs O(b^d)
        - Requires ability to generate neighbors (same for both directions)
        """
        if start == goal:
            return [start], 1
        
        # Forward search from start
        forward_queue = deque([start])
        forward_parent = {start: None}
        forward_visited_count = 0
        
        # Backward search from goal
        backward_queue = deque([goal])
        backward_parent = {goal: None}
        backward_visited_count = 0
        
        while forward_queue and backward_queue:
            # Expand forward
            for _ in range(len(forward_queue)):
                word = forward_queue.popleft()
                forward_visited_count += 1
                
                if word in backward_parent:
                    # Build complete path
                    path = []
                    curr = word
                    while curr is not None:
                        path.append(curr)
                        curr = forward_parent[curr]
                    path.reverse()
                    curr = backward_parent[word]
                    while curr is not None:
                        path.append(curr)
                        curr = backward_parent[curr]
                    return path, forward_visited_count + backward_visited_count
                
                for neighbor in self.get_neighbors(word):
                    if neighbor not in forward_parent:
                        forward_parent[neighbor] = word
                        forward_queue.append(neighbor)
            
            # Expand backward
            for _ in range(len(backward_queue)):
                word = backward_queue.popleft()
                backward_visited_count += 1
                
                if word in forward_parent:
                    path = []
                    curr = word
                    while curr is not None:
                        path.append(curr)
                        curr = forward_parent[curr]
                    path.reverse()
                    curr = backward_parent[word]
                    while curr is not None:
                        path.append(curr)
                        curr = backward_parent[curr]
                    return path, forward_visited_count + backward_visited_count
                
                for neighbor in self.get_neighbors(word):
                    if neighbor not in backward_parent:
                        backward_parent[neighbor] = word
                        backward_queue.append(neighbor)
        
        return None, forward_visited_count + backward_visited_count


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Dictionary of valid English words
    dictionary = {
        'hit', 'hot', 'dot', 'lot', 'log', 'dog', 'cog', 'cat', 'bat', 'hat',
        'mat', 'pat', 'rat', 'sat', 'bet', 'get', 'jet', 'let', 'met', 'net',
        'pet', 'set', 'wet', 'yet', 'bit', 'fit', 'kit', 'lit', 'pit', 'sit',
        'wit', 'cot', 'got', 'not', 'pot', 'rot', 'tot', 'cut', 'hut', 'nut',
        'put', 'rut', 'but', 'gut', 'jut', 'lot', 'dot', 'hot', 'pot', 'rot'
    }
    
    start = 'hit'
    goal = 'cog'
    
    word_ladder = WordLadder(dictionary)
    
    print("=" * 60)
    print(f"Problem 8: Word Ladder from '{start}' to '{goal}'")
    print("=" * 60)
    
    print(f"\nDictionary size: {len(dictionary)} words")
    print(f"Sample dictionary: {sorted(dictionary)[:20]}...")
    
    # BFS
    print("\n--- BFS (Breadth-First Search) ---")
    path, explored = word_ladder.bfs(start, goal)
    if path:
        print(f"Transformation: {' -> '.join(path)}")
        print(f"Steps: {len(path)-1}")
        print(f"Words explored: {explored}")
    else:
        print(f"No transformation found from '{start}' to '{goal}'")
    
    # A*
    print("\n--- A* Search ---")
    path, explored = word_ladder.a_star(start, goal)
    if path:
        print(f"Transformation: {' -> '.join(path)}")
        print(f"Steps: {len(path)-1}")
        print(f"Words explored: {explored}")
    else:
        print("No transformation found")
    
    # Bidirectional Search
    print("\n--- Bidirectional Search ---")
    path, explored = word_ladder.bidirectional_search(start, goal)
    if path:
        print(f"Transformation: {' -> '.join(path)}")
        print(f"Steps: {len(path)-1}")
        print(f"Total words explored: {explored}")
    else:
        print("No transformation found")
    
    # Test with different word pairs
    print("\n" + "=" * 60)
    print("Testing Different Word Pairs:")
    print("=" * 60)
    
    test_pairs = [('cat', 'dog'), ('hot', 'cold'), ('bat', 'rat'), ('pit', 'pot')]
    
    for s, g in test_pairs:
        # Check if both words are in dictionary
        if s not in dictionary or g not in dictionary:
            print(f"\n{s} -> {g}: One or both words not in dictionary")
            continue
        
        path, _ = word_ladder.bfs(s, g)
        if path:
            print(f"\n{s} -> {g}: {' -> '.join(path)}")
            print(f"  Steps: {len(path)-1}")
        else:
            print(f"\n{s} -> {g}: No transformation found")
    
    # Comparison of algorithm performance
    print("\n" + "=" * 60)
    print("Algorithm Performance Comparison:")
    print("=" * 60)
    
    import time
    
    def time_algorithm(algo_func, *args):
        start_time = time.time()
        result, explored = algo_func(*args)
        end_time = time.time()
        return result, explored, end_time - start_time
    
    # BFS timing
    _, bfs_explored, bfs_time = time_algorithm(word_ladder.bfs, start, goal)
    
    # A* timing
    _, astar_explored, astar_time = time_algorithm(word_ladder.a_star, start, goal)
    
    # Bidirectional timing
    _, bi_explored, bi_time = time_algorithm(word_ladder.bidirectional_search, start, goal)
    
    print(f"\nBFS:           {bfs_explored} nodes, {bfs_time:.4f} seconds")
    print(f"A*:            {astar_explored} nodes, {astar_time:.4f} seconds")
    print(f"Bidirectional: {bi_explored} nodes, {bi_time:.4f} seconds")
    
    if astar_explored < bfs_explored:
        print(f"\nA* explored {bfs_explored - astar_explored} fewer nodes than BFS")
    if bi_explored < bfs_explored:
        print(f"Bidirectional explored {bfs_explored - bi_explored} fewer nodes than BFS")