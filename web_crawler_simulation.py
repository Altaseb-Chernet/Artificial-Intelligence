"""
Problem 4: Web Crawler Simulation
Goal: Reach target webpage by following hyperlinks efficiently
Algorithms: BFS, DFS, DLS (Depth-Limited Search)
"""

from collections import deque
from typing import List, Tuple, Optional, Set

class WebCrawler:
    """
    Simulated web crawler that navigates hyperlinks.
    Web pages are nodes, hyperlinks are directed edges.
    """
    
    def __init__(self):
        # Simulated web graph: page -> list of linked pages
        self.web_graph = {
            'Home': ['News', 'Products', 'About'],
            'News': ['Home', 'Tech', 'Sports', 'World'],
            'Products': ['Home', 'Product1', 'Product2', 'Support'],
            'About': ['Home', 'Contact', 'Team'],
            'Tech': ['News', 'Gadgets', 'AI'],
            'Sports': ['News', 'Football', 'Basketball'],
            'World': ['News', 'Politics', 'Economy'],
            'Product1': ['Products', 'Buy'],
            'Product2': ['Products', 'Buy'],
            'Support': ['Products', 'FAQ', 'Contact'],
            'Contact': ['About', 'Support'],
            'Team': ['About'],
            'Gadgets': ['Tech'],
            'AI': ['Tech'],
            'Football': ['Sports'],
            'Basketball': ['Sports'],
            'Politics': ['World'],
            'Economy': ['World'],
            'Buy': ['Product1', 'Product2', 'Checkout'],
            'FAQ': ['Support'],
            'Checkout': ['Buy', 'Payment'],
            'Payment': ['Checkout']
        }
    
    def get_links(self, page: str) -> List[str]:
        """Return list of pages linked from given page"""
        return self.web_graph.get(page, [])
    
    def bfs_search(self, start: str, target: str) -> Tuple[Optional[List[str]], int]:
        """
        BFS for web crawling:
        - Explores pages level by level (breadth-first)
        - Good for finding shortest path (fewest clicks)
        - Memory intensive (stores entire frontier)
        """
        queue = deque([(start, [start])])  # (page, path)
        visited = {start}
        pages_visited = 0
        
        while queue:
            page, path = queue.popleft()
            pages_visited += 1
            
            if page == target:
                return path, pages_visited
            
            for link in self.get_links(page):
                if link not in visited:
                    visited.add(link)
                    queue.append((link, path + [link]))
        
        return None, pages_visited
    
    def dfs_search(self, start: str, target: str, max_depth: int = 10) -> Tuple[Optional[List[str]], int]:
        """
        DFS for web crawling:
        - Explores deep paths first (depth-first)
        - Memory efficient (only stores current path)
        - May not find shortest path
        - Can get stuck in deep branches
        """
        visited = set()
        pages_visited = 0
        
        def dfs_recursive(page: str, path: List[str], depth: int):
            nonlocal pages_visited
            pages_visited += 1
            
            if page == target:
                return path
            if depth >= max_depth:
                return None
            
            for link in self.get_links(page):
                if link not in visited:
                    visited.add(link)
                    result = dfs_recursive(link, path + [link], depth + 1)
                    if result:
                        return result
                    visited.remove(link)  # Backtrack
            return None
        
        visited.add(start)
        result = dfs_recursive(start, [start], 0)
        return result, pages_visited
    
    def dls_search(self, start: str, target: str, depth_limit: int = 5) -> Tuple[Optional[List[str]], int]:
        """
        Depth-Limited Search (DLS):
        - DFS with a maximum depth limit
        - Prevents infinite loops in cyclic graphs
        - May miss solutions beyond depth limit
        """
        visited = set()
        pages_visited = 0
        
        def dls_recursive(page: str, path: List[str], depth: int):
            nonlocal pages_visited
            pages_visited += 1
            
            if page == target:
                return path
            if depth >= depth_limit:
                return None
            
            for link in self.get_links(page):
                if link not in visited:
                    visited.add(link)
                    result = dls_recursive(link, path + [link], depth + 1)
                    if result:
                        return result
                    visited.remove(link)
            return None
        
        visited.add(start)
        result = dls_recursive(start, [start], 0)
        return result, pages_visited
    
    def iterative_deepening_search(self, start: str, target: str, max_depth: int = 10) -> Tuple[Optional[List[str]], int]:
        """
        Iterative Deepening Search (IDS):
        - Repeatedly runs DLS with increasing depth limits
        - Combines DFS memory efficiency with BFS completeness
        - Guarantees shortest path
        """
        total_pages_visited = 0
        
        for depth_limit in range(max_depth + 1):
            result, pages = self.dls_search(start, target, depth_limit)
            total_pages_visited += pages
            if result is not None:
                return result, total_pages_visited
        
        return None, total_pages_visited


# ==================== Main Execution ====================
if __name__ == "__main__":
    crawler = WebCrawler()
    start_page = 'Home'
    target_page = 'Payment'
    
    print("=" * 60)
    print(f"Problem 4: Web Crawler Simulation")
    print(f"Start: {start_page} -> Target: {target_page}")
    print("=" * 60)
    
    print("\nWeb Graph Structure (partial):")
    print("Home -> News, Products, About")
    print("Products -> Product1, Product2, Support")
    print("Buy -> Checkout -> Payment")
    
    # BFS
    print("\n--- BFS (Breadth-First Search) ---")
    path, visited = crawler.bfs_search(start_page, target_page)
    if path:
        print(f"Path found: {' -> '.join(path)}")
        print(f"Clicks needed: {len(path)-1}")
        print(f"Pages visited: {visited}")
    else:
        print("Target not reachable")
    
    # DFS
    print("\n--- DFS (Depth-First Search) ---")
    path, visited = crawler.dfs_search(start_page, target_page, max_depth=8)
    if path:
        print(f"Path found: {' -> '.join(path)}")
        print(f"Clicks needed: {len(path)-1}")
        print(f"Pages visited: {visited}")
    else:
        print("Target not reachable within depth limit")
    
    # DLS with depth limit
    print("\n--- DLS (Depth-Limited Search, limit=3) ---")
    path, visited = crawler.dls_search(start_page, target_page, depth_limit=3)
    if path:
        print(f"Path found: {' -> '.join(path)}")
        print(f"Clicks needed: {len(path)-1}")
    else:
        print("Target not reachable within depth limit 3")
    
    # DLS with higher depth limit
    print("\n--- DLS (Depth-Limited Search, limit=6) ---")
    path, visited = crawler.dls_search(start_page, target_page, depth_limit=6)
    if path:
        print(f"Path found: {' -> '.join(path)}")
        print(f"Clicks needed: {len(path)-1}")
    else:
        print("Target not reachable within depth limit 6")
    
    # Iterative Deepening
    print("\n--- Iterative Deepening Search ---")
    path, visited = crawler.iterative_deepening_search(start_page, target_page, max_depth=6)
    if path:
        print(f"Path found: {' -> '.join(path)}")
        print(f"Clicks needed: {len(path)-1}")
        print(f"Total pages visited across all depths: {visited}")