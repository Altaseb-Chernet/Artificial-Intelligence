"""
Problem 5: Social Network Connection Finder
Goal: Find shortest friendship chain between two people
Algorithms: BFS, Bidirectional Search
"""

from collections import deque
from typing import List, Optional

class SocialNetwork:
    """
    Social network graph where:
    - Nodes = people
    - Edges = friendships (undirected)
    """
    
    def __init__(self):
        # Friendship graph (undirected)
        self.friendships = {
            'Alice': ['Bob', 'Carol', 'David'],
            'Bob': ['Alice', 'Eve', 'Frank'],
            'Carol': ['Alice', 'Grace'],
            'David': ['Alice', 'Henry'],
            'Eve': ['Bob', 'Ivy'],
            'Frank': ['Bob', 'Jack'],
            'Grace': ['Carol'],
            'Henry': ['David'],
            'Ivy': ['Eve'],
            'Jack': ['Frank'],
            'Kevin': ['Lisa'],      # Isolated group
            'Lisa': ['Kevin']
        }
    
    def get_friends(self, person: str) -> List[str]:
        """Return list of friends for a given person"""
        return self.friendships.get(person, [])
    
    def bfs_connection(self, start: str, goal: str) -> Optional[List[str]]:
        """
        BFS to find shortest friendship chain.
        - Guarantees minimum number of connections (degrees of separation)
        - Explores friends level by level
        - Time: O(V + E), Space: O(V)
        """
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])  # (person, path)
        visited = {start}
        
        while queue:
            person, path = queue.popleft()
            
            for friend in self.get_friends(person):
                if friend == goal:
                    return path + [friend]
                if friend not in visited:
                    visited.add(friend)
                    queue.append((friend, path + [friend]))
        
        return None
    
    def bidirectional_connection(self, start: str, goal: str) -> Optional[List[str]]:
        """
        Bidirectional Search for shortest friendship chain.
        - Searches from both start and goal simultaneously
        - Much faster: O(b^(d/2)) vs O(b^d) for BFS
        - Requires undirected graph (friendships are mutual)
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
                person = forward_queue.popleft()
                
                # Check if frontiers meet
                if person in backward_parent:
                    # Build complete path
                    path = []
                    # Forward portion
                    curr = person
                    while curr is not None:
                        path.append(curr)
                        curr = forward_parent[curr]
                    path.reverse()
                    # Backward portion (excluding meeting point)
                    curr = backward_parent[person]
                    while curr is not None:
                        path.append(curr)
                        curr = backward_parent[curr]
                    return path
                
                for friend in self.get_friends(person):
                    if friend not in forward_parent:
                        forward_parent[friend] = person
                        forward_queue.append(friend)
            
            # Expand backward frontier
            for _ in range(len(backward_queue)):
                person = backward_queue.popleft()
                
                if person in forward_parent:
                    path = []
                    curr = person
                    while curr is not None:
                        path.append(curr)
                        curr = forward_parent[curr]
                    path.reverse()
                    curr = backward_parent[person]
                    while curr is not None:
                        path.append(curr)
                        curr = backward_parent[curr]
                    return path
                
                for friend in self.get_friends(person):
                    if friend not in backward_parent:
                        backward_parent[friend] = person
                        backward_queue.append(friend)
        
        return None
    
    def degrees_of_separation(self, person1: str, person2: str) -> int:
        """Calculate degrees of separation between two people"""
        path = self.bfs_connection(person1, person2)
        if path:
            return len(path) - 1  # number of edges
        return -1  # No connection


# ==================== Main Execution ====================
if __name__ == "__main__":
    network = SocialNetwork()
    
    print("=" * 60)
    print("Problem 5: Social Network Connection Finder")
    print("=" * 60)
    
    print("\nFriendship Graph:")
    for person, friends in network.friendships.items():
        print(f"  {person}: {friends}")
    
    # Test case 1: Alice to Eve
    print("\n" + "-" * 40)
    print("Test 1: Alice -> Eve")
    
    path = network.bfs_connection('Alice', 'Eve')
    if path:
        print(f"BFS Path: {' -> '.join(path)}")
        print(f"Degrees of separation: {len(path)-1}")
    else:
        print("No connection found")
    
    path = network.bidirectional_connection('Alice', 'Eve')
    if path:
        print(f"Bidirectional Path: {' -> '.join(path)}")
    
    # Test case 2: Alice to Jack
    print("\n" + "-" * 40)
    print("Test 2: Alice -> Jack")
    
    path = network.bfs_connection('Alice', 'Jack')
    if path:
        print(f"BFS Path: {' -> '.join(path)}")
        print(f"Degrees of separation: {len(path)-1}")
    
    # Test case 3: Carol to Frank
    print("\n" + "-" * 40)
    print("Test 3: Carol -> Frank")
    
    path = network.bfs_connection('Carol', 'Frank')
    if path:
        print(f"BFS Path: {' -> '.join(path)}")
        print(f"Degrees of separation: {len(path)-1}")
    
    # Test case 4: Unconnected people
    print("\n" + "-" * 40)
    print("Test 4: Alice -> Kevin (different component)")
    
    path = network.bfs_connection('Alice', 'Kevin')
    if path:
        print(f"Path: {' -> '.join(path)}")
    else:
        print("No connection exists (different social circles)")
    
    # Test case 5: Same person
    print("\n" + "-" * 40)
    print("Test 5: Alice -> Alice")
    
    path = network.bfs_connection('Alice', 'Alice')
    print(f"Path: {' -> '.join(path) if path else 'None'}")
    print(f"Degrees of separation: 0")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary of Connections:")
    print("=" * 60)
    pairs = [('Alice', 'Eve'), ('Alice', 'Jack'), ('Carol', 'Frank'), 
             ('Bob', 'Grace'), ('David', 'Ivy')]
    for p1, p2 in pairs:
        degrees = network.degrees_of_separation(p1, p2)
        if degrees >= 0:
            print(f"  {p1} -> {p2}: {degrees} degree(s) of separation")
        else:
            print(f"  {p1} -> {p2}: No connection")