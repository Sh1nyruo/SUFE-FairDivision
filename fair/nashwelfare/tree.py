import networkx as nx

# Add a parent attribute to TreeNode for easy access to the parent node
class TreeNode:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.children = []

def build_tree_structure(graph, root_agent):
    """
    Convert the graph to a tree structure, avoiding cycles by maintaining a set of visited nodes,
    and including a parent reference in each TreeNode.
    
    :param graph: The networkx graph representing the forest.
    :param root_agent: The identifier of the root agent for the tree.
    :return: The root TreeNode of the constructed tree.
    """
    # Initialize the root of the tree
    root = TreeNode(root_agent)
    visited = set([root_agent])  # Set of visited nodes to avoid cycles

    # Use a stack to perform BFS traversal
    stack = [(root, root_agent)]

    while stack:
        parent_node, _ = stack.pop()

        # Iterate over the neighbors of the current node
        for child_value in graph.neighbors(parent_node.value):
            # Skip if the child has already been visited (to avoid cycles)
            if child_value in visited:
                continue
            
            # Otherwise, process the child
            child_node = TreeNode(child_value, parent=parent_node)
            parent_node.children.append(child_node)
            stack.append((child_node, child_value))
            visited.add(child_value)  # Mark the child as visited

    return root



# Assuming you have a function to choose the root agent for each tree
def choose_root_agent(tree):
    # Placeholder function for choosing the root agent for a tree.
    # You need to implement this according to your own logic or criteria.
    agents = [node for node in tree.nodes if 'agent' in node]
    # Let's say we pick the first agent as the root for simplicity.
    return agents[0] if agents else None