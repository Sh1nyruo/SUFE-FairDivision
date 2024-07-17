import numpy as np
from fair import ValuationMatrix
from fair.nashwelfare.approximation_nash_welfare import MarketModel
import networkx as nx
import random
from fair.nashwelfare.tree import *
import math

import logging
logger = logging.getLogger(__name__)

def calculate_nash_welfare(allocation, values, progress_updater=None):
    """
    Calculates the Nash welfare from a given allocation matrix and valuation matrix.

    :param allocation: A NumPy array representing the allocation matrix where each row corresponds to an agent and each column to an item.
    :param values: A ValuationMatrix object containing the values each agent assigns to each item.
    :param progress_updater: An optional callback function to update the progress of the computation.
    
    :return: The calculated Nash welfare as a float.
    """
    # Update progress
    if progress_updater is not None:
        progress_updater.update(1)
        
    utilities = np.sum(values._v * allocation, axis=1, keepdims=False)
    # Calculate Nash welfare as the product of utilities
    nash_welfare = np.sum(np.log(utilities)) if all(utilities) else 0  # If any utility is 0, Nash welfare is 0

    return nash_welfare

def assign_items(allocation, values, current_item, max_welfare, best_allocation, progress_updater=None):
    """
    Recursive function to assign items to agents and calculate Nash welfare, backtracking to explore all possible allocations.

    :param allocation: Current allocation matrix being considered.
    :param values: ValuationMatrix object representing agents' valuations of items.
    :param current_item: Index of the current item being assigned.
    :param max_welfare: List containing the maximum Nash welfare found so far.
    :param best_allocation: Allocation matrix corresponding to the maximum Nash welfare found so far.
    :param progress_updater: Optional callback function for updating progress.
    """
    num_agents, num_items = values.num_of_agents, values.num_of_objects

    # If all items are assigned, calculate Nash welfare and update best allocation if necessary
    if current_item == num_items:
        current_welfare = calculate_nash_welfare(allocation, values, progress_updater)
        if current_welfare > max_welfare[0]:
            max_welfare[0] = current_welfare
            best_allocation[:] = allocation[:]
        return

    
    # Try assigning current item to each agent and backtrack
    for agent in range(num_agents):
        allocation[agent, current_item] = 1  # Assign item to agent
        assign_items(allocation, values, current_item + 1, max_welfare, best_allocation, progress_updater)
        allocation[agent, current_item] = 0  # Backtrack


def max_nash_welfare_brute_force(v:ValuationMatrix) -> np.array:
    """
    Finds the allocation that maximizes Nash welfare using a brute-force approach.

    :param v: ValuationMatrix object representing agents' valuations of items.
    
    :return: An allocation matrix representing the optimal allocation of items to agents.
    """
    num_agents = v.num_of_agents
    num_items = v.num_of_objects
    allocation = np.zeros((num_agents, num_items), dtype=int)
    max_welfare = [-1]  # Use list for mutable reference
    best_allocation = np.zeros((num_agents, num_items), dtype=int)
    
    # Initialize progress bar
    total_combinations = num_agents ** num_items
    
    #with tqdm(total=total_combinations, desc="Calculating Nash Welfare") as pbar:
    assign_items(allocation, v, 0, max_welfare, best_allocation)

    return best_allocation
    
    
def rv_approximating_nash_welfare(v: ValuationMatrix) -> np.array:
    """
    Compute an allocation of items to agents that approximates the Nash welfare.

    This function implements the spending-restricted rounding algorithm which begins
    by finding a fractional allocation that maximizes the Nash welfare under spending 
    restrictions and then rounds it to an integral allocation.

    :param v: A ValuationMatrix object containing agents' valuations of items.
              This object must have methods or attributes that allow access to the number
              of agents, the number of items, and the valuation each agent assigns to each item.

    :return: A 2D numpy array representing the integral allocation of items to agents.
             Each row corresponds to an agent and each column to an item, with a 1
             indicating that the item has been allocated to the agent and a 0 otherwise.
    """
    Market_Model = MarketModel(v)
    Market_Model.weakly_polynomial_algorithm()
    Market_SRR = Market_Model.spending_restricted_outcome()
    forest = convert_to_forest(Market_Model, Market_SRR)
    integral_allocation = spending_restricted_rounding(forest, Market_Model, v, Market_SRR)
    return integral_allocation
        

def convert_to_forest(model: MarketModel, fractional_allocation: np.array) -> nx.DiGraph:
    """
    Convert the allocation to a forest.
    :param m: a MarketModel object
    :param allocation_matrix: a matrix alloc of a similar shape in which alloc[i][j] is the fraction allocated to agent i from object j.
    :return: a directed graph G in which each node represents an agent or an object, and each edge represents an allocation.
    """
    # Initialize a new graph
    forest = nx.Graph()
    
    # Add nodes and edges to the graph based on the fractional allocation
    for agent_index, items_spent in enumerate(fractional_allocation):
        for item_index, amount_spent in enumerate(items_spent):
            if amount_spent > 0:
                # Add an edge between agent and item with the amount spent as the weight
                forest.add_edge(f'agent_{agent_index}', f'item_{item_index}', weight=amount_spent)

    # Find the connected components of the graph, each will be a separate tree in the forest
    trees = [forest.subgraph(c).copy() for c in nx.connected_components(forest)]

    return trees

def spending_restricted_rounding(trees, model: MarketModel, v: ValuationMatrix, Market_SRR) -> np.array:
    """
    Perform spending-restricted rounding on the forest of trees to get an integral allocation.
    
    :param trees: A list of trees, each representing a spending graph.
    :param v: A ValuationMatrix object containing the valuations of agents for items.
    :return: A 2D numpy array representing the integral allocation.
    """
    # Initialize the integral allocation matrix with zeros
    num_agents = model.n
    num_items = model.m
    integral_allocation = np.zeros((num_agents, num_items), dtype=int)
    cnt = 0
    # Convert the trees from NetworkX graphs to our tree structure and perform assignments
    for nx_tree in trees:
        # Skip single-node trees
        if len(nx_tree.nodes) == 1:
            continue
        # Find the root agent for each tree and build the tree structure
        root_agent = choose_root_agent(nx_tree)
        tree_root = build_tree_structure(nx_tree, root_agent)
        
        # Now we have a tree structure, we can perform steps 3 and 4
        items_to_remove = set()
        assign_leaf_low_price_items(tree_root, int(root_agent.split('_')[1]), model, integral_allocation, items_to_remove)
        #print("Current Allocation: ", integral_allocation)
        # Now remove the items from the NetworkX graph
        nx_tree.remove_nodes_from(items_to_remove)

        bipartite_graph = create_weighted_bipartite_graph(nx_tree, model, v, integral_allocation, Market_SRR)
        #nx.draw(bipartite_graph, with_labels=True, font_weight='bold')
        #print(nx.get_edge_attributes(bipartite_graph, 'weight'))
        bi_matching = nx.max_weight_matching(bipartite_graph, maxcardinality=True, weight='weight')
        
        for agent_node, item_node in bi_matching:
            # Extract indices from the node labels
            if 'dummy' in item_node or 'dummy' in agent_node:
                continue
            if 'agent' in agent_node and 'item' in item_node:
                agent_idx = int(agent_node.split('_')[1])
                item_idx = int(item_node.split('_')[1])
            elif 'item' in agent_node and 'agent' in item_node:  # Order in pair could be reversed
                agent_idx = int(item_node.split('_')[1])
                item_idx = int(agent_node.split('_')[1])
            else:
                continue  # Skip if it doesn't conform to the expected pattern
            
            # Assign the item to the agent in the integral allocation matrix
            integral_allocation[agent_idx, item_idx] = 1

        
    unassigned_items = [j for j in range(num_items) if not integral_allocation[:, j].any()]
    # Copy the initial allocation to avoid modifying it directly
    current_allocation = np.copy(integral_allocation)
    #print("current_allocation", current_allocation.sum())
    max_welfare = [-1]  # Use list for mutable reference
    best_allocation = np.copy(integral_allocation)  # Start with the initial allocation
    #print("unassigned_items", unassigned_items)
    # Initialize progress bar
    total_combinations = num_agents ** np.count_nonzero(np.sum(current_allocation, axis = 0) == 0)
    # with tqdm(total=total_combinations, desc="Calculating Nash Welfare") as pbar:
    assign_rest_items(current_allocation, v, unassigned_items, max_welfare, best_allocation)
    #print("best_allocation", best_allocation.sum(axis=0))
    return best_allocation

def assign_rest_items(allocation, values: ValuationMatrix, unassigned_items, max_welfare, best_allocation,  progress_updater=None):
    """
    Assign the remaining unassigned items to agents using brute force.
    
    :param allocation: Current state of item allocation.
    :param values: 2D numpy array of agents' valuations for items.
    :param unassigned_items: List of item indices that are yet to be assigned.
    :param max_welfare: Current maximum Nash welfare found.
    :param best_allocation: Best allocation corresponding to max_welfare.
    :param agent_index: Index of the agent to whom items are currently being assigned.
    """
    # If there are no unassigned items left, compute Nash welfare
    if not unassigned_items:
        current_welfare = calculate_nash_welfare(allocation, values, progress_updater)
        if current_welfare > max_welfare[0]:
            max_welfare[0] = current_welfare
            # Update best_allocation with the current allocation
            np.copyto(best_allocation, allocation)
        return

    # Assign next unassigned item to each agent
    item_to_assign = unassigned_items[0]
    for agent in range(values.num_of_agents):
        allocation[agent, item_to_assign] = 1
        assign_rest_items(
            allocation, values, unassigned_items[1:], max_welfare, best_allocation, progress_updater=progress_updater
        )
        allocation[agent, item_to_assign] = 0  # Reset the item assignment (backtrack)


def assign_leaf_low_price_items(tree_node, parent_agent_index, model: MarketModel, integral_allocation, items_to_remove):
    """
    Recursively assign leaf items and items with price <= 0.5 to their parent-agent.
    """
    # If the current node is an item, process it
    if 'item' in tree_node.value:
        item_index = int(tree_node.value.split('_')[1])
        
        # Assign leaf item to its parent-agent
        if not tree_node.children:
            integral_allocation[parent_agent_index, item_index] = 1
            items_to_remove.add(tree_node.value)
        
        # Independently, assign the item if its price <= 0.5 to its parent-agent
        if model.price[item_index] <= 0.5:
            # print("item_index: ", item_index)
            integral_allocation[parent_agent_index, item_index] = 1
            items_to_remove.add(tree_node.value)

    # Traverse the children, updating the parent agent index if necessary
    for child in tree_node.children:
        # If the child is an agent, it becomes the new parent as we go down
        new_parent_agent_index = parent_agent_index
        if 'agent' in child.value:
            new_parent_agent_index = int(child.value.split('_')[1])
        assign_leaf_low_price_items(child, new_parent_agent_index, model, integral_allocation, items_to_remove)

def add_nodes_edges_from_tree(node, B, model, v, integral_allocation, Market_SRR, current_allocation={}):
    # Base case: if node is None
    if not node:
        return
    
    if 'agent' in node.value:
        agent_idx = int(node.value.split('_')[1])
        B.add_node(node.value, bipartite=0)  # Agent partition
        # Keep track of the total value allocated to this agent
        current_allocation[node.value] = max(sum(v[agent_idx, :] * integral_allocation[agent_idx, :]), 1)
        
        dummy_node = 'dummy_node_' + str(agent_idx)
        B.add_node(dummy_node, bipartite=1)  # Dummy node partition
        weight = math.log(current_allocation[node.value])
        B.add_edge(node.value, dummy_node, weight=weight)

    elif 'item' in node.value:
        item_idx = int(node.value.split('_')[1])
        B.add_node(node.value, bipartite=1)  # Item partition
        
        # Add an edge from the item to the parent agent with the appropriate weight
        parent_agent_idx = int(node.parent.value.split('_')[1])
        weight = math.log(v[parent_agent_idx, item_idx] + current_allocation[node.parent.value])
        B.add_edge(node.parent.value, node.value, weight=weight)
        
        if node.children:
            max_spending_child_idx = 0
            max_spend = -1
            for child in node.children:
                child_idx = int(child.value.split('_')[1])
                if Market_SRR[child_idx][item_idx] > max_spend:
                    max_spend = Market_SRR[child_idx][item_idx]
                    max_spending_child_idx = child_idx
            current_allocation[max_spending_child_idx] = max(sum(v[max_spending_child_idx, :] * integral_allocation[max_spending_child_idx, :]), 1)
            child_node = 'agent_' + str(max_spending_child_idx)
            B.add_node(child_node, bipartite=0)
            weight = math.log(v[max_spending_child_idx, item_idx] + current_allocation[max_spending_child_idx])
            B.add_edge(node.value, child_node, weight=weight)

    for child in node.children:
        add_nodes_edges_from_tree(child, B, model, v, integral_allocation, Market_SRR, current_allocation)

def create_weighted_bipartite_graph(nx_tree, model, v, integral_allocation, Market_SRR):
    # Initialize an empty graph for the bipartite structure
    B = nx.Graph()
    
    # Create the tree structure from the nx_tree
    root_agent = choose_root_agent(nx_tree)
    tree_root = build_tree_structure(nx_tree, root_agent)
    
    # Recursively add nodes and edges from the tree to the bipartite graph
    add_nodes_edges_from_tree(tree_root, B, model, v, integral_allocation, Market_SRR)
                    
    return B

        
import logging
logger = logging.getLogger(__name__)


