from fair.nashwelfare import rv_approximating_nash_welfare, max_nash_welfare_brute_force
import numpy as np
from fair import ValuationMatrix
from fair import Allocation, AllocationMatrix, ListBundle
import fair
import networkx as nx
from fair.utils import *

divide = fair.divide

def efx_with_bounded_charity(v: ValuationMatrix, input_data=None) -> Allocation:
    """
    This function initializes the allocation process to find an EFX (Envy-Free up to any item) 
    allocation with bounded charity.

    Parameters:
    - v: ValuationMatrix object containing valuations of items for each agent.
    - input_data: Additional data required for allocation, if any.

    Returns:
    - x_alloc: A 2D numpy array representing the allocation of items to agents.
    """
    # Initialize the allocation matrix with zeros (no items allocated yet)
    x_alloc = np.zeros_like(v._v)
    # Initialize a set of all unallocated items
    unallocated = set(range(v.num_of_objects))
    # Loop until all items are allocated or no further EFX allocation is possible
    while unallocated:
        # Check if any update rules are applicable and apply them
        # This is a placeholder for your update rules logic
        updated, x_alloc, unallocated = apply_update_rules(x_alloc, unallocated, v, input_data)

        if not updated:
            # If no rules can be applied, exit the loop
            break
        
        # Decycle the envy-graph if necessary to maintain the invariant
        x_alloc = decycle_envy_graph(x_alloc, v)

    # Return the current allocation matrix
    return x_alloc

def apply_update_rules(x_alloc, unallocated, v: ValuationMatrix, input_data):
    updated = False
    # Apply the update rule 1 to maintain EFX
    condition_efx_for_g, agent_i, item_g = update_rule_efx(x_alloc, v, unallocated, input_data)
    if condition_efx_for_g:
        x_alloc[agent_i][item_g] = 1
        unallocated.remove(item_g)
        updated = True
        return updated, x_alloc, unallocated
    # Apply the update rule 2 to maintain EFX
    condition_envy_pool = update_rule_envy_pool(x_alloc, v, unallocated)
    if condition_envy_pool:
        x_alloc, unallocated = update_rule_2(x_alloc, v, unallocated)
        updated = True
        return updated, x_alloc, unallocated
    
    return updated, x_alloc, unallocated
    
    
def update_rule_efx(allocation, v: ValuationMatrix, unallocated_pool, input_data):
    """
    This function applies the update rule 1 to maintain an EFX allocation.
    
    Parameters:
    - allocation: The current allocation of goods to agents.
    - v: ValuationMatrix object containing valuations of items for each agent.
    - unallocated_pool: A list of indices representing unallocated items.
    - input_data: Additional data required for the allocation, if any.
    
    Returns:
    - A tuple containing:
        - condition_efx_for_g: A boolean indicating whether a good g was successfully allocated.
        - agent_i: The index of the agent to whom g was allocated.
        - item_g: The index of the good that was allocated.
    """
    # Convert the current allocation to the Allocation class
    alloc_x = convert_to_allocation(v, allocation, input_data)
    
    for item_g in unallocated_pool:
        for agent_i in range(v.num_of_agents):
            # Create a new allocation by adding the unallocated good to the agent's allocation
            new_allocation = allocation.copy()
            new_allocation[agent_i][item_g] = 1  # Assign the item to the agent
            new_alloc_x = convert_to_allocation(v, new_allocation, input_data)
            
            # Check if the new allocation is EFX
            if is_efx(new_alloc_x):
                # Return the successful allocation
                return True, agent_i, item_g
    
    # If no good can be allocated to maintain EFX, return False
    return False, None, None

def convert_unallocated_pool_to_list(unallocated_pool, num_items) -> np.array:
    # Initialize an array with zeros
    allocation_array = np.zeros(num_items, dtype=int)
    
    # Set the positions of unallocated items to 1
    for item in unallocated_pool:
        allocation_array[item] = 1
        
    return allocation_array


def convert_list_to_unallocated_pool(unallocated_list) -> np.array:
    # Initialize an empty set
    unallocated_pool = set()
    
    # Add the indices of unallocated items to the set
    for item, value in enumerate(unallocated_list):
        if value == 1:
            unallocated_pool.add(item)

    return unallocated_pool

    
def update_rule_envy_pool(alloc_x, v: ValuationMatrix, unallocated_pool):
    """
    Determines if there is an agent who envies the pool of unallocated items more than their 
    current allocation.

    Parameters:
    - alloc_x: A numpy array representing the current allocation of items to agents.
    - v: ValuationMatrix object containing valuations.
    - unallocated_pool: A set of indices representing unallocated items.

    Returns:
    - True if there is such an agent, False otherwise.
    """
    # Calculate the valuation for each agent's current allocation
    current_valuations = np.sum(alloc_x * v._v, axis=1)
    # Initialize an array to hold the valuation of the pool for each agent
    pool_valuations = np.zeros(v.num_of_agents)
    # Calculate the valuation of the unallocated pool for each agent
    for item in unallocated_pool:
        pool_valuations += v._v[:, item]
    # Check if there's any agent who values the pool more than their current allocation
    return np.any(pool_valuations > current_valuations)

def update_rule_2(alloc_x, v: ValuationMatrix, unallocated_list):
    last_unalloc = convert_unallocated_pool_to_list(unallocated_list, v.num_of_objects)
    z_mimimal_envied_subset, most_envious_agent = find_inclusion_wise_minimal_envied_subset(
        alloc_x, v, last_unalloc)
    new_alloc_x = np.copy(alloc_x)
    new_alloc_x[most_envious_agent] = z_mimimal_envied_subset
    new_unalloc = np.logical_or(new_alloc_x[most_envious_agent],
                             np.logical_and(last_unalloc, np.logical_not(z_mimimal_envied_subset))).astype(int)
    
    return new_alloc_x, convert_list_to_unallocated_pool(new_unalloc)

def create_envy_graph_from_matrices(allocation, valuation_matrix: ValuationMatrix):
    # Create a new directed graph
    G_X = nx.DiGraph()
    num_agents = valuation_matrix.num_of_agents

    # Iterate over all pairs of agents
    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                # Compute the valuations for each agent's bundle
                valuation_i_for_i = np.sum(valuation_matrix[i] * allocation[i])
                valuation_i_for_j = np.sum(valuation_matrix[i] * allocation[j])

                # If agent i envies agent j, add a directed edge
                if valuation_i_for_i < valuation_i_for_j:
                    G_X.add_edge(i, j)

    return G_X

def decycle_envy_graph(allocation, valuation_matrix: ValuationMatrix):
    envy_graph = create_envy_graph_from_matrices(allocation, valuation_matrix)
    


def find_inclusion_wise_minimal_envied_subset(alloc, v: ValuationMatrix, pool):
    """
    Finds an inclusion-wise minimal envied subset for the given agent.
    """
    num_agents = v.num_of_agents
    num_items = v.num_of_objects
    
    most_envious_agents = -1
    
    Z = np.copy(pool)
    for agent in range(num_agents):
        for item in range(num_items):
            if pool[item] == 0:
                continue
            temp_pool = np.copy(Z)
            temp_pool[item] = 0
            if sum(v[agent] * alloc[agent]) < sum(v[agent] * temp_pool):
                Z = temp_pool
                most_envious_agents = agent

    return Z, most_envious_agents