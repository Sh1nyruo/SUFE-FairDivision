from fair.nashwelfare import rv_approximating_nash_welfare, max_nash_welfare_brute_force
from fair.nashwelfare.round_robin_alg import round_robin
import numpy as np
from fair import ValuationMatrix
from fair import Allocation, AllocationMatrix, ListBundle
import fair
import networkx as nx
from fair.utils import *
from typing import Callable, Any

import fair.utils

divide = fair.divide

def divsion_with_donating(v: ValuationMatrix, NSW_algorithm: Callable = round_robin, optimal = False, input_data = None, delta = 0.1, ) -> Allocation:
    if v.num_of_objects <= 1:
        return max_nash_welfare_brute_force(v)
    if optimal:
        return division_optimal(v, input_data = input_data)
    else:
        return division_suboptimal(v, NSW_algorithm, input_data, delta = delta)

def division_optimal(v: ValuationMatrix, input_data = None) -> Allocation:
    # Compute an allocation Z that maximizes Nash welfare
    Z = divide(max_nash_welfare_brute_force, input = input_data)
    #print("Initial allocation: ", Z)
    # Initialize matching M
    matching = {}
    touched_bundles = []
    donations = []
    # Loop until every agent is matched with a bundle (|M| = n)
    while len(matching) < v.num_of_agents:
        # Initialize Y as the allocation matrix of Z (numpy 2d array)
        Y = Z.matrix._z
        # Build the EFX feasibility graph for the current Z
        G = build_efx_feasibility_graph(Z, touched_bundles, v)
        #print("Graph: ", G.edges(data=True))
        # Find the matching in G that satisfies the conditions a-c
        matching = find_weighted_matching(G)
        #print("Matching: ", matching)
        # Extract agent indices from the matching
        matched_agents = set(int(agent.split('_')[1]) for agent in matching.keys())
        # Find agents that are not matched
        unmatched_agents = set(range(v.num_of_agents)) - matched_agents
        #print("Unmatched agents: ", unmatched_agents)
        # If all agents are matched, we are done
        if not unmatched_agents:
            break
        # If there are agents not matched in M, find their robust demand bundles
        for agent in unmatched_agents:
            Z_bundles = robust_demand(agent, Y, v)
            #print("Robust demand: ", Z_bundles)
            Z = update_bundle(agent, Z_bundles, Y, input_data, donations, touched_bundles)
            #print("Updated allocation: ", Z)
            break
    # print(matching)
    # print("Donations: ", donations)
    # After the loop, Y contains the final allocation
    return convert_matching_to_allocation(matching, v, Z)


def division_suboptimal(v: ValuationMatrix, NSW_algorithm: Callable, input_data = None,  delta = 0.1) -> Allocation:
    # Compute an allocation Z that maximizes Nash welfare
    X = divide(NSW_algorithm, input_data)

    # X = divide(random_nash_welfare, input = input_data)
    Z = create_Allocation(X, v, input_data)
    delta_1 = 2 * delta / (1 - delta)
    M_0 = create_identity_matching(v.num_of_agents)
    #print("Initial allocation: ", Z)
    # Initialize matching M
    matching = {}
    touched_bundles = []
    donations = []
    # Loop until every agent is matched with a bundle (|M| = n)
    while len(matching) < v.num_of_agents:
        # Initialize Y as the allocation matrix of Z (numpy 2d array)
        Y = Z.matrix._z
        # Build the EFX feasibility graph for the current Z
        G = build_efx_feasibility_graph(Z, touched_bundles, v)
        #nx.draw(G, with_labels=True)
        #print(nx.to_dict_of_dicts(G))
        # Find the matching in G that satisfies the conditions a-c
        matching = find_weighted_matching(G)
        #print(matching)
        # Extract agent indices from the matching
        # Find all bundles that are currently matched
        matched_bundles = set(int(agent.split('_')[1]) for agent in matching.values())
        # Find the first unmatched bundle Zj1
        unmatched_bundles = set(range(v.num_of_agents)) - matched_bundles
        # If all agents are matched, we are done
        if not unmatched_bundles:
            break
        # If there are agents not matched in M, find their robust demand bundles
        if unmatched_bundles:
            #fair.utils.append_to_file("input_donation.txt", input_data)
            Zj1 = next(iter(unmatched_bundles))
            itemremoved = True
            while itemremoved:
                augmenting_path, j_k = find_augmenting_path(Zj1, matching, v.num_of_agents)
                #print("Augmenting path: ", augmenting_path)
                #print("Matching: ", matching)
                Z_jstar = robust_demand(j_k, Y, v)
                #print("J_k: ", j_k)
                #print("Z_jstar: ", Z_jstar)
                # Check if Z_jstar is part of the augmenting path P
                # Find the agent index for Z_jstar in the augmenting path, if it exists
                agent_j0 = None
                for agent, bundle in augmenting_path:
                    if bundle == f'bundle_{Z_jstar}':
                        agent_j0 = agent
                        break
                # If Z_jstar is in the augmenting path, update the matching
                if agent_j0 is not None:
                    # Remove the current match for Z_jstar and add the new match with j_k
                    matching.pop(agent_j0)  # Remove the existing match
                    matching[f'agent_{j_k}'] = f'bundle_{Z_jstar}'  # Add the new match
                    #print("Matching: ", matching)
                else:
                    Z = update_bundle(j_k, Z_jstar, Y, input_data, donations, touched_bundles)

                    if (2 + delta_1) * Z.utility_profile_matrix()[Z_jstar][Z_jstar] < X.utility_profile_matrix()[Z_jstar][Z_jstar]:
                        #fair.utils.append_to_file("input_improving_rv.txt", input_data)
                        #print("After removal: ", Z.utility_profile_matrix()[Z_jstar][Z_jstar])
                        #print("Initial: ", X.utility_profile_matrix()[Z_jstar][Z_jstar])
                        # Extracting all j_i from the augmenting path
                        j_values = [int(j.split('_')[1]) for i, j in augmenting_path]
                        initial_allocation = X.matrix._z
                        X_bar = np.copy(initial_allocation)
                        if len(j_values) >= 1:
                            j_1 = int(augmenting_path[0][0].split('_')[1])
                            X_bar[j_1] = np.logical_or(initial_allocation[j_1], Z.matrix._z[j_values[0]]).astype(int)
                            for i in range(0, len(j_values) - 1):
                                X_bar[j_values[i]] = np.logical_or(np.logical_and(initial_allocation[j_values[i]], np.logical_not(Z.matrix._z[j_values[i]])),
                                                                   Z.matrix._z[j_values[i+1]]).astype(int)
                            X_bar[j_k] = np.logical_or(np.logical_and(initial_allocation[j_k], np.logical_not(Z.matrix._z[j_k])),
                                                                   Z.matrix._z[Z_jstar]).astype(int)
                            X_bar[Z_jstar] = np.logical_and(initial_allocation[Z_jstar], np.logical_not(Z.matrix._z[Z_jstar])).astype(int)
                        else:
                            X_bar[j_k] = np.logical_or(np.logical_or(np.logical_and(initial_allocation[j_k], np.logical_not(Z.matrix._z[j_k])),
                                                                   Z.matrix._z[Z_jstar]), X_bar[j_k]).astype(int)
                            X_bar[Z_jstar] = np.logical_and(initial_allocation[Z_jstar], np.logical_not(Z.matrix._z[Z_jstar])).astype(int)

                        return X_bar
                    itemremoved = False

    # After the loop, Y contains the final allocation
    return convert_matching_to_allocation(matching, v, Z)

def build_efx_feasibility_graph(Z: Allocation, T: list[any], v: ValuationMatrix):
    """
    Build a weighted EFX feasibility graph where edges represent EFX feasible allocations between agents and bundles.
    The weights of the edges are higher if the bundle is 'touched'.

    :param Z: The current allocation instance containing the bundles assigned to each agent.
    :param T: The list containing the indices of touched bundles.
    :param v: The valuation matrix of agents over items.
    :return: A bipartite graph representing EFX feasible allocations with weights.
    """
    G =nx.Graph()
    num_agents = v.num_of_agents
    num_items = v.num_of_objects
    
    # Add agents and bundles as nodes in the graph
    agents = [f'agent_{i}' for i in range(num_agents)]
    bundles = [f'bundle_{i}' for i in range(num_agents)]  # Assuming each agent initially has one bundle
    G.add_nodes_from(agents, bipartite=0)
    G.add_nodes_from(bundles, bipartite=1)
    
    # Define the weight
    weight_a = num_agents ** 4 
    weight_b = num_agents ** 2 
    weight_c = 1

    # Add edges with weights based on EFX feasibility and whether the bundle is touched
    for agent_index in range(num_agents):
        for bundle_index in range(num_agents):  # Assuming bundle indices correspond to agents
            if is_efx_feasible(agent_index, bundle_index, Z, v):
                # Check if the bundle is touched and assign weight accordingly
                weight = weight_c
                if bundle_index in T:
                    weight += weight_a
                if bundle_index == agent_index:
                    weight += weight_b
                G.add_edge(f'agent_{agent_index}', f'bundle_{bundle_index}', weight=weight)
    return G
    
def is_efx_feasible(agent_index, bundle_index, Z: Allocation, v: ValuationMatrix) -> bool:
    """
    Check if a given bundle is EFX feasible for an agent.

    :param agent_index: Index of the agent.
    :param bundle_index: Index of the bundle.
    :param Y: The current allocation matrix.
    :param v: The valuation matrix.
    :return: True if the bundle is EFX feasible for the agent, False otherwise.
    """

    condition_a = Z.agents[agent_index].is_EFx(Z.bundles[bundle_index], Z.bundles)
    
    condition_b = True if (agent_index == bundle_index) or (Z.agent_bundle_value_matrix[agent_index][bundle_index] > Z.agent_bundle_value_matrix[agent_index][agent_index]) else False
    
    return condition_a and condition_b

def find_weighted_matching(G: nx.Graph):
    """
    Find a maximum weight matching in the given weighted bipartite graph.

    :param G: A weighted bipartite graph where nodes represent agents and bundles,
              and edges represent EFX feasible allocations with weights.
    :return: A dictionary representing the matching. Keys are agents, and values are bundles.
    """
    # Find the maximum weight matching in the graph
    # This function returns a set of frozensets, where each frozenset contains two elements that are matched
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)

    # Convert the matching into a more usable format
    matching_dict = {}
    for agent, bundle in matching:
        if 'agent_' in agent:
            matching_dict[agent] = bundle
        else:
            # If the first item is not the agent, swap the order
            matching_dict[bundle] = agent

    return matching_dict

def robust_demand(agent_index, allocation, valuations):
    """
    Find the robust demand bundle for the given agent. The robust demand bundle is the one which,
    if the least valued item is removed, provides the agent with the highest valuation.

    :param agent_index: Index of the agent.
    :param allocation: Current allocation matrix where rows correspond to agents and columns to items.
    :param valuations: Valuation matrix for all agents over items.
    :return: Index of the robust demand bundle.
    """
    best_bundle_value = -1
    best_bundle_index = -1

    for bundle_index, bundle in enumerate(allocation):

        least_valued_item_index = find_least_valued_item(agent_index, bundle, valuations)

        # If the bundle is empty or the agent has no allocated items, continue to the next bundle
        if least_valued_item_index is None:
            continue

        # Calculate the valuation of the bundle without the least valued item
        bundle_valuation = sum(
            valuations[agent_index][i] for i, allocated in enumerate(bundle) if allocated and i != least_valued_item_index
        )

        # Update the best bundle if this one has a higher valuation
        if bundle_valuation > best_bundle_value:
            best_bundle_value = bundle_valuation
            best_bundle_index = bundle_index

    return best_bundle_index

def update_bundle(agent_index, bundle_index, allocation, input, donations, touched_bundles):
    """
    Update the allocation matrix with the new bundle for the given agent.

    :param agent_index: Index of the agent.
    :param bundle_index: Index of the new bundle.
    :param allocation: Current allocation matrix.
    :param input_data: The input data for the division algorithm.
    :return: The updated allocation matrix.
    """
    # Step 1. Adapt the input:
    valuation_matrix = list_of_valuations = object_names = agent_names = None
    if isinstance(input, ValuationMatrix): # instance is already a valuation matrix
        valuation_matrix = input
    elif isinstance(input, np.ndarray):    # instance is a numpy valuation matrix
        valuation_matrix = ValuationMatrix(input)
    elif isinstance(input, list) and isinstance(input[0], list):            # list of lists
        list_of_valuations = input
        valuation_matrix = ValuationMatrix(list_of_valuations)
    elif isinstance(input, dict):  
        agent_names = list(input.keys())
        list_of_valuations = list(input.values())
        if isinstance(list_of_valuations[0], dict): # maps agent names to dicts of valuations
            object_names = list(list_of_valuations[0].keys())
            list_of_valuations = [
                [valuation[object] for object in object_names]
                for valuation in list_of_valuations
            ]
        valuation_matrix = ValuationMatrix(list_of_valuations)
    else:
        raise TypeError(f"Unsupported input type: {type(input)}")
    # Create a copy of the allocation matrix to update
    updated_allocation = np.copy(allocation)
    
    # Identify the least valued item in the agent's current bundle
    least_valued_item_index = find_least_valued_item(agent_index, allocation[bundle_index], valuation_matrix)
    donations.append(least_valued_item_index)
    
    if least_valued_item_index is not None:  # Check to ensure there was a least valued item
        updated_allocation[:, least_valued_item_index] = 0
        touched_bundles.append(bundle_index)
    
    output = updated_allocation
    
    # Step 3. Adapt the output:
    if isinstance(output, Allocation):
        return output
    
    if agent_names is None:
        agent_names = [f"Agent #{i}" for i in valuation_matrix.agents()]
        
    if isinstance(output, np.ndarray) or isinstance(output, AllocationMatrix):  # allocation matrix
        allocation_matrix = AllocationMatrix(output)
        if isinstance(input, dict):
            # Since items are indivisible, convert allocation matrix to list of allocated items per agent
            dict_of_bundles = {}
            for i, agent_name in enumerate(agent_names):
                # Find which items are allocated to this agent (non-zero entries in allocation matrix)
                allocated_items = [object_names[j] for j, allocation in enumerate(allocation_matrix[i]) if allocation > 0]
                dict_of_bundles[agent_name] = ListBundle(allocated_items)
            return Allocation(input, dict_of_bundles, matrix=allocation_matrix)
        else:
            return Allocation(valuation_matrix, allocation_matrix)
    elif isinstance(output, list):
        if object_names is None:
            list_of_bundles = output
        else:
            list_of_bundles = [
                [object_names[object_index] for object_index in bundle]
                for bundle in output
            ]
        dict_of_bundles = dict(zip(agent_names,list_of_bundles))
        return Allocation(input if isinstance(input,dict) else valuation_matrix, dict_of_bundles)
    else:
        raise TypeError(f"Unsupported output type: {type(output)}")

def convert_matching_to_allocation(matching, v: ValuationMatrix, Z: Allocation):
    """
    Convert the matching result into an allocation matrix.

    :param matching: Dictionary with matched agents and bundles.
    :param v: ValuationMatrix object for agents' valuations.
    :param Z: Allocation object containing the initial allocation matrix.
    :return: A new Allocation object with the updated allocation matrix based on matching.
    """
    # Initialize a zero matrix for the allocation
    alloc = np.zeros_like(Z.matrix._z)

    # Update the allocation matrix based on the matching
    for agent_str, bundle_str in matching.items():
        # Extract numerical indices from the string identifiers
        agent_index = int(agent_str.replace('agent_', ''))
        bundle_index = int(bundle_str.replace('bundle_', ''))

        # Assign the bundle to the agent in the allocation matrix
        alloc[agent_index] = Z.matrix._z[bundle_index]

    # Assuming Allocation class has a way to handle np.ndarray directly
    # If not, adjust the return statement as needed to fit your Allocation class constructor
    return alloc

def create_identity_matching(num_agents):
    """
    Create an identity matching where each agent is matched with their corresponding bundle.

    :param num_agents: The number of agents (and bundles).
    :return: A dictionary representing the identity matching.
    """
    M0 = {f'agent_{i}': f'bundle_{i}' for i in range(num_agents)}
    return M0

def find_augmenting_path(Zj1, matching, num_agents):
    """
    Find the augmented path starting from an unmatched bundle Zj1.

    :param Zj1: The index of the unmatched bundle from which to start the path.
    :param matching: The current matching dictionary where keys are agents and values are bundles.
    :param num_agents: Total number of agents (and bundles).
    :return: The augmented path as a list of tuples (agent, bundle).
    """
    # Initialize the augmented path
    augmented_path = []
    
    # Current agent being considered
    current_agent_index = Zj1
    
    # While we haven't found an unmatched agent
    while True:
        # Check if current agent is matched
        if f'agent_{current_agent_index}' in matching:
            # Get the bundle matched to the current agent
            current_bundle = matching[f'agent_{current_agent_index}']
            augmented_path.append((f'agent_{current_agent_index}', current_bundle))
            
            # Find the next agent to consider by retrieving the agent index from the matched bundle
            current_agent_index = int(current_bundle.split('_')[1])
        else:
            # If the agent is unmatched, we've reached the end of the augmented path
            break
    
    return augmented_path, current_agent_index

def create_Allocation(X: Allocation, v: ValuationMatrix, input) -> Allocation:
    # Step 1. Adapt the input:
    valuation_matrix = list_of_valuations = object_names = agent_names = None
    if isinstance(input, ValuationMatrix): # instance is already a valuation matrix
        valuation_matrix = input
    elif isinstance(input, np.ndarray):    # instance is a numpy valuation matrix
        valuation_matrix = ValuationMatrix(input)
    elif isinstance(input, list) and isinstance(input[0], list):            # list of lists
        list_of_valuations = input
        valuation_matrix = ValuationMatrix(list_of_valuations)
    elif isinstance(input, dict):  
        agent_names = list(input.keys())
        list_of_valuations = list(input.values())
        if isinstance(list_of_valuations[0], dict): # maps agent names to dicts of valuations
            object_names = list(list_of_valuations[0].keys())
            list_of_valuations = [
                [valuation[object] for object in object_names]
                for valuation in list_of_valuations
            ]
        valuation_matrix = ValuationMatrix(list_of_valuations)
    else:
        raise TypeError(f"Unsupported input type: {type(input)}")
    
    output = np.copy(X.matrix._z)
    
    # Step 3. Adapt the output:
    if isinstance(output, Allocation):
        return output
    
    if agent_names is None:
        agent_names = [f"Agent #{i}" for i in valuation_matrix.agents()]
        
    if isinstance(output, np.ndarray) or isinstance(output, AllocationMatrix):  # allocation matrix
        allocation_matrix = AllocationMatrix(output)
        if isinstance(input, dict):
            # Since items are indivisible, convert allocation matrix to list of allocated items per agent
            dict_of_bundles = {}
            for i, agent_name in enumerate(agent_names):
                # Find which items are allocated to this agent (non-zero entries in allocation matrix)
                allocated_items = [object_names[j] for j, allocation in enumerate(allocation_matrix[i]) if allocation > 0]
                dict_of_bundles[agent_name] = ListBundle(allocated_items)
            return Allocation(input, dict_of_bundles, matrix=allocation_matrix)
        else:
            return Allocation(valuation_matrix, allocation_matrix)
    elif isinstance(output, list):
        if object_names is None:
            list_of_bundles = output
        else:
            list_of_bundles = [
                [object_names[object_index] for object_index in bundle]
                for bundle in output
            ]
        dict_of_bundles = dict(zip(agent_names,list_of_bundles))
        return Allocation(input if isinstance(input,dict) else valuation_matrix, dict_of_bundles)
    else:
        raise TypeError(f"Unsupported output type: {type(output)}")
