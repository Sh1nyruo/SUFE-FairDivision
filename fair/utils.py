import json
import random
import numpy as np
from fair import *
import math

def append_to_file(file_path, data):
    with open(file_path, 'a') as file:  # Open file in append mode
        # Convert the dictionary to a JSON-formatted string
        json_data = json.dumps(data)
        file.write(json_data + "\n\n")

def read_json_objects_from_file(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        json_string = ''
        for line in file:
            if line.strip():  # If the line contains text, add it to the current JSON string
                json_string += line
            elif json_string:  # If the line is empty and there's a JSON string, parse it
                try:
                    data_list.append(json.loads(json_string))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e} - JSON string: {json_string}")
                json_string = ''  # Reset for the next JSON object
        # Catch any final JSON object not followed by a blank line
        if json_string:
            try:
                data_list.append(json.loads(json_string))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} - JSON string: {json_string}")
    return data_list

def create_agents_example(num_agents, num_attributes, max_value=100):
    """
    Generates a dictionary of agents with specified attributes, each having random values in the range 0 to 100.

    Parameters:
    - num_agents: Number of agents to generate.
    - num_attributes: Number of attributes each agent should have.

    Returns:
    A dictionary with agent names as keys and another dictionary of their attributes as values.
    """
    # Define the base attribute names
    base_attr_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    # To extend this to 100 items, we will add a number after the letter once we've used all 26 letters once
    extended_attr_names = []
    for i in range(0, math.ceil(num_attributes / len(base_attr_names))):  # Start from 1 to avoid 'a1' and go up to 'z3'
        for attr in base_attr_names:
            extended_attr_names.append(f"{attr}_{i}")
    
    # Select the required number of attributes
    attributes = extended_attr_names[:num_attributes]
    
    # Create the agents dictionary
    agents = {}
    for agent_num in range(1, num_agents + 1):
        agent_name = f"Agent{agent_num}"
        agents[agent_name] = {attr: random.randint(0, max_value) for attr in attributes}

    return agents

def is_efx(alloc: Allocation) -> bool:
    """
    Checks if the given allocation is envy-free up to any good (EFX).

    Parameters:
    - alloc: The allocation to check for EFX.

    Returns:
    True if the allocation is EFX; False otherwise.
    """
    for i, agent in enumerate(alloc.agents):
        if not agent.is_EFx(alloc.bundles[i], alloc.bundles):
            return False
    return True

def is_ef1(alloc: Allocation) -> bool:
    """
    Checks if the given allocation is envy-free up to any good (EFX).

    Parameters:
    - alloc: The allocation to check for EFX.

    Returns:
    True if the allocation is EFX; False otherwise.
    """
    for i, agent in enumerate(alloc.agents):
        if not agent.is_EF1(alloc.bundles[i], alloc.bundles):
            return False
    return True

def is_ef(alloc: Allocation) -> bool:
    """
    Checks if the given allocation is envy-free up to any good (EFX).

    Parameters:
    - alloc: The allocation to check for EFX.

    Returns:
    True if the allocation is EFX; False otherwise.
    """
    for i, agent in enumerate(alloc.agents):
        if not agent.is_EF(alloc.bundles[i], alloc.bundles):
            return False
    return True

def convert_to_allocation(v: ValuationMatrix, alloc: np.array, input) -> Allocation:
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
    
    output = alloc
    
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


def envy_free_up_to_any_item(v: ValuationMatrix, agent, alloc_1, alloc_2) -> bool:
    """
    Checks if the given allocations are envy-free up to any item (EF1).

    Parameters:
    - v: The valuation matrix.
    - agent: The index of the agent to check for EFx.
    - alloc_1: The first allocation to check for EFx.
    - alloc_2: The second allocation to check for EFx.

    Returns:
    True if the allocations are EFx; False otherwise.
    """
    value_of_bundle_1 = np.sum(v[agent] * alloc_1)
    value_of_bundle_2 = np.sum(v[agent] * alloc_2)
    
    least_valued_item_of_bundle_2 = find_least_valued_item(agent, alloc_2, v)
    if least_valued_item_of_bundle_2 is None:
        return value_of_bundle_1 < value_of_bundle_2
    else:
        return value_of_bundle_1 < (value_of_bundle_2 - v[agent][least_valued_item_of_bundle_2])

def find_least_valued_item(agent_index, bundle, valuations):
    """
    Find the index of the least valued item for a specific agent within a specific bundle.

    :param agent_index: Index of the agent.
    :param bundle: Allocation of items to the agent where each item's presence is indicated by a truthy value.
    :param valuations: Valuation matrix for all agents over items.
    :return: Index of the least valued item for the agent. Returns None if no items are allocated.
    """
    # Get the indices of the items allocated to the agent
    allocated_items = [i for i, allocated in enumerate(bundle) if allocated]

    # If no items are allocated, return None
    if not allocated_items:
        return None

    # Find the least valued item among the allocated ones
    least_valued_item_index = min(allocated_items, key=lambda item: valuations[agent_index][item])

    return least_valued_item_index