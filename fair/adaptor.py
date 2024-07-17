from typing import Callable, Any
from fair import AgentList, Allocation, ValuationMatrix, AllocationMatrix, ListBundle
import numpy as np

def divide(algorithm: Callable, input: Any, *args, **kwargs):
    """
    An adaptor function for fair division of indivisible items among agents.

    :param algorithm: A specific fair division algorithm. Should accept parameters:
        - agents: An AgentList representing the agents participating in the division.
        - valuation_matrix: ValuationMatrix - a matrix V where V[i,j] is the value of agent i to item j.
      It can also accept additional parameters specific to the algorithm.

    :param input_data: The input to the algorithm, which can be in one of the following forms:
       * A dict mapping agent names to their valuations (dict of dicts), e.g., {"Alice": {"x": 1, "y": 2}, "George": {"x": 3, "y": 4}}. Each agent's valuation is represented as a dict mapping item names to their respective values.
       * A list of dicts, where each dict represents an agent's valuation, e.g., [{"x": 1, "y": 2}, {"x": 3, "y": 4}]. Agents are named by their order: "Agent #0", "Agent #1", etc.
       * A list of lists, representing valuations in a matrix form, e.g., [[1, 2], [3, 4]]. Agents and items are indexed by their order.
       * A numpy array, representing valuations in a matrix form, e.g., np.array([[1, 2], [3, 4]]).
       * A list of Valuation objects, e.g., [AdditiveValuation({"x": 1, "y": 2}), AdditiveValuation({"x": 3, "y": 4})], where each Valuation object represents an agent's valuation.
       * A list of Agent objects, where each Agent is initialized with a Valuation, e.g., [AdditiveAgent({"x": 1, "y": 2}), AdditiveAgent({"x": 3, "y": 4})].

    :param kwargs: Any additional arguments that the specific algorithm might require.

    :return: An Allocation object representing the allocation of items among the agents. This object encapsulates the outcome of the division algorithm, detailing which items have been allocated to each agent.

    The function dynamically adapts the input_data to the format expected by the specified algorithm and then invokes the algorithm with the adapted input. The result is converted back into an Allocation object for consistent output handling.
    """
    annotations_list = list(algorithm.__annotations__.items())
    first_argument_type = annotations_list[0][1]
    
    
    ### Convert input to ValuationMatrix
    if first_argument_type==ValuationMatrix:
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
        
        special_algorithms = ["efx_with_bounded_charity", "divsion_with_donating"]
        
        
        if np.any(np.all(valuation_matrix._v == 0, axis = 0)):
            raise ValueError("Valuation matrix should not have all zero columns")
        
        # Step 2. Run the algorithm:
        if algorithm.__name__ in special_algorithms:
            output = algorithm(valuation_matrix, input_data = input, *args, **kwargs)
        else:
            output = algorithm(valuation_matrix, *args, **kwargs)
        
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
        
    else:
        return algorithm(input, *args, **kwargs)
    
if __name__ == "__main__":
    # from fairpy.items.round_robin import round_robin
    # print(divide(algorithm=round_robin, instance = [[11,22,44,0],[22,11,66,33]]))
    import doctest, sys
    (failures, tests) = doctest.testmod(report=True)
    print(f"{failures} failures, {tests} tests")
    # if failures > 0:
    #     sys.exit(1)
