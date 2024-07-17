import numpy as np
from fair import ValuationMatrix

def round_robin(v: ValuationMatrix) -> np.array:
    num_agents = v.num_of_agents
    num_items = v.num_of_objects
    
    # Initialize the allocation as a zero matrix
    allocation = np.zeros((num_agents, num_items), dtype=int)
    
    # Keep track of unassigned items using a set
    unassigned_items = set(range(num_items))
    
    while unassigned_items:
        # Iterate over each agent
        for agent in range(num_agents):
            # If there are no unassigned items left, break the loop
            if not unassigned_items:
                break
            
            # Get the agent's valuations for unassigned items and find the highest valued one
            agent_valuations = {item: v[agent][item] for item in unassigned_items}
            chosen_item = max(agent_valuations, key=agent_valuations.get)
            
            # Assign the chosen item to the agent and update the allocation and unassigned items
            allocation[agent][chosen_item] = 1
            unassigned_items.remove(chosen_item)
    
    return allocation



### MAIN

if __name__ == "__main__":
    # logger.addHandler(logging.StreamHandler())
    # logger.setLevel(logging.INFO)
    #
    import doctest
    (failures,tests) = doctest.testmod(report=True)
    print ("{} failures, {} tests".format(failures,tests))
