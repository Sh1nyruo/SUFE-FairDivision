from typing import List, Any, Dict
import numpy as np
from collections import defaultdict
from collections.abc import Iterable
import fair
from fair.valuation import ValuationMatrix
from fair.agentlist import AgentList
from fair.bundle import *

DEFAULT_PRECISION = 3     # number of significant digits in printing
DEFAULT_SEPARATOR = ","   # separator between items in printing

class AllocationMatrix:
    """
    A matrix where each row represents an agent, each column represents an object,
    and a value of 1 indicates the object is allocated to the agent, while 0 indicates it is not.
    """
    
    def __init__(self, allocation_matrix:np.ndarray):
        if isinstance(allocation_matrix,list):
            allocation_matrix = np.array(allocation_matrix)
        elif isinstance(allocation_matrix,AllocationMatrix):
            allocation_matrix = allocation_matrix._z
        self._z = allocation_matrix
        self.num_of_agents = len(allocation_matrix)
        self.num_of_objects = len(allocation_matrix[0])
        
    def agents(self):
        return range(self.num_of_agents)

    def objects(self):
        return range(self.num_of_objects)
    
    def utility_profile(self, v: ValuationMatrix) -> np.array:
        """
        Computes the utility profile of the agents given a valuation matrix.
        """
        return np.array([
            self[i] @ v[i]
            for i in v.agents()
        ])

    def __getitem__(self, key):
        if isinstance(key,tuple):
            return self._z[key[0]][key[1]]  # 'key' (agent index, item index); return this agent's valuation for that item.
        else:
            return self._z[key]  # 'key' is the index of an agent; return this agent's valuation.

    def __repr__(self):
        return np.array2string (self._z, max_line_width=100)	
    
class Allocation:
    def __init__(self, agents: List[Any], bundles: List[List[Any]], matrix=None):
        """
        Initializes an allocation for indivisible items.

        :param agents: A list of agent names or a dict mapping agent names to valuations.
        :param bundles: A list of bundles (each a list of item names) or a dict mapping agent names to bundles.
        """
        self.valuation_matrix = agents
        
        # Ensuring the structure of agents matches with bundles
        if isinstance(agents,dict) and not isinstance(bundles,dict):
            raise ValueError(f"Cannot match agents to bundles: agents is a dict but bundles is {type(bundles)}")

        if isinstance(agents,dict) or (isinstance(agents,list) and isinstance(agents[0],list)):       # If "agents" is a dict mapping an agent name to its valuation...
            agents = AgentList(agents)  # ... convert it to a list mapping an agent index to its valuation.
            map_agent_index_to_name = agents.agent_names()
        else:
            map_agent_index_to_name = fair.agent_names_from(agents)
            
        self.map_agent_index_to_name = map_agent_index_to_name
        
        if isinstance(bundles,dict):       # If "bundles" is a dict mapping an agent name to its bundle... 
            bundles = [bundles.get(name,None) for name in map_agent_index_to_name]  # ... convert it to a list mapping an agent index to its bundle.

        if isinstance(bundles, np.ndarray):
            bundles = AllocationMatrix(bundles)
            
        if isinstance(agents, ValuationMatrix):
            agent_valuations = agents
        
        if isinstance(bundles, AllocationMatrix):
            self.matrix = bundles
            bundles = self._convert_allocation_matrix_to_bundles(bundles, agent_valuations)
        else:
            bundles = [bundle_from(bundles[i]) for i in range(len(bundles))]
            
        if matrix is not None:
            self.matrix = AllocationMatrix(matrix)
        
        # Compute num_of_agents:
        num_of_agents = agents.num_of_agents if hasattr(agents,'num_of_agents') else len(agents)
        num_of_bundles = len(bundles)
        
        if num_of_agents!=num_of_bundles:
            raise ValueError(f"Numbers of agents and bundles must be identical, but they are not: {num_of_agents}, {num_of_bundles}")

        # Verify that all bundles are iterable:
        for i_bundle,bundle in enumerate(bundles):
            if bundle is not None and not isinstance(bundle, Iterable):
                raise ValueError(f"Bundle {i_bundle} should be iterable, but it is {type(bundle)}")

        # Initialize bundles and agents:
        self.num_of_agents = num_of_agents
        self.num_of_bundles = num_of_bundles
        self.agents = agents
        self.bundles = bundles
        self.agent_bundle_value_matrix = compute_agent_bundle_value_matrix(agents, bundles, num_of_agents, num_of_bundles)

    def _convert_allocation_matrix_to_bundles(self, allocation_matrix, agent_valuations):
        # Convert AllocationMatrix into a list of sets, each representing allocated items for an agent
        new_bundles = []
        for agent_index in range(allocation_matrix.num_of_agents):
            allocated_items = {item_index for item_index in range(allocation_matrix.num_of_objects) if allocation_matrix[agent_index, item_index] > 0}
            new_bundles.append(allocated_items)
        return new_bundles
    
    def get_bundles(self):
        return self.bundles
    
    def map_agent_to_bundle(self):
        """
        Return a mapping from each agent's name to the bundle he/she received.
        """
        return {
            self.map_agent_index_to_name[i_agent]: self.bundles[i_agent].items
            for i_agent in range(self.num_of_agents)
        }
    
    
    def map_item_to_agents(self, sortkey=None)->Dict[str,Any]:
        """
        Return a mapping from each item to the agent/s who receive this item (may be more than one if there are multiple copies)
        """
        result = defaultdict(list)
        for i_agent, bundle in enumerate(self.bundles):
            if bundle is None:
                continue
            for item in bundle:
                result[item].append(self.map_agent_index_to_name[i_agent])
        if sortkey is not None:
            for item,winners in result.items():
                winners.sort(key=sortkey)
        return dict(result)
    
    def __getitem__(self, agent_index:int):
        return self.bundles[agent_index]

    def __iter__(self):
       return self.bundles.__iter__() 


    def utility_profile(self)->list:
        """
        Returns a vector that maps each agent index to its utility (=sum of values) under this allocation.
        """
        return np.array([self.agent_bundle_value_matrix[i_agent,i_agent] for i_agent in range(self.num_of_agents)])

    def utility_profile_matrix(self)->list:
        """
        Returns a vector that maps each agent index to its utility (=sum of values) under this allocation.
        """
        return self.agent_bundle_value_matrix
    
    def str_with_values(self, precision=None)->str:
        """
        Returns a representation of the current allocation, showing the value of each agent to his own bundle.
        """
        if precision is None: precision=DEFAULT_PRECISION
        result = ""
        for i_agent, agent_name in enumerate(self.map_agent_index_to_name):
            agent_bundle = self.bundles[i_agent]
            agent_value = self.agent_bundle_value_matrix[i_agent,i_agent]
            result += f"{agent_name} gets {agent_bundle} with value {agent_value:.{precision}g}.\n"
        result += f"Nash Social Welfare: {self.nash_welfare():.{precision}g}\n"
        return result
    
    def str_with_value_matrix(self, separator=None, precision=None)->str:
        """
        Returns a representation of the current allocation, showing the value of each agent to *all* bundles.
        """
        if separator is None: separator=DEFAULT_SEPARATOR
        if precision is None: precision=DEFAULT_PRECISION
        result = ""
        for i_agent, agent_name in enumerate(self.map_agent_index_to_name):
            agent_bundle = self.bundles[i_agent]
            values_str = ""
            for i_bundle, bundle in enumerate(self.bundles):
                agent_value_to_bundle = self.agent_bundle_value_matrix[i_agent,i_bundle]
                agent_value_to_bundle_str = f"{agent_value_to_bundle:.{precision}g}"
                if i_bundle==i_agent:
                    agent_value_to_bundle_str = "["+agent_value_to_bundle_str+"]"
                values_str += " " + agent_value_to_bundle_str
            result += f"{agent_name} gets {agent_bundle}. Values:{values_str}.\n"
        return result
    
    def nash_welfare(self):
        return np.prod(self.utility_profile())**(1/self.num_of_agents)
    
    def __repr__(self)->str:
        return self.str_with_values(precision=DEFAULT_PRECISION)
    
def compute_agent_bundle_value_matrix(agents, bundles, num_of_agents, num_of_bundles):
    """
    Compute a matrix U in which each row is an agent, each column is a bundle,
    and U[i,j] is the value of agent i to bundle j.
    """
    agent_bundle_value_matrix = np.zeros([num_of_agents,num_of_bundles])
    # print("bundles: ",bundles)
    if hasattr(agents, 'agent_value_for_bundle'):  # E.g. when agents is a ValuationMatrix.
        # print("agents 1: ",agents)
        for i_agent in range(num_of_agents):
            for i_bundle in range(num_of_bundles):
                agent_bundle_value_matrix[i_agent,i_bundle] = agents.agent_value_for_bundle(i_agent, bundles[i_bundle])
    elif hasattr(next(iter(agents)), 'value'):              # E.g. when agents is a list of Agent objects
        # print("agents 2: ",agents)
        for i_agent in range(num_of_agents):
            for i_bundle in range(num_of_bundles):
                agent = agents[i_agent]
                bundle = bundles[i_bundle]
                try:
                    value = agent.value(bundle)
                except TypeError as err:
                    raise TypeError(f"Cannot compute the value of agent {type(agent)} for bundle {bundle} of type {type(bundle)}") from err
                agent_bundle_value_matrix[i_agent,i_bundle] = value
                    
    else:
        # WARNING: Cannot compute agents' values at all
        for i_agent in range(num_of_agents):
            for i_bundle in range(num_of_bundles):
                agent_bundle_value_matrix[i_agent,i_bundle] = np.nan
    return agent_bundle_value_matrix


class AllocationWithDonation(Allocation):
    def __init__(self, agents: List[Any], bundles: List[List[Any]], matrix=None, donations=None):
        super().__init__(agents, bundles, matrix)



if __name__ == "__main__":
    import doctest
    (failures, tests) = doctest.testmod(report=True)
    print("{} failures, {} tests".format(failures, tests))