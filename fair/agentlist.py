from typing import Any, List
from fair.valuation import *
from fair.agent import Agent, AdditiveAgent

class AgentList:
    def __init__(self, input_data: Any):
        """
        Initializes an AgentList from various input formats, specifically tailored for
        additive valuations and additive agents.
        """
        if isinstance(input_data, AgentList):
            self.agents = input_data.agents
        else:
            self.agents = _agents_from(input_data)
            
    def all_items(self):
        return self.agents[0].all_items()

    def agent_names(self):
        return [agent.name() for agent in self.agents]

    def __getitem__(self, i:int):
        return self.agents[i]

    def remove(self, index):
        self.agents.remove(index)

    def __repr__(self):
        return repr(self.agents)

    def __str__(self):
        return str(self.agents)

    def __len__(self):
        return len(self.agents)
    
def _agents_from(input: Any) -> List[Agent]:
    """
    Constructs a list of agents from various input formats.
    """
    if isinstance(input,np.ndarray):
        input = ValuationMatrix(input)
    if isinstance(input,ValuationMatrix):
        return [
            Agent(AdditiveValuation(input[index]), name=f"Agent #{index}")
            for index in input.agents()
        ]
    input_0 = _representative_item(input)
    if input_0 is None:
        return []
    elif isinstance(input_0, Agent):  # The input is already a list of Agent objects - nothing more to do.
        return input
    elif hasattr(input_0, "value"):   # The input is a list of Valuation objects - we just need to add names.
        return [
            Agent(valuation, name=f"Agent #{index}")
            for index,valuation in enumerate(input)
        ]
    else:
        return AdditiveAgent.list_from(input)


def agent_names_from(input:Any)->List[str]:
    """
    Attempts to extract a list of agent names from various input formats.
    The returned value is a list of strings.
    """
    if hasattr(input, "keys"):
        return sorted(input.keys())
    elif hasattr(input, 'num_of_agents'):
        num_of_agents = input.num_of_agents
        return [f"Agent #{i}" for i in range(num_of_agents)]
    elif isinstance(input, AgentList):
        return agent_names_from(input.agents)

    if len(input)==0:
        return []

    input_0 = next(iter(input))
    if hasattr(input_0, "name"):  
        return [agent.name() for agent in input]
    elif isinstance(input_0, int):
        return [f"Agent #{index}" for index in input]
    elif isinstance(input_0, str):
        return list(input)  # convert to a list; keep the original order
    else:
        return [f"Agent #{i}" for i in range(len(input))]
    
######## UTILITY FUNCTIONS #######


def _representative_item(input:Any):
    if isinstance(input, list):
        if len(input)==0:
            return None
        else:
            return input[0]
    elif isinstance(input, dict):
        if len(input)==0:
            return None
        else:
            return next(iter(input.values()))
    else:
        raise ValueError(f"input should be a list or a dict, but it is {type(input)}")



if __name__ == "__main__":
    import doctest
    (failures,tests) = doctest.testmod(report=True)
    print (f"{failures} failures, {tests} tests")
