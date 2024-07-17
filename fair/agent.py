from fair.valuation import *
from abc import ABC, abstractmethod
from typing import *

Item = Any
Bundle = Set[Item]

class Agent(ABC):
    """
    Represents a participant in allocation problems for indivisible items.
    This abstract class outlines the basic structure of an agent.
    """

    def __init__(self, valuation, name=None, duplicity=1):
        """
        Initializes an agent with a valuation function, an optional name, and duplicity.
        
        :param valuation: The valuation function, determining how the agent values items.
        :param name: Optional. A name for the agent for easier identification.
        :param duplicity: Optional. Indicates how many agents share this valuation function.
        """
        self.valuation = valuation
        if name is not None:
            self._name = name
        self.duplicity = duplicity
        
    
    def name(self):
        """
        Returns the name of the agent. If not specified, returns 'Anonymous'.
        """
        if hasattr(self, '_name') and self._name is not None:
            return self._name
        else:
            return "Anonymous"
    
    
    def value(self, bundle):
        """
        Calculates and returns the value of a given bundle of items according to the agent's valuation function.
        """
        return self.valuation.value(bundle)
    
    def total_value(self):
        """
        Returns the total value the agent assigns to all items.
        """
        return self.valuation.total_value()
    
    def all_items(self):
        """
        Returns a list or set of all items considered by the agent's valuation.
        """
        return self.valuation.all_items()
    
    def best_index(self, allocation):
        """
        Returns the index of the most valuable bundle for the agent from a list of bundles.
        
        :param allocation: A list of bundles, where each bundle is a collection of items.
        :return: The index of the most valuable bundle according to the agent's valuation.
        """
        # Iterate through the allocation, evaluating each bundle's value, and return the index of the highest value.
        values = [self.valuation.value(bundle) for bundle in allocation]
        return values.index(max(values))
    
    def value_except_best_c_goods(self, bundle, c=1):
        """
        Calculates the value of a bundle excluding the value of the 'c' most valuable goods in it.
        
        :param bundle: The bundle of items being evaluated.
        :param c: The number of the most valuable goods to exclude from the valuation.
        :return: The adjusted value of the bundle.
        """
        return self.valuation.value_except_best_c_goods(bundle, c)
    
    def value_except_worst_c_goods(self, bundle, c=1):
        """
        Calculates the value of a bundle excluding the value of the 'c' least valuable goods in it.
        
        :param bundle: The bundle of items being evaluated.
        :param c: The number of the least valuable goods to exclude from the valuation.
        :return: The adjusted value of the bundle.
        """
        return self.valuation.value_except_worst_c_goods(bundle, c)
    
    def value_1_of_c_MMS(self, c=1):
        """
        Calculates the value of the 1-out-of-c maximin-share (MMS) for the agent.
        
        The MMS is the highest value that an agent can guarantee to receive,
        if they were to partition the items into c bundles and receive the least preferred bundle.
        
        :param c: The number of partitions to consider for the MMS calculation.
        :return: The value of the 1-out-of-c MMS for the agent.
        """
        return self.valuation.value_1_of_c_MMS(c)
    
    def partition_1_of_c_MMS(self, c, items):
        """
        Attempts to partition the items into c bundles in a way that maximizes the value
        of the least preferred bundle by the agent, approximating the agent's 1-out-of-c MMS.
        
        :param c: The number of bundles to partition the items into.
        :param items: The list of items to be partitioned.
        :return: A list of bundles (partitions) that represents an approximation of the 1-out-of-c MMS partition.
        """
        return self.valuation.partition_1_of_c_MMS(c, items)
    
    def is_EFc(self, own_bundle, all_bundles, c):
        """
        Checks if the agent's allocation is envy-free except for 'c' goods.
        """
        return self.valuation.is_EFc(own_bundle, all_bundles, c)
    
    def is_EF1(self, own_bundle, all_bundles):
        """
        Checks if the agent's allocation is envy-free except for one good (EF1).
        """
        return self.valuation.is_EF1(own_bundle, all_bundles)
    
    def is_EFx(self, own_bundle, all_bundles):
        """
        Checks if the agent's allocation is envy-free except for any good (EFx).
        """
        return self.valuation.is_EFx(own_bundle, all_bundles)
    
    def is_EF(self, own_bundle, all_bundles):
        """
        Checks if the agent's allocation is entirely envy-free (EF).
        """
        return self.valuation.is_EF(own_bundle, all_bundles)

    def is_1_of_c_MMS(self, own_bundle, c, approximation_factor=1):
        """
        Checks if the agent's allocation meets the 1-out-of-c maximin share criterion.
        """
        return self.valuation.is_1_of_c_MMS(own_bundle, c, approximation_factor)
    
    def __repr__(self):
        if self.duplicity==1:
            return f"{self.name()} is an agent with a {self.valuation.__repr__()}"
        else:
            return f"{self.name()} are {self.duplicity} agents with a {self.valuation.__repr__()}"
        
class AdditiveAgent(Agent):
    """
    Represents an agent (or several agents with identical preferences) with an additive valuation function
    for indivisible items. This class specializes the Agent class for scenarios where valuations are additive.
    """
    def __init__(self, map_good_to_value, name=None, duplicity=1):
        """
        Initializes an AdditiveAgent with a given additive valuation function.

        :param map_good_to_value: A dict that maps each item to its value indicating the agent's
                                  valuation of individual items. Can also be a list of values 
                                  corresponding to each item's value.
        :param name: Optional; a name for the agent for identification purposes.
        :param duplicity: Optional; the number of agents represented by this instance that share
                          the same valuation function.
        """
        # Initialize the base Agent class with an AdditiveValuation instance
        super().__init__(AdditiveValuation(map_good_to_value), name=name, duplicity=duplicity)
        
    @staticmethod
    def list_from(input: Any) -> List['AdditiveAgent']:
        """
        Constructs a list of AdditiveAgent instances from various input formats.

        :param input_data: Input data in one of the following formats:
                           - Dict of dicts: Each key-value pair represents an agent and its item-value map.
                           - Dict of lists: Each key-value pair represents an agent and its list of item values.
                           - List of dicts: Each element is a dict representing an agent's item-value map.
                           - List of lists: Each element is a list of item values for an agent.
        :return: A list of AdditiveAgent instances created from the input data.
        """
        if isinstance(input, dict):
            return [
                AdditiveAgent(valuation, name=name)
                for name,valuation in input.items()
            ]
        elif isinstance(input, list):
            if len(input)==0:
                return []
            if isinstance(input[0], Agent):
                return input
            return [
                AdditiveAgent(valuation, name=f"Agent #{index}")
                for index,valuation in enumerate(input)
            ]
        else:
            raise ValueError(f"Input to list_from should be a dict or a list, but it is {type(input)}")
        
        
if __name__ == "__main__":
    import doctest
    (failures,tests) = doctest.testmod(report=True)
    print (f"{failures} failures, {tests} tests")
