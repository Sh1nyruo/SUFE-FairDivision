from abc import ABC, abstractmethod
from typing import *
import numpy as np
import math,itertools
import prtpy
from numbers import Number
from more_itertools import set_partitions

Item = Any
Bundle = Set[Item]

class Valuation(ABC):
    """
    An abstract class that describes a valuation function for indivisible items.
    It can evaluate a set of items, providing a framework for various valuation implementations.
    """
    
    def __init__(self, desired_items: Bundle):
        """
        Initializes the Valuation with a set of desired items.
        
        :param desired_items: The set of all goods that are desired by this agent/s.
        """
        self.desired_items_list = sorted(desired_items)  # Sorted list of items for ordered processing
        self.desired_items = set(desired_items)  # Set of desired items for quick lookup
        self.total_value_cache = self.value(self.desired_items)  # Cache the total value of all desired items


    @abstractmethod
    def value(self, bundle: Bundle) -> float:
        """
        Calculate the value for this agent of the given set of items.

        :param bundle: A set of items to be evaluated.
        :return: The value of the given set of items.
        """
        pass
    
    
    def total_value(self) -> float:
        """
        Return the total value for this agent of all items together.
        
        :return: The total value of all desired items.
        """
        return self.total_value_cache
    
    
    def all_items(self) -> Set[str]:
        """
        Return the set of all items handled by this valuation.
        
        :return: A set of all desired items.
        """
        return self.desired_items
    
    
    def best_index(self, allocation: List[Bundle]) -> int:
        """
        Returns the index of the most valuable bundle for the agent within an allocation.

        :param allocation: A list of bundles (sets of items).
        :return: The index of the most valuable bundle for the agent.
        """
        return max(range(len(allocation)), key=lambda i: self.value(allocation[i]))
    
    
    def value_except_best_c_goods(self, bundle: Set[str], c: int = 1) -> float:
        """
        Calculate the value of the bundle with the best 'c' items removed.
        
        This is a subroutine in checking whether an allocation is EF1.
        """
        if len(bundle) <= c:
            return 0
        else:
            return min(
                self.value(bundle.difference(sub_bundle))
                for sub_bundle in itertools.combinations(bundle, c)
            )
            
            
    def value_except_worst_c_goods(self, bundle: Set[str], c: int = 1) -> float:
        """
        Calculate the value of the bundle with the worst 'c' items removed.
        
        This is a subroutine in checking whether an allocation is EFx.
        """
        if len(bundle) <= c:
            return 0
        else:
            return max(
                self.value(bundle.difference(sub_bundle))
                for sub_bundle in itertools.combinations(bundle, c)
            )
            
    def value_1_of_c_MMS(self, c:int=1)->int:
        """
        Calculates the value of the 1-out-of-c maximin-share
        """
        if c > len(self.desired_items):
            return 0
        else:
            return max(
                min([self.value(bundle) for bundle in partition])
                for partition in set_partitions(self.desired_items_list, c)
            )
            
            
    def is_EF(self, own_bundle: Set[str], all_bundles: List[Set[str]]) -> bool:
        """
        Checks if the allocation is entirely envy-free.
        """
        own_value = self.value(own_bundle)
        for other_bundle in all_bundles:
            if own_value < self.value(other_bundle):
                return False
        return True
    
    
    def is_EFx(self, own_bundle: Set[str], all_bundles: List[Set[str]]) -> bool:
        """
        Checks if the allocation is envy-free except for any good.
        This implementation assumes checking EFx by considering the removal of the worst valued good for the envying agent.
        """
        own_value = self.value(own_bundle)
        for other_bundle in all_bundles:
            if own_value < self.value_except_worst_c_goods(other_bundle, 1):
                return False
        return True
    
    
    def is_EF1(self, own_bundle: Set[str], all_bundles: List[Set[str]]) -> bool:
        """
        Checks if the allocation is envy-free except for 1 good.
        """
        return self.is_EFc(own_bundle, all_bundles, 1)
    
    
    def is_EFc(self, own_bundle: Set[str], all_bundles: List[Set[str]], c: int) -> bool:
        """
        Checks if the allocation is envy-free except for c goods.
        """
        own_value = self.value(own_bundle)
        for other_bundle in all_bundles:
            if own_value < self.value_except_best_c_goods(other_bundle, c):
                return False
        return True
    
    

class AdditiveValuation(Valuation):
    """
    Represents an additive valuation function, assigning values to individual items
    and calculating the value of bundles accordingly.
    """
    
    def __init__(self, map_good_to_value):
        """
        Initializes an agent with a given additive valuation function.
        :param map_good_to_value: a dict that maps each single good to its value,
                                or a list that lists the values of individual items.
        """
        if isinstance(map_good_to_value, AdditiveValuation):
            # If the input is already an instance of AdditiveValuation,
            # extract its map_good_to_value, desired_items, and all_items directly.
            map_good_to_value = map_good_to_value.map_good_to_value
            desired_items = map_good_to_value.desired_items
            all_items = map_good_to_value._all_items
        elif isinstance(map_good_to_value, dict):
            # If the input is a dictionary mapping goods to their values,
            # the keys represent all items, and desired items are those with positive values.
            all_items = map_good_to_value.keys()
            desired_items = set(g for g in all_items if map_good_to_value[g] > 0)
        elif isinstance(map_good_to_value, list) or isinstance(map_good_to_value, np.ndarray):
            # If the input is a list or numpy array, it represents the values of items indexed by their position.
            # All items are represented by their indices, and desired items are those with positive values.
            all_items = set(range(len(map_good_to_value)))
            desired_items = set(g for g in all_items if map_good_to_value[g] > 0)
        else:
            raise ValueError(f"Input to AdditiveValuation should be a dict or a list, but it is {type(map_good_to_value)}")

        self.map_good_to_value = map_good_to_value
        self._all_items = all_items
        super().__init__(desired_items)
        
    def value(self, bundle: Union[Set[str], str, int]) -> int:
        """
        Calculates the agent's value for the given good or set of goods.
        """
        if bundle is None:
            return 0
        elif isinstance(bundle, (str, int)):  # Direct lookup
            if bundle in self.map_good_to_value:
                return self.map_good_to_value[bundle]
            else:
                return sum([self.map_good_to_value[g] for g in bundle])
        elif isinstance(bundle, Iterable):  # Iterable of items
            return sum([self.map_good_to_value[g] for g in bundle])
        else:
            raise TypeError(f"Unsupported bundle type: {type(bundle)}")

    def all_items(self):
        return self._all_items
    
    def value_except_best_c_goods(self, bundle: Bundle, c: int = 1) -> int:
        """
        Calculates the value of the given bundle when the "best" (at most) c goods are removed from it.
        """
        if len(bundle) <= c: return 0
        sorted_bundle = sorted(bundle, key=lambda g: -self.map_good_to_value[g]) # sort the goods from best to worst
        return self.value(sorted_bundle[c:])  # remove the best c goods
    
    def value_except_worst_c_goods(self, bundle:Bundle, c:int=1)->int:
        """
        Calculates the value of the given bundle when the "worst" c goods are removed from it.
        Formally, it calculates the maximum value after removing up to c least valuable goods.
        """
        if len(bundle) <= c: return 0
        sorted_bundle = sorted(bundle, key=lambda g: self.map_good_to_value[g])  # sort the goods from worst to best:
        return self.value(sorted_bundle[c:])  # remove the worst c goods
    
    def value_of_cth_best_good(self, c:int)->int:
        """
        Return the value of the agent's c-th most valuable good.
        """
        if c > len(self.desired_items):
            return 0
        else:
            sorted_values = sorted(self.map_good_to_value.values(), reverse=True)
            return sorted_values[c-1]
        
    def partition_1_of_c_MMS(self, c: int, items: list) -> List[Set[str]]:
        """
        Compute a 1-out-of-c MMS partition of the given items.
        """
        partition = prtpy.partition(
            algorithm=prtpy.partitioning.complete_greedy,
            numbins=c,
            items=items,
            valueof=lambda item: self.value(item),
            objective=prtpy.obj.MaximizeSmallestSum,
            outputtype=prtpy.out.Partition
        )
        return [set(x) for x in partition]
    
    def __repr__(self):
        # Initialize an empty string to build the representation
        values_as_string = ""
        
        # Handle dictionary input: each item and its value
        if isinstance(self.map_good_to_value, dict):
            values_as_string = " ".join([f"{k}={v}" for k, v in sorted(self.map_good_to_value.items())])
        
        # Handle list or numpy array input: values indexed by position
        elif isinstance(self.map_good_to_value, (list, np.ndarray)):
            values_as_string = " ".join([f"v{i}={value}" for i, value in enumerate(self.map_good_to_value)])
        
        # Construct and return the representation string
        return f"Additive valuation: {values_as_string}."
    
    
class ValuationMatrix:
    """
    A matrix representing the valuations of agents over items.
    Each row represents an agent, each column represents an item,
    and each cell's value indicates the value assigned by an agent to an item.
    """

    def __init__(self, valuation_matrix):
        """
        Initializes the valuation matrix.
        :param valuation_matrix: Can be a list of lists, a numpy array, or another ValuationMatrix instance.
        """
        if isinstance(valuation_matrix, list):
            valuation_matrix = np.array(valuation_matrix)
        elif isinstance(valuation_matrix, np.matrix):
            valuation_matrix = np.asarray(valuation_matrix)
        # Copy the internal numpy array if another ValuationMatrix is given
        elif isinstance(valuation_matrix, ValuationMatrix):
            valuation_matrix = valuation_matrix._v

        self._v = valuation_matrix
        self.num_of_agents = len(valuation_matrix)
        self.num_of_objects = 0 if self.num_of_agents == 0 else len(valuation_matrix[0])
    
    def agents(self):
        """
        Returns a range object representing the indices of agents in the valuation matrix.
        """
        return range(self.num_of_agents)
    
    def objects(self):
        """
        Returns a range object representing the indices of objects in the valuation matrix.
        """
        return range(self.num_of_objects)
    
    def __getitem__(self, key):
        """
        Allows indexing into the valuation matrix to retrieve values.
        - If key is a tuple, returns the value for a specific agent-object pair.
        - If key is an integer, returns the values for all objects for a specific agent.
        """
        if isinstance(key, tuple):
            return self._v[key[0]][key[1]]  # Agent's value for a single object
        else:
            return self._v[key]             # Agent's values for all objects
        
    def agent_value_for_bundle(self, agent: int, bundle):
        """
        Calculates the total value of a specified bundle of objects for a given agent,
        specifically for scenarios dealing with indivisible items.
        
        - If bundle is None or empty, returns 0.
        - Returns the sum of values for the objects in the bundle for the specified agent.
        """
        if bundle is None:
            return 0
        # Sum the values of specified objects in the bundle for the given agent.
        # Assumes 'bundle' is an iterable of object indices.
        return sum(self._v[agent][object] for object in bundle)
    
    def without_agent(self, agent: int) -> 'ValuationMatrix':
        """
        Returns a copy of this valuation matrix with the specified agent removed.
        """
        if isinstance(agent, int):
            # np.delete removes the row corresponding to the agent from the matrix.
            new_matrix = np.delete(self._v, agent, axis=0)
            return ValuationMatrix(new_matrix)
        else:
            raise IndexError("Agent index should be an integer.")
        
    def without_object(self, object: int) -> 'ValuationMatrix':
        """
        Returns a copy of this valuation matrix with the specified object removed.
        """
        if isinstance(object, int):
            # np.delete removes the column corresponding to the object from the matrix.
            new_matrix = np.delete(self._v, object, axis=1)
            return ValuationMatrix(new_matrix)
        else:
            raise IndexError("Object index should be an integer.")
        
    def submatrix(self, agents: List[int], objects: List[int]):
        """
        Returns a submatrix of this valuation matrix, containing only specified agents and objects.
        """
        return ValuationMatrix(self._v[np.ix_(agents, objects)])
    
    def verify_ordered(self) -> bool:
        """
        Verifies that the instance is ordered --- all valuations are ordered by descending value.
        Raises a ValueError if any agent's valuations are not in descending order.
        """
        for i in self.agents():
            v_i = self._v[i]
            # Check if any valuation is greater than the next one, indicating they are not in descending order.
            if any(v_i[j] < v_i[j+1] for j in range(self.num_of_objects-1)):
                raise ValueError(f"Valuations of agent {i} are not ordered: {v_i}")
        return True  # Return True if all agents' valuations are correctly ordered.

    def total_values(self) -> np.ndarray:
        """
        Returns a 1-dimensional array in which element i is the total value of agent i for all items.
        """
        return np.sum(self._v, axis=1, keepdims=False)
    
    def normalize(self):
        """
        Normalizes the valuation matrix so that each agent's total valuation equals a common value.
        For integer valuations, scales values to maintain integer data type using the least common multiple (LCM).
        For floating-point valuations, scales values to a common total value, typically 1.

        Returns:
            self (ValuationMatrix): For method chaining, the instance itself is returned after normalization.
        """
        total_values = self.total_values()
        if np.issubdtype(self._v.dtype, np.integer):
            lcm_total_value = np.lcm.reduce(total_values)
            for i in range(self.num_of_agents):
                scale_factor = lcm_total_value // total_values[i]
                self._v[i] = self._v[i] * scale_factor
        else:
            common_value = 1  # Or any other value you see fit for normalization
            for i in range(self.num_of_agents):
                self._v[i] = (self._v[i] / total_values[i]) * common_value
        return self  # Optionally return self for chaining
    
    def verify_normalized(self):
        """
        Verifies that the valuation matrix is normalized, meaning each agent's total valuation is the same.

        Returns:
            bool: True if the matrix is normalized, indicating all agents have the same total valuation.

        Raises:
            ValueError: If the matrix is not normalized, indicating a discrepancy in total valuations among agents.
        """
        total_values = self.total_values()
        # np.allclose allows for floating point comparisons, considering potential rounding differences
        if not np.allclose(total_values, total_values[0]):
            raise ValueError(f"Valuation matrix is not normalized. Total values: {total_values}")
        return True  # Indicate successful verification
    
    def equals(self, other)->bool:
        return np.array_equal(self._v, other._v)

    def __repr__(self):
        return np.array2string (self._v, max_line_width=100)		



if __name__ == "__main__":
    import doctest
    (failures,tests) = doctest.testmod(report=True)
    print ("{} failures, {} tests".format(failures,tests))