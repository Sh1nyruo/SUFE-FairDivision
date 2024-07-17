import numpy as np
from fair import ValuationMatrix
import math
import networkx as nx
import time


class MarketModel:
    def __init__(self, valuationMatrix: ValuationMatrix):
        """
        Initialize the MarketModel with a given valuation matrix.
        This sets up the number of agents, number of items, the valuation matrix itself,
        and initializes prices and delta. It also computes the initial fractional allocation.
        
        :param valuationMatrix: An instance of ValuationMatrix containing agents' valuations of items.
        """
        self.n = valuationMatrix.num_of_agents
        self.m = valuationMatrix.num_of_objects
        self._v = valuationMatrix._v
        self.price, self.delta = self.initialize()
        # self.price = self.spending_capacity()
        self.constant = self.delta
        self.price_vector = np.array([np.copy(self.price)])
        
        
    def initialize(self):
        """
        Initialize the prices and delta for the market.
        Prices are set to half the maximum bang-per-buck across all items.
        Delta is calculated based on the number of items.
        
        :return: Tuple of (initial prices array, delta value).
        """
        m = self.m
        m_bar = 2 ** math.ceil(math.log2(2*m))
        
        delta = 1 / (2 * m_bar)

        price = 0.5 * np.max(self._v / (np.sum(self._v, axis = 1).reshape(-1,1)), axis = 0)
        
        return price, delta
    
    def calculate_MBB_ratios(self) -> np.array:
        """
        Calculate the Maximum Bang per Buck (MBB) ratios for each agent.
        
        :return: A boolean numpy array where each row corresponds to an agent and
                 each column to an item. True indicates the item is in the MBB set for that agent.
        """
        
        ratios = self._v / self.price
        #ratios = ratios / ratios.sum(axis=1, keepdims = True)
        # print(ratios)
        max_ratios = ratios.max(axis=1)
        MBB_items = (np.abs(ratios - max_ratios[:, np.newaxis]) < (1e-6))
        
        return MBB_items
    
    def spending_capacity(self):
        """
        Calculate the spending capacity for each item based on the prices and delta.
        Spending capacity is adjusted to be a multiple of delta.
        
        :return: A numpy array of the adjusted prices (spending capacity) for each item.
        """
        delta_prices = np.where(self.price % self.delta == 0,
                                np.minimum(1, self.price + self.delta),
                                np.minimum(1, self.delta * np.ceil(self.price / self.delta)))
        return delta_prices
    
    def spending_restricted_outcome(self):
        """
        Compute the maximum flow in the network to find the fractional allocation of items to agents.
        This allocation respects the agents' spending capacities and the items' MBB constraints.
        
        :return: A numpy 2D array where each row represents an agent and each column an item.
                 The values represent the fraction of the item allocated to the agent.
        """
        G = nx.DiGraph()
        
        # Add edges from source to agents with capacity 1 (total budget)
        for agent in range(self.n):
            G.add_edge('s', f'agent_{agent}', capacity=1)
        
        capacity = self.spending_capacity()
        
        # Add edges from items to sink with capacity min(1, price)
        for item in range(self.m):
                G.add_edge(f'item_{item}', 't', capacity = min(1, capacity[item]))
            
        # Calculate the MBB ratios
        MBB_items = self.calculate_MBB_ratios()

        # Add edges between agents and their MBB items with capacity as the price
        for agent in range(self.n):
            for item in range(self.m):
                if MBB_items[agent, item]:
                    G.add_edge(f'agent_{agent}', f'item_{item}', capacity=capacity[item])

        # Compute the maximum flow
        flow_value, flow_dict = nx.maximum_flow(G, 's', 't')

        spendingMatrix = np.zeros((self.n, self.m))
        
        for agent_key, items in flow_dict.items():
            if 'agent' in agent_key:
                agent_idx = int(agent_key.split('_')[1])
                for item_key, flow_value in items.items():
                    if 'item' in item_key:
                        item_idx = int(item_key.split('_')[1])
                        spendingMatrix[agent_idx, item_idx] = flow_value

        return spendingMatrix

    def PriceIncrease(self):
        """
        Repeatedly increase the prices of items reachable by agents with unspent money until the
        conditions for a new MBB edge or an increase in c_j(p, Δ) are met, then reassess all agents.
        """
        # If there is only one item, assign it proportionally based on valuations
        if self.m == 1:
            total_valuation = sum(self._v[:, 0])
            for agent_index in range(self.n):
                proportion = self._v[agent_index, 0] / total_valuation
                self.price[0] = proportion  # Set the price as the proportional valuation
            return self.price
        
        # Initially compute the delta allocation and set a flag to indicate if we need to increase prices
        delta_allocation = self.spending_restricted_outcome()
        
        total_spent_by_agents = delta_allocation.sum(axis=1)
        agents_with_unspent_money = np.where(total_spent_by_agents < 1)[0]

        # Continue the process until there are no agents with unspent money
        while agents_with_unspent_money.size > 0:
            for agent in agents_with_unspent_money:
                # Compute the old spending capacities and MBB items
                old_delta_prices = self.spending_capacity()
                old_MBB_items = self.calculate_MBB_ratios()
                agent_old_MBB_items = np.where(old_MBB_items[agent])[0]
                # Get the reachable set R for the current agent
                reachable_set = np.array(list(self.find_reachable_set(agent)))
                #print("Reachable Set: \n", reachable_set)
                # Increase the prices of items in R
                self.price[reachable_set] *= (1 + self.constant)
                #print("Prices: \n", self.price)
                
                new_delta_prices = self.spending_capacity()
                # Recompute the delta allocation and MBB items after the price change
                
                new_MBB_items = self.calculate_MBB_ratios()
                agent_new_MBB_items = np.where(new_MBB_items[agent])[0]
                #Get the new spending capacities after the price change
                #print("MBB: \n", new_MBB_items)
                # Check if a new MBB edge appears or if c_j(p, Δ) increases for some item j in R
                new_edges = not np.array_equal(new_MBB_items, old_MBB_items)
                price_increase = any(new_delta_prices[reachable_set] > old_delta_prices[reachable_set])

                # If a new MBB edge has appeared or if c_j(p, Δ) increased, we need to reassess all agents
                if new_edges:
                    agent_new_MBB_item = agent_new_MBB_items[0]
                    update_ratio = self.price[agent_new_MBB_item] / self._v[agent, agent_new_MBB_item]
                    self.price[agent_old_MBB_items] = self._v[agent, agent_old_MBB_items] * update_ratio
                    # print("Agent: ",agent)
                    # print("Agent old MBB Items: ", agent_old_MBB_items)
                    # print("Agent new MBB Items: ", agent_new_MBB_items)
                    # print("Update Ratio: ", update_ratio)
                    # print("Prices: ", self.price)
                    break
                elif price_increase:
                    break
                #else:
                    #print("Prices: ", self.price)
                    #return self.price
            self.price_vector = np.vstack((self.price_vector, np.copy(self.price)))

            new_delta_allocation = self.spending_restricted_outcome()
            #print("New Delta Allocation: \n", new_delta_allocation)
            # Recompute the total spending by agents to update the list of agents with unspent money
            total_spent_by_agents = new_delta_allocation.sum(axis=1)

            agents_with_unspent_money = np.where(total_spent_by_agents < 1)[0]

        return self.price
            
    def find_reachable_set(self, agent_index):
        """
        Find the reachable set R for an agent via the MBB graph G(p) and the spending graph Q(x).
        
        :param agent_index: The index of the agent for whom to find the reachable set.
        :return: A set of item indices that are reachable by the agent.
        """
        # Get the MBB items for the agent
        MBB_items = self.calculate_MBB_ratios()[agent_index]
        
        # Get the items where the agent has a positive allocation in the spending matrix
        spending_matrix = self.spending_restricted_outcome()
        directly_allocated_items = np.where(spending_matrix[agent_index] > 0)[0]
        
        # Initialize the reachable set with items from MBB and directly allocated items
        reachable_set = set(directly_allocated_items).union(np.where(MBB_items)[0]) 
        
        # Perform a BFS to find other reachable items through the spending graph
        # Initialize the search with the directly allocated items
        search_queue = list(directly_allocated_items)
        while search_queue:
            current_item = search_queue.pop(0)
            # Find other agents who spent money on this item
            other_agents = np.where(spending_matrix[:, current_item] > 0)[0]
            for other_agent in other_agents:
                # Ensure we don't revisit the original agent
                if other_agent != agent_index:
                    # Find other items that these agents have spent money on
                    other_items = np.where(spending_matrix[other_agent] > 0)[0]
                    # Add these items to the reachable set and search queue if they are new
                    for item in other_items:
                        if item not in reachable_set:
                            reachable_set.add(item)
                            search_queue.append(item)
        
        return reachable_set 
        
    def weakly_polynomial_algorithm(self):
        """
        Implements the weakly polynomial algorithm for fair division.
        
        :return: The final prices for the items.
        """
        # Step 1: Initialize Δ and prices p_j
        m = self.m  # Number of objects
        V_max = self.calculate_V_max()  # Maximum ratio V_max of values for one agent
        
        # Step 2: Initial call to PriceIncrease
        self.PriceIncrease()
        
        # Define r as the number of iterations in O(m log V_max)
        num_iterations = int(np.ceil(m * np.log(V_max)))  # Example of O(m log V_max) iterations
        
        
        # iteration_times = []  # List to store the duration of each iteration
        # Step 3: Iteratively update Δ and prices p
        for _ in range(num_iterations):
            # start_time = time.time()  # Record the start time of the iteration
            self.delta /= 2.0  # Step 4: Halve the value of Δ
            self.PriceIncrease()  # Step 5: Call PriceIncrease with updated Δ and prices
            # end_time = time.time()  # Record the end time of the iteration
            # iteration_times.append(end_time - start_time)  # Calculate and store the duration

            # print(f"Iteration {_ + 1} took {iteration_times[-1]:.4f} seconds. Delta: {self.delta}")

        return self.price


    def calculate_V_max(self):
        """
        Calculate the maximum ratio V_max of values for one agent, defined as the maximum
        over all agents of the ratio of their maximum valuation to their minimum non-zero valuation.

        :return: The maximum ratio V_max.
        """
        # Get the maximum valuation for each agent
        max_valuations = np.max(self._v, axis=1)

        # Replace zero valuations with np.inf to avoid division by zero
        # This replacement is safe as we are looking for the minimum values greater than zero
        min_valuations = np.where(self._v == 0, np.inf, self._v)
        min_valuations = np.min(min_valuations, axis=1)

        # Calculate the ratio of max to min for each agent
        valuation_ratios = max_valuations / min_valuations

        # Take the maximum of these ratios as V_max
        V_max = np.max(valuation_ratios)
        return V_max


        
        
    