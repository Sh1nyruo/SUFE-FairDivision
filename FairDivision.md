# Fair division of indivisible goods: Algorithms for finding EFX allocation with donations

```python
import fair
divide = fair.divide
```

`fair` includes several algorithms for fair division with indivisible items. Here's the example inputs:

```python
input_4_agents_small_instance_1 = {
    "Agent1": {"a": 1, "b": 0, "c": 0, "d": 0, "e": 0},
    "Agent2": {"a": 15, "b": 2, "c": 0, "d": 0, "e": 0},
    "Agent3": {"a": 15, "b": 0, "c": 1, "d": 1, "e": 1},
    "Agent4": {"a": 3, "b": 2, "c": 1, "d": 1, "e": 1},
}
input_3_agents =   {'Agent1': {'a': 10, 'b': 9, 'c': 4, 'd': 6},
  'Agent2': {'a': 10, 'b': 6, 'c': 9, 'd': 4},
  'Agent3': {'a': 10, 'b': 4, 'c': 6, 'd': 9}}
```

## Maximizing Nash Social Welfare Algorithms

Brute Force

```python
divide(fair.nashwelfare.max_nash_welfare_brute_force, input_4_agents_small_instance_1)
```

```
Agent1 gets {a} with value 1.
Agent2 gets {b} with value 2.
Agent3 gets {c,d} with value 2.
Agent4 gets {e} with value 1.
Nash Social Welfare: 1.41
```

Round Robin:

```python
divide(fair.nashwelfare.round_robin, input_4_agents_small_instance_1)
```

```
Agent1 gets {a,e} with value 1.
Agent2 gets {b} with value 2.
Agent3 gets {c} with value 1.
Agent4 gets {d} with value 1.
Nash Social Welfare: 1.19
```

Approximating Algorithm:

```python
divide(fair.nashwelfare.rv_approximating_nash_welfare, input_4_agents_small_instance_1)
```

```
Agent1 gets {a} with value 1.
Agent2 gets {b} with value 2.
Agent3 gets {e} with value 1.
Agent4 gets {c,d} with value 2.
Nash Social Welfare: 1.41
```

## Algorithm for finding EFX allocation with donations

Optimal:

```python
divide(fair.nashwelfare.divsion_with_donating, input_4_agents_small_instance_1, optimal = True)
```

```
Agent1 gets {a} with value 1.
Agent2 gets {b} with value 2.
Agent3 gets {c,d} with value 2.
Agent4 gets {e} with value 1.
Nash Social Welfare: 1.41
```

Rounb Robin as input allocation:

```python
divide(fair.nashwelfare.divsion_with_donating, input_4_agents_small_instance_1, NSW_algorithm = fair.nashwelfare.round_robin)
```

```
Agent1 gets {a} with value 1.
Agent2 gets {b} with value 2.
Agent3 gets {c} with value 1.
Agent4 gets {d} with value 1.
Nash Social Welfare: 1.19
```

Approximating Algorithm as input allocation:

```python
divide(fair.nashwelfare.divsion_with_donating, input_4_agents_small_instance_1, NSW_algorithm = fair.nashwelfare.rv_approximating_nash_welfare)
```

```
Agent1 gets {a} with value 1.
Agent2 gets {b} with value 2.
Agent3 gets {e} with value 1.
Agent4 gets {c,d} with value 2.
Nash Social Welfare: 1.41
```
