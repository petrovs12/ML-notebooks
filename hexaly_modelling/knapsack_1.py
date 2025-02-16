import hexaly.optimizer
import numpy as np
# Define the knapsack problem
def generate_knapsack_data(num_items, max_weight, max_value):
    weights = np.random.randint(1, max_weight, size=num_items)
    values = np.random.randint(1, max_value, size=num_items)
    capacity = int(0.5 * np.sum(weights))  # Set capacity to half of the total weight
    return weights, values, capacity

# Example usage
num_items = 1000000
max_weight = 100
max_value = 100

weights, values, capacity = generate_knapsack_data(num_items, max_weight, max_value)

print("Weights:", weights)
print("Values:", values)
print("Capacity:", capacity)

class CallbackExample:

    def __init__(self):
        self.iteration = 0

    def my_callback(self, optimizer, cb_type):
        print('Callback called')
        print(f"current objective value {optimizer.model.objectives[0].value}")


with hexaly.optimizer.HexalyOptimizer() as optimizer:
    model = optimizer.model
    n = len(weights)
    # Define the decision variables x - to take or not each item
    x = [model.bool() for _ in range(n)]
    knapsack_weight = model.sum(x[i] * weights[i] for i in range(n)) 
    model.constraint(knapsack_weight <= capacity)
    knapsack_value = model.sum(x[i] * values[i] for i in range(n))
    model.maximize(knapsack_value)
    model.close()
    cb = CallbackExample()
    optimizer.param.time_limit = 10
    optimizer.param.time_between_displays = 1
    # optimizer.param.timeBetweenDisplays = 0.001
    optimizer.param.set_iteration_between_ticks(1)
    optimizer.add_callback(hexaly.optimizer.HxCallbackType.DISPLAY, 
                            cb.my_callback)
    optimizer.solve()
    print(f"Objective value: {knapsack_value.value}")
    # print(f"Selecte
    # d items: {[i for i in range(n) if x[i].value == 1]}")



