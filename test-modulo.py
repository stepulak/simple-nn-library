from nn import *

random.seed(datetime.now())

"""Convert an integer to binary value represented by list of [0, 1]"""


def convert_number_to_bin_list(number, min_bin_positions=5):
    result = []
    while number:
        result.append(number % 2)
        number //= 2

    for _ in range(len(result), min_bin_positions):
        result.append(0)

    result.reverse()
    return result


"""Create dataset with ascending binary values represented by list of [0, 1]"""


def create_dataset(bin_positions=10):
    def modulo(n): return n % 3 == 0 and n % 4 == 0
    dataset = [(convert_number_to_bin_list(x, bin_positions), [modulo(x)])
               for x in range(0, 2**bin_positions)]
    return dataset


num_input_nodes = 6 # 2^6 maximum number
dataset = create_dataset(num_input_nodes)
layers = [num_input_nodes, num_input_nodes, 1]
ac_funcs = [ActivationFunction.create_identity,
            ActivationFunction.create_sigmoid, ActivationFunction.create_sigmoid]

nn = NeuralNetwork(layers, ac_funcs)

print("Training started...")

for i in range(200000):
    data = dataset[random.randint(0, len(dataset) - 1)]
    nn.train(data[0], data[1])

print("Training finished.")

for x in dataset:
    print("Input: ", x[0], " target: ", x[1], " output: ", nn.predict(x[0]))