from nn import *

random.seed(datetime.now())

nn = NeuralNetwork([2, 2, 1], [ActivationFunction.create_identity,
                               ActivationFunction.create_sigmoid,
                               ActivationFunction.create_sigmoid])

dataset = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

print("Training started...")

for i in range(100000):
    data = dataset[random.randint(0, len(dataset) - 1)]
    nn.train(data[0], data[1])

print("Training finished.")

for x in dataset:
    print("Input: ", x[0], " target: ", x[1], " output: ", nn.predict(x[0]))
