from nn import *
import matplotlib.pyplot as plt

random.seed(datetime.now())

# Window size = number of input nodes
window_size = 15

ts = [266, 145.9, 183.1, 119.3, 180.3, 168.5, 231.8,
      224.5, 192.8, 122.9, 336.5, 185.9, 194.3, 149.5,
      210.1, 273.3, 191.4, 287, 226, 303.6, 289.9,
      421.6, 264.5, 342.3, 339.7, 440.4, 315.9, 439.3,
      401.3, 437.4, 575.5, 407.6, 682, 475.3, 581.3, 646.9]

# Normalize values
ts_min = min(ts)
ts_max = max(ts)
ts_scaled = [(x - ts_min) / (ts_max - ts_min) for x in ts]

# Create NN
nn = NeuralNetwork([window_size, window_size, 1],
                   [ActivationFunction.create_identity,
                    ActivationFunction.create_sigmoid,
                    ActivationFunction.create_sigmoid])

# Train data
print("Training started...")
for i in range(100000):
    index = random.randint(0, len(ts_scaled) - window_size - 2)
    input = [ts_scaled[j] for j in range(index, index + window_size)]
    target = [ts_scaled[index + window_size]]
    nn.train(input, target)

print("Training finished...")

errors = []
outputs = []

# Print results
for i in range(0, len(ts_scaled) - window_size):
    input = [ts_scaled[j] for j in range(i, i + window_size)]
    target = ts[i + window_size]
    output = nn.predict(input)[0] * (ts_max - ts_min) + ts_min # scale back
    outputs.append(output)
    errors.append(target - output)

    print("Input range: ", (i, i + window_size), " target: ", target,
          " output: ", output, " error: ", errors[-1])

# Count MSE
avg = float(sum(errors)) / len(errors)
mse = sum([(x - avg)**2 for x in errors]) / len(errors)

print("MSE: ", mse)

# Show graphs
plt.plot(range(0, len(ts)), ts, color='blue')
plt.plot(range(window_size, len(ts)), outputs, color='red')
plt.show()