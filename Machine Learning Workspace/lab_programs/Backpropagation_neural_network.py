import numpy as np
# i think this is AND gate implementation.
X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
print(X)
Y = np.array(([0], [0], [0], [1]), dtype=float)
print(Y)
#X = X/np.amax(X, axis=0)
#Y = Y/100


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


# variable initialization
iterations = 5000
increment = 0.1
inputNeurons = 2
hiddenNeurons = 3
outputNeurons = 1 # output layer
weight = np.random.uniform(size=(inputNeurons, hiddenNeurons))
# random.uniform(low, high, size) can generate between low-high and size = an array of n random numbers
bias = np.random.uniform(size=(1, hiddenNeurons))  # size means size of the square matrix.
weightOut = np.random.uniform(size=(hiddenNeurons, outputNeurons))
biasOutput = np.random.uniform(size=(1, outputNeurons))
# next we draw random numbers uniformly of dim X * Y


for i in range(iterations):
    hinput0 = np.dot(X, weight)
    hinput = hinput0 + bias
    hActivation = sigmoid(hinput)

    output0 = np.dot(hActivation, weightOut)
    outInput = output0 + biasOutput
    output = sigmoid(outInput)
    # for Backpropagation
    EO = Y - output
    outputGradient = derivative_sigmoid(output)
    derivated_output = EO * outputGradient
    EH = derivated_output.dot(weightOut.T) # weightOut.T means transpose of the matrix
    hiddenGradient = derivative_sigmoid(hActivation)# to determine how much did the hidden neurons contribute to the error
    derivated_hiddenGrad = EH * hiddenGradient

    weightOut += hActivation.T.dot(derivated_output) * increment
   # biasOutput += np.sum(derivated_output, axis=0, keepdims=True) * increment
    weight += X.T.dot(derivated_hiddenGrad) * increment
    #bias += np.sum(derivated_hiddenGrad, axis=0, keepdims=True)

print('input \n ' + str(X))
print('output \n :' + str(Y))
print('predicted output \n ', output)
