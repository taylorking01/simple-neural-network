import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
import numpy as np
from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU
from Activation_Softmax import Activation_Softmax
from Loss_CategoricalCrossEntropy import Loss_CategoricalCrossEntropy

def main():
    X, y = spiral_data(samples=100, classes=3)
    d1 = Layer_Dense(2,3)
    a1 = Activation_ReLU()
    
    d2 = Layer_Dense(3,12)
    a2 = Activation_Softmax()

    loss_function = Loss_CategoricalCrossEntropy()
    
    d1.forward(X)
    a1.forward(d1.output)
    
    d2.forward(a1.output)
    a2.forward(d2.output)
    
    print(a2.output[:5])

    loss = loss_function.calculate(a2.output, y)
    print("Loss: ", loss)

    predictions = np.argmax(a2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis = 1)
    accuracy = np.mean(predictions == y)

    print("Accuracy: ", accuracy)