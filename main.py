import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

def exec():
    X, y = spiral_data(samples=100, classes=3)
    d1 = Layer_Dense(2,3)
    a1 = Activation_ReLU()
    
    d2 = Layer_Dense(3,12)
    a2 = Activation_Softmax()
    
    d1.forward(X)
    a1.forward(d1.output)
    
    d2.forward(a1.output)
    a2.forward(d2.output)
    
    print(a2.output[:5])