import numpy as np

class Activation_Softmax():
  def forward(self, inputs):
    exponent_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exponent_values / np.sum(exponent_values, axis=1, keepdims=True)
    self.output = probabilities