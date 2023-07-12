import torch
import numpy as np
import matplotlib.pyplot as plt

from model import SimpleMLP
from train import train

np_x = np.arange(1, 100)
np_y = np_x ** 2
model = SimpleMLP()

nb_epochs = 500
learning_rate = 1e-2
model = train(model, np_x, np_y, learning_rate=learning_rate, epochs=nb_epochs)

tensor_x = torch.Tensor(np_x).unsqueeze(1)
pred_y = model(tensor_x).squeeze(1).detach().numpy()

plt.scatter(np_x, np_y, label='True')
plt.scatter(np_x, pred_y, label="Prediction")
plt.legend()
plt.show()
