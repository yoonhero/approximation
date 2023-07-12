import torch
import numpy as np
import matplotlib.pyplot as plt

from model import NotSimpleMLP
from train import train


# def fib(n):
    # return n if n < 2 else fib(n-1) + fib(n-2)

li = [1, 1]
for i in range(98):
    li.append(li[i]+li[i+1])

samples = 30
np_x = np.arange(1, 1+samples)
np_y = np.array(li[:samples])/10000
# np_y = np_y/np.sum(np.exp(np_y))*10000
model = NotSimpleMLP()

nb_epochs = 50
learning_rate = 5e-3
model = train(model, np_x, np_y, learning_rate=learning_rate, epochs=nb_epochs)

tensor_x = torch.Tensor(np_x).unsqueeze(1)
pred_y = model(tensor_x).squeeze(1).detach().numpy()

plt.scatter(np_x, np_y, label='True')
plt.scatter(np_x, pred_y, label="Prediction")
plt.legend()
plt.show()
