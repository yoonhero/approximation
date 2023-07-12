import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def train(model, data_x, data_y, learning_rate=2e-2, epochs=100):
    tensor_x = torch.Tensor(data_x).unsqueeze(1) 
    tensor_y = torch.Tensor(data_y).unsqueeze(1)

    my_dataset = TensorDataset(tensor_x,tensor_y) 
    my_dataloader = DataLoader(my_dataset, shuffle=True) 

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(my_dataloader):
            pred = model(x)

            loss = F.mse_loss(y, pred)
            # loss = torch.abs(y-pred)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print(f"EPOCH {epoch}: {loss.item():.2f}")

    return model