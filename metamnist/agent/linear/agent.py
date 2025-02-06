import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MNISTLogisticRegression(nn.Module):
    def __init__(self, input_size=784):  # 28x28 = 784 pixels per image
        super().__init__()
        self.flatten = nn.Flatten()
        # Logistic regression is effectively a linear layer followed by softmax
        self.logistic = nn.Sequential(
            nn.Linear(input_size, 10),  # 10 classes (digits 0-9)
            nn.LogSoftmax(dim=1)  # Log softmax for numerical stability
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.logistic(x)

class Agent:
    def __init__(self, env=None):
        self.env = env
        
        # Extract shapes from environment dimensions
        obs_size = env.observation_space.shape[0]
        self.train_size = env.train_images_shape[0]
        self.test_size = env.test_images_shape[0]

    # timeout 10 min
    def train_predict(X_train, Y_train, X_test):

    # timeout 30 sec
    def predict(X_test):

        return self.model.predict(X_test)



env 


create data 

agent = Agent()

thread (10 minute timeou)

wait agent.train(data)
-> time, cpu, memory 

thread (10 minute timeout)

result = wait agent.predict(data)
-> compute accuracy
