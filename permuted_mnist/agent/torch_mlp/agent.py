import torch
from torch import nn
import numpy as np
from time import time


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        hidden_sizes = [400, 400]

        layers = []
        d_in = 28 ** 2
        for i, n in enumerate(hidden_sizes):
            layers.append(nn.Linear(d_in, n))
            layers.append(nn.BatchNorm1d(n))
            layers.append(nn.ReLU())
            d_in = n

        layers += [nn.Linear(d_in, 10)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.model(x)




class Agent:

    def __init__(self, env=None):
        self.env = env
        self.model = Model()

        self.batch_size = 16
        self.validation_fraction = 0.2
        self.verbose=True

    def test(self, X_test):
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float() / 255.0
        logits = self.model.forward(X_test)
        return logits.argmax(-1).detach().cpu().numpy()


    def train(self, X_train, Y_train):

        # validation set:
        N_val = int(X_train.shape[0] * self.validation_fraction)
        X_train, X_val = X_train[N_val:], X_train[:N_val]
        Y_train, Y_val = Y_train[N_val:], Y_train[:N_val]

        X_train = torch.from_numpy(X_train).float() / 255.0
        X_val = torch.from_numpy(X_val).float() / 255.0
        Y_train = torch.from_numpy(Y_train)

        N = len(X_train)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        ce = nn.CrossEntropyLoss()

        for i_epoch in range(10):
            perm = np.random.permutation(N)
            X = X_train[perm]
            Y = Y_train[perm]

            for i in range(0, N, self.batch_size):
                x = X[i:i + self.batch_size]
                y = Y[i:i + self.batch_size]

                optimizer.zero_grad()
                logits = self.model(x)
                loss = ce(logits, y)
                loss.backward()
                optimizer.step()

            if self.verbose and self.validation_fraction > 0:
                y_predict = self.test(X_val)
                is_correct = y_predict == Y_val
                acc = np.mean(is_correct)
                print(f"epoch {i_epoch}: {acc:0.04f}%")


if __name__ == "__main__":
    agent = Agent()
    X_train = np.load("./data/mnist_train_images.npy")
    Y_train = np.load("./data/mnist_train_labels.npy")
    X_test = np.load("./data/mnist_test_images.npy")
    Y_test = np.load("./data/mnist_test_labels.npy")

    t0 = time()
    agent.train(X_train, Y_train)
    y_predict = agent.test(X_test)
    is_correct = y_predict == Y_test
    acc = np.mean(is_correct)
    print(f"Test accuracy: {acc:0.04f} in {time() - t0:.2f} seconds")


