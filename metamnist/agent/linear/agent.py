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
    def __init__(self, env, player_name=None):
        self.env = env
        
        # Extract shapes from environment dimensions
        obs_size = env.observation_space.shape[0]
        self.train_size = env.train_images_shape[0]
        self.test_size = env.test_images_shape[0]
        
        # Initialize logistic regression model
        self.model = MNISTLogisticRegression()
        # Use NLL Loss since we're using LogSoftmax (equivalent to CrossEntropy)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def _extract_data(self, observation):
        """Extract and reshape train/test data from flattened observation"""
        # Calculate sizes
        train_images_size = np.prod(self.env.train_images_shape)
        train_labels_size = self.env.train_labels_shape[0]
        test_images_size = np.prod(self.env.test_images_shape)
        
        # Extract components
        idx = 0
        
        # Train images
        train_images = observation[idx:idx + train_images_size]
        train_images = train_images.reshape(self.env.train_images_shape)
        idx += train_images_size
        
        # Train labels (denormalize from [0,1] back to [0,9])
        train_labels = observation[idx:idx + train_labels_size]
        train_labels = (train_labels * 9.0).round().astype(np.int64)
        idx += train_labels_size
        
        # Test images
        test_images = observation[idx:idx + test_images_size]
        test_images = test_images.reshape(self.env.test_images_shape)
        
        return train_images, train_labels, test_images
        
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None, action_mask=None):
        # Extract and process data
        train_images, train_labels, test_images = self._extract_data(observation)
        
        # Convert to PyTorch tensors
        train_images = torch.FloatTensor(train_images)
        train_labels = torch.LongTensor(train_labels)
        test_images = torch.FloatTensor(test_images)
        
        # Train the model
        self.model.train()
        for _ in range(5):  # Multiple epochs per step
            self.optimizer.zero_grad()
            output = self.model(train_images)
            loss = self.criterion(output, train_labels)
            loss.backward()
            self.optimizer.step()
        
        # Predict test labels
        self.model.eval()
        with torch.no_grad():
            test_output = self.model(test_images)
            predictions = test_output.argmax(dim=1).numpy()
            
        return predictions