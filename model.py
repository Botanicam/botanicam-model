from typing import *
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from const import *
from log_cfg import logger

"""
The best model to fine-tune for plant image classification is the ResNet-50 model.

However, we will use the MobileNetV2 model since its less resource-intensive.
"""

class BotanicamModel:
    def __init__(self):
        self.model = mobilenet_v2(progress=True, weights=MobileNet_V2_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, NUM_CLASSES)
        )

        # loss & optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.best_accuracy = 0.0
        # how many epochs have passed without improvement
        self.convergence = 0
        # validation history
        self.history = []

        logger.info(f"Loaded model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Loss: {self.loss.__class__.__name__}")
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Learning rate: {LR}")

    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> None:
        """
        Trains the model

        Args:
            train_loader (torch.utils.data.DataLoader): Training data loader
            val_loader (torch.utils.data.DataLoader): Validation data loader
        """
        logger.info("Training the model...")
        start_time = time.time()

        # train the model
        endFlag = False
        for epoch in range(EPOCHS):
            logger.info(f"Epoch {epoch + 1} of {EPOCHS}")

            if endFlag:
                logger.info("End flag is set. Stopping training...")
                break

            # train
            self.model.train()

            loop = tqdm(train_loader)
            for batch, (data, targets) in enumerate(loop):
                # move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)

                # forward propagation
                scores = self.model(data)
                loss = self.loss(scores, targets)

                # backward propagation
                self.optimizer.zero_grad()
                loss.backward()

                # gradient descent or adam step
                self.optimizer.step()

                # update tqdm loop
                loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
                loop.set_postfix(loss=loss.item())

            # then we validate so we can track improvements
            endFlag, val = self.__validate(val_loader)
            self.history.append(val)
        
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"Training time: {elapsed:.2f}s")

    def __validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[bool, float]:
        """
        Validates the model
        This uses the validation set to validate the model during training
        If the model has not improved MAX_NON_IMPROVEMENT_EPOCHS epochs, (based on CONVERGENCE_THRESHOLD)
        it will return True (end flag for training)

        Args:
            val_loader (torch.utils.data.DataLoader): Validation data loader

        Returns:
            bool: Whether to end training or not
            float: Validation accuracy
        """
        logger.debug("Validating the model...")
        start_time = time.time()

        # set model to evaluation mode
        self.model.eval()

        # initialize variables
        num_correct = 0
        num_samples = 0

        # disable gradient calculation
        with torch.no_grad():
            loop = tqdm(val_loader)
            for batch, (data, targets) in enumerate(loop):
                # move data to device
                data = data.to(self.device)
                targets = targets.to(self.device)

                # forward propagation
                scores = self.model(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)

                # update tqdm loop
                loop.set_description(f"Validation")
                loop.set_postfix(correct=num_correct, samples=num_samples, accuracy=(float(num_correct) / float(num_samples)) * 100.0)

        # calculate accuracy
        end_time = time.time()
        elapsed = end_time - start_time
        accuracy = float(num_correct) / float(num_samples) * 100.0
        logger.info(f"Validation accuracy: {accuracy:.2f}%, time: {elapsed:.2f}s")

        # save the model if it has improved
        if accuracy > self.best_accuracy:
            logger.debug(f"Validation accuracy increased ({self.best_accuracy:.6f} --> {accuracy:.6f}). Saving model...")
            self.best_accuracy = accuracy
            torch.save(self.model.state_dict(), MODEL_PATH)
            logger.debug("Model saved")
        
        # check if model has converged
        if accuracy - self.best_accuracy < CONVERGENCE_THRESHOLD:
            self.convergence += 1
        else:
            self.convergence = 0

        return (self.convergence >= MAX_NON_IMPROVEMENT_EPOCHS), accuracy
    
    def save(self, path=MODEL_PATH) -> None:
        """
        Saves the model
        """
        logger.info("Saving the model...")
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved")

    def load(self, path=MODEL_PATH) -> None:
        """
        Loads the model
        """
        logger.info("Loading the model...")
        self.model.load_state_dict(torch.load(path))
        logger.info("Model loaded")
    
    def plot_training(self) -> None:
        """
        Plots the training history
        """
        logger.info("Plotting training history...")
        plt.plot(self.history)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training History')
        plt.show()