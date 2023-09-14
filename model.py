from typing import *
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import DataLoader
from const import *
from log_cfg import logger



"""
The best model to fine-tune for plant image classification is the ResNet-50 model.

However, we will use the MobileNetV2 model since its less resource-intensive.
"""

class BotanicamModel:
    def __init__(self):

        self.model = efficientnet_b3(progress=True, weights=EfficientNet_B3_Weights.DEFAULT)
        # replace the last layer with a new, untrained layer with 1081 classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, NUM_CLASSES)

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
        logger.info(f"Learning rate: {LR}")

        # directory to save epoch checkpoints
        self.checkpoints_dir = 'checkpoints'
        # create dict if it doesnt exist
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        self.checkpoint_frequency = SAVE_EVERY_N_EPOCHS

    def train(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            epochs: int = EPOCHS,
            lr: float = LR,
            skip_validation: bool = False,
            checkpoint_number: int = 0
        ) -> None:
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
        for epoch in range(checkpoint_number, epochs):
            logger.info(f"Epoch {epoch + 1} of {epochs}")

            if endFlag:
                logger.info("End flag is set. Stopping training...")
                break

            # train without accumulating gradients
            self.model.train()
            self.optimizer.zero_grad()

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
                loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
                loop.set_postfix(loss=loss.item())

                # memory cleanup
                del data, targets, scores, loss
                torch.cuda.empty_cache()
                
            # Checkpointing: Save the model after every N epochs
            if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
                checkpoint_filename = os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                self.save(checkpoint_filename, epoch + 1)


            # then we validate so we can track improvements
            if not skip_validation:
                endFlag, val = self.__validate(val_loader)
                self.history.append(val)
            else:
                # append loss instead
                self.history.append(loss.item())
        
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"Training time: {elapsed:.2f}s")


    def resume_training(self, checkpoint_path: str, train_loader: DataLoader, val_loader: DataLoader):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file to resume from.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            val_loader (torch.utils.data.DataLoader): Validation data loader.
        """
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")

        # Load the checkpoint epoch data
        self.load(checkpoint_path)
        # Continue training
        self.train(train_loader, val_loader, lr=LR, checkpoint_number=self.epoch)
    
    def test(self, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Tests the model by calculating the accuracy on the test set

        Args:
            test_loader (torch.utils.data.DataLoader): Test data loader

        Returns:
            float: Test accuracy (0 to 1)
        """
        # set model to evaluation mode
        self.model.eval()

        # initialize variables
        num_correct = 0
        num_samples = 0

        # disable gradient calculation
        with torch.no_grad():
            loop = tqdm(test_loader)
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
                loop.set_description(f"Testing")
                loop.set_postfix(correct=num_correct, samples=num_samples, accuracy=(float(num_correct) / float(num_samples)) * 100.0)

        # calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100.0
        logger.info(f"Test accuracy: {accuracy:.2f}%")

        return accuracy
    
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predicts the class of an image

        Args:
            image (torch.Tensor): Image tensor

        Returns:
            torch.Tensor: Predicted class
        """
        # set model to evaluation mode
        self.model.eval()

        # disable gradient calculation
        with torch.no_grad():
            # move data to device
            image = image.to(self.device)

            # forward propagation
            scores = self.model(image)
            _, predictions = scores.max(1)

            return predictions
        
    def predict_classify(self, image: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the class of an image and returns the class and the probability

        Args:
            image (torch.Tensor): Image tensor
            k (int, optional): Number of classes to return

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted class and probability
        """
        # set model to evaluation mode
        self.model.eval()

        # disable gradient calculation
        with torch.no_grad():
            # move data to device
            image = image.to(self.device)

            # forward propagation
            scores = self.model(image)
            _, predictions = scores.topk(k, dim=1)
            probabilities = nn.functional.softmax(scores, dim=1)

            return predictions, probabilities

    def image_to_tensor(self, image: Image) -> torch.Tensor:
        """
        Converts an image to a tensor

        Args:
            image (Image): Image to convert

        Returns:
            torch.Tensor: Converted image
        """
        # convert image to tensor
        transform = transforms.Compose([
            Resize((224, 224)),
            ToTensor()
        ])
        image = transform(image)

        return image

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

        
        # check if model has converged
        if accuracy - self.best_accuracy < CONVERGENCE_THRESHOLD:
            self.convergence += 1
        else:
            self.convergence = 0

        return (self.convergence >= MAX_NON_IMPROVEMENT_EPOCHS), accuracy
    
    def save(self, path=MODEL_PATH, epoch=0) -> None:
        """
        Saves the model checkpoint

        Args:
            path (str): Path to save the checkpoint
            epoch (int): Current epoch
        """
        logger.info("Saving the model checkpoint...")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_accuracy': self.best_accuracy,
            'history': self.history,
            'convergence': self.convergence
        }
        torch.save(checkpoint, path)
        logger.info("Model checkpoint saved")

    def load(self, path=MODEL_PATH) -> None:
        """
        Loads the model
        """
        logger.info("Loading the model...")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch'] 
        self.best_accuracy = checkpoint['best_accuracy']
        self.history = checkpoint['history']
        self.convergence = checkpoint['convergence']
        logger.info("Model loaded")
        return checkpoint
    
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

