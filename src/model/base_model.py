import torch
import os
import tools

from abc import ABC, abstractmethod

class BaseModel(ABC):

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.logger = args.logger

    @abstractmethod
    def load_pretrained_weights(checkpoint):
        """
        load weights from checkpoint
        """
        pass

    @abstractmethod
    def save(self, save_path=None):
        """
        code to save the model weights
        """
        pass

    @abstractmethod
    def set_inputs(self, inputs):
        """
        set initial inputs to the model
        """
        pass

    @abstractmethod
    def forward(self):
        """
        handle forward pass
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        handle backward pass and weights update
        """
        pass
    
    @abstractmethod
    def setup(self):
        """
        setup models for training/testing
        """
        pass
    
    @abstractmethod
    def print_stats(self, logger=None):
        """
        print stats
        """
        pass
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        