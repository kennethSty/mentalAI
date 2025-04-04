import torch

class DeviceManager:
    def __init__(self, verbose=False):
        self.device = self.set_device()
        if verbose:
            print("Using device:", self.device)

    def set_device(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        return device

    def get_device(self):
        return self.device