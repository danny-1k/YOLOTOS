import socket
from datetime import datetime
import torch

def denormalize(image):
    """Perform Inverse transforms on image

    Args:
        image : Image Tensor of shape (3, H, W)
    """

    mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1) # (3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1) # (3, 1, 1)

    image = (image*std) + mean

    return image


def get_hostname_and_time_string():
    current_time = datetime.now().strftime("%Y%m%d_%H:%M:%S_%f")
    hostname = socket.gethostname()
    string = f"{current_time}_{hostname}"
    return string