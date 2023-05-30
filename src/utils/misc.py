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