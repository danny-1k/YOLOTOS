import torch

def clean_prediction(classes, bbs):
    """Clean output of model - Remove EOS & padding tokens, etc

    Args:
        classes : Tensor of shape (N,) where N is the number of objects detected (Including EOS & PAD)
        bbs : Tensor of shape (N, 4). N must be equal to N of `classes`.

    """

    N = classes.shape[0]
    indices = torch.arange(N).to(bbs.device)
    # indices 0, 1  correspond to <s>, </s> and </p> respectively
    mask = (classes != 0) & (classes != 1)

    indices = indices[mask]

    classes = classes[indices]
    bbs = bbs[indices]

    return classes, bbs