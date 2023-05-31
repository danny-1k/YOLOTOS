import torch

def dataloader_collate_fn(data):
    """Creates padded batch of irregular sequences
    """

    # image, classes, coords, lengths

    data.sort(key=lambda x: x[3], reverse=True)
    images, classes, coords, lengths = zip(*data)

    images = torch.stack(images, 0)

    batch_size = images.shape[0]
    max_length = max(lengths)

    classes_padded = torch.zeros(batch_size, max_length).fill_(1) # <p/> token is @ index #1 in vocab
    coords_padded = torch.zeros(batch_size, max_length, 4)

    for i, (classes_seq, coords_seq) in enumerate(zip(classes, coords)):
        end = lengths[i] # length of this sequence
        
        classes_padded[i, :end] = classes_seq
        coords_padded[i, :end] = coords_seq

    return images, classes_padded, coords_padded, lengths