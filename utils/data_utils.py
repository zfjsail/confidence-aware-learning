import numpy as np

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset


class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired data points
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


# Custom_Dataset class
class Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, transform=None):
        self.x_data = x
        self.y_data = y
        self.data = data_set
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    # return idx
    def __getitem__(self, idx):
        # if self.data == 'cifar':
        #     img = Image.fromarray(self.x_data[idx])
        # elif self.data == 'svhn':
        #     img = Image.fromarray(np.transpose(self.x_data[idx], (1, 2, 0)))
        #
        # x = self.transform(img)

        return self.x_data[idx], self.y_data[idx], idx


def one_hot_encoding(label):
    print("one_hot_encoding process")
    cls = set(label)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, label)))

    return one_hot
