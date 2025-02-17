import numpy as np
from data_loader import MNetDataset


def split_types(dataset: MNetDataset):
    types = dataset.types
    np.random.shuffle(types)
    n_types = len(types)
    avg_n_types = n_types // 3
    train_types = types[: avg_n_types + 1]
    val_types = types[avg_n_types + 1 : avg_n_types * 2 + 1]
    test_types = types[avg_n_types * 2 + 1 :]
    return train_types, val_types, test_types
