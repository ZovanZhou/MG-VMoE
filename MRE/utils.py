import numpy as np

def split_types(types):
    filtered_types = ["Other", "None", "none", "NA"]
    types = [ele for ele in types if ele not in filtered_types]
    np.random.shuffle(types)
    n_types = len(types)
    avg_n_types = n_types // 3
    train_types = types[: avg_n_types + 1]
    val_types = types[avg_n_types + 1 : avg_n_types * 2 + 1]
    test_types = types[avg_n_types * 2 + 1 :]
    return train_types, val_types, test_types
