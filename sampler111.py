"""Separate balanced version of sampler.py; no extra class-1 draw."""
import random
from torch.utils.data import Sampler


class BalancedSampler111(Sampler):
    def __init__(self, dataset, batch_size=32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.labels = [int(label) % 3 for label in dataset.label_list]
        self.class_to_index = [[], [], []]
        for index, label in enumerate(self.labels):
            self.class_to_index[label].append(index)
        self.class_counts = [len(group) for group in self.class_to_index]
        if not all(self.class_counts):
            raise ValueError("All three dialects must be present")
        self.max_len = max(self.class_counts)

    def __len__(self):
        return (3 * self.max_len + self.batch_size - 1) // self.batch_size

    def prepare_batches(self):
        # Fresh lists per epoch, matching the old per-epoch sampler construction.
        groups = [group.copy() for group in self.class_to_index]
        for group in groups:
            random.shuffle(group)
        indices = []
        classes = [0, 1, 2]
        for index in range(self.max_len):
            random.shuffle(classes)
            for label in classes:
                indices.append(groups[label][index % len(groups[label])])
        return [indices[i:i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)]

    def __iter__(self):
        yield from self.prepare_batches()
