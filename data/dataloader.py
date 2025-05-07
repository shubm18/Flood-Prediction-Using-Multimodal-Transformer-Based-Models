import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class FloodForecastingDataLoader(Dataset):
    def __init__(self, data_path: str, forecast_horizon: int):
        self.data_path = data_path
        self.forecast_horizon = forecast_horizon
        self.data, self.labels = self.load_data()
        self.balance_data()

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(self.data_path, allow_pickle=True)
        data = data['data']
        labels = data['labels']
        return data, labels

    def balance_data(self):
        flood_indices = np.where(self.labels[:, -self.forecast_horizon:].sum(axis=1) > 0)[0]
        no_flood_indices = np.where(self.labels[:, -self.forecast_horizon:].sum(axis=1) == 0)[0]

        min_samples = min(len(flood_indices), len(no_flood_indices))

        flood_indices = np.random.choice(flood_indices, min_samples, replace=False)
        no_flood_indices = np.random.choice(no_flood_indices, min_samples, replace=False)

        balanced_indices = np.concatenate((flood_indices, no_flood_indices))
        np.random.shuffle(balanced_indices)

        self.data = self.data[balanced_indices]
        self.labels = self.labels[balanced_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx, -self.forecast_horizon:]
        return torch.from_numpy(data).float(), torch.from_numpy(labels).float()
