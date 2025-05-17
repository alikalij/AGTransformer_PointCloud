import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os
import h5py

class PointCloudProcessor:
    def __init__(self, num_points, knn_param, use_cache=True):
        self.num_points = num_points
        self.knn_param = knn_param
        self.scaler = StandardScaler()
        self.cache = {} if use_cache else None

    def normalize_data(self, points, features):
        coords = points.copy()
        coords = coords.astype(np.float64)
        self.scaler.partial_fit(coords)
        coords = self.scaler.transform(coords)
        return coords, features

    # متدهای sample_points و create_graph می‌توانند بر حسب نیاز اضافه شوند.


class H5Dataset(Dataset):
    def __init__(self, file_paths, processor, dataset_path="", cache=True):
        """
        کلاس Dataset برای خواندن فایل‌های H5 با امکان caching.

        Args:
            file_paths (list): لیستی از مسیر فایل‌های H5 (نسبت به dataset_path).
            processor (PointCloudProcessor): پردازشگر ابرنقاط برای نرمال‌سازی و ساخت گراف.
            dataset_path (str): مسیر پایه داده‌ها.
            cache (bool): فعال بودن کشینگ.
        """
        self.file_paths = file_paths
        self.processor = processor
        self.dataset_path = dataset_path
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dataset_path, self.file_paths[idx])
        if self.cache is not None and file_path in self.cache:
            return self.cache[file_path]

        try:
            with h5py.File(file_path, 'r') as f:
                data_full = f['data'][:]   # فرض: داده‌ها در کلید 'data'
                labels = f['label'][:]
        except Exception as e:
            print(f"خطا در خواندن فایل {file_path}: {e}")
            return None

        points = data_full[:, :3]
        features = data_full[:, 3:]
        points, features = self.processor.normalize_data(points, features)

        points_tensor = torch.tensor(points, dtype=torch.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        data_obj = Data(x=features_tensor, pos=points_tensor, y=labels_tensor)

        if self.cache is not None:
            self.cache[file_path] = data_obj

        return data_obj


def read_file_list(file_path):
    """
    خواندن مسیر فایل‌های داده از فایل متنی.
    """
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines


class PointCloudDataset(Dataset):
    def __init__(self, data, processor):
        """
        Dataset برای داده‌های ابرنقاط که داده‌ها از قبل بارگذاری شده‌اند.

        Args:
            data (list): لیستی از نمونه‌های داده (numpy array).
            processor (PointCloudProcessor): پردازشگر داده‌ها.
        """
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        points = sample['points']
        features = sample['features']
        labels = sample['labels']

        points, features = self.processor.normalize_data(points, features)
        points_tensor = torch.tensor(points, dtype=torch.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        data_obj = Data(x=features_tensor, pos=points_tensor, y=labels_tensor)
        return data_obj
