import torch
import torch.optim as optim
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from models.model import ASGFormer
from data.dataset import PointCloudProcessor, H5Dataset, read_file_list
from train import train_model

def main():
    # پیکربندی داده و مدل
    hyperparams = {
        'learning_rate': 1e-4,
        'batch_size': 2,
        'num_epochs': 45,
        'knn_param': 4,
        'dropout_param': 0.1,
        'weight_decay': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_points': 4096
    }
    stages_config = [
        {'hidden_dim': 32, 'num_layers': 1, 'downsample_ratio': 1.0},
        {'hidden_dim': 64, 'num_layers': 2, 'downsample_ratio': 0.25},
        {'hidden_dim': 128, 'num_layers': 4, 'downsample_ratio': 0.25},
        {'hidden_dim': 256, 'num_layers': 2, 'downsample_ratio': 0.25},
        {'hidden_dim': 512, 'num_layers': 2, 'downsample_ratio': 0.25},
    ]

    dataset_path = "./data/s3dis-small"
    train_files = read_file_list(f"{dataset_path}/list/train5.txt")
    val_files = read_file_list(f"{dataset_path}/list/val5.txt")

    processor = PointCloudProcessor(num_points=hyperparams['num_points'],
                                    knn_param=hyperparams['knn_param'],
                                    use_cache=True)
    train_dataset = H5Dataset(train_files, processor, dataset_path)
    val_dataset = H5Dataset(val_files, processor, dataset_path)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

    feature_dim = 6
    main_input_dim = 9
    main_output_dim = 13

    model = ASGFormer(feature_dim=feature_dim,
                      main_input_dim=main_input_dim,
                      main_output_dim=main_output_dim,
                      stages_config=stages_config,
                      knn_param=hyperparams['knn_param'],
                      dropout_param=hyperparams['dropout_param'])

    model.to(hyperparams['device'])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, hyperparams)


if __name__ == "__main__":
    main()
