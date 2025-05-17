import os
import torch
from datetime import datetime

def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, base_path="./saved_models/asgformer"):
    """
    ذخیره‌ی checkpoint مدل همراه با مدیریت نسخه.
    """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"{base_path}_epoch{epoch}_{timestamp}.pth"

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': list(train_losses),
        'val_losses': list(val_losses),
    }
    try:
        torch.save(checkpoint, checkpoint_path)
        print(f"مدل در epoch {epoch} ذخیره شد: {checkpoint_path}")
    except Exception as e:
        print(f"خطا در ذخیره checkpoint: {e}")

def load_checkpoint(model, optimizer=None, for_training=False, checkpoint_path=""):
    """
    بارگذاری checkpoint مدل با مدیریت خطا.
    """
    if not os.path.exists(checkpoint_path):
        print(f"فایل checkpoint '{checkpoint_path}' یافت نشد. آموزش از ابتدا آغاز می‌شود.")
        return model, optimizer, 0, [], []
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0)
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        if not isinstance(train_losses, list):
            train_losses = [train_losses]
        if not isinstance(val_losses, list):
            val_losses = [val_losses]
        if for_training and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"مدل از epoch {start_epoch} بارگذاری شد. آخرین Loss: {train_losses[-1] if train_losses else 'N/A'}")
        return model, optimizer, start_epoch, train_losses, val_losses
    except Exception as e:
        print(f"خطا در بارگذاری checkpoint: {e}")
        return model, optimizer, 0, [], []

def find_latest_checkpoint(directory):
    """
    پیدا کردن آخرین checkpoint در دایرکتوری مشخص شده.
    """
    if not os.path.isdir(directory):
        return None
    checkpoint_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
    if not checkpoint_files:
        return None
    latest_file = max(checkpoint_files)
    return os.path.join(directory, latest_file)

def load_checkpoint_dynamic(model, optimizer=None, for_training=False, directory="./saved_models"):
    checkpoint_path = find_latest_checkpoint(directory)
    if checkpoint_path is None:
        print("هیچ checkpointی در دایرکتوری یافت نشد. آموزش از ابتدا آغاز می‌شود.")
        return model, optimizer, 0, [], []
    return load_checkpoint(model, optimizer, for_training, checkpoint_path)

def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN!")
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf!")

def read_file_list(file_path):
    """
    خواندن مسیر فایل‌های داده از فایل متنی.
    """
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines
