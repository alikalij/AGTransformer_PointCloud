import torch
from utils.utils import load_checkpoint_dynamic, save_checkpoint, check_tensor

def train_model(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, hyperparams):
    """
    حلقه آموزش مدل.
    """
    model, optimizer, start_epoch, train_losses, val_losses = load_checkpoint_dynamic(
        model= model,
        directory= hyperparams['checkpoint_dir'], 
        optimizer= optimizer, 
        for_training= True)
    
    print(f"شروع آموزش از epoch {start_epoch + 1}")
    device = torch.device(hyperparams['device'])
    model = model.to(device)

    accumulation_steps = 4
    scaler = torch.amp.GradScaler()

    for epoch in range(start_epoch, hyperparams['num_epochs']):
        model.train()
        train_loss_total = 0.0
        total_train_points = 0

        for step, batch in enumerate(dataloader_train):
            x, pos, labels = batch.x, batch.pos, batch.y
            x, pos, labels = x.to(device), pos.to(device), labels.to(device)
            check_tensor(x, "x")
            check_tensor(pos, "pos")
            check_tensor(labels, "labels")

            with torch.amp.autocast(device_type=hyperparams['device']):
                outputs, sampled_labels  = model(x, pos, labels)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_points = x.size(0)
            train_loss_total += loss.item() * batch_points
            total_train_points += batch_points

        train_loss = train_loss_total / total_train_points
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{hyperparams['num_epochs']}: Train Loss = {train_loss:.4f}")

        model.eval()
        val_loss_total = 0.0
        total_val_points = 0
        with torch.no_grad():
            for batch in dataloader_val:
                x, pos, labels = batch.x, batch.pos, batch.y
                x, pos, labels = x.to(device), pos.to(device), labels.to(device)
                check_tensor(x, "x")
                check_tensor(pos, "pos")
                check_tensor(labels, "labels")
                with torch.amp.autocast(device_type=hyperparams['device']):
                    outputs, sampled_labels = model(x, pos, labels)
                    loss = criterion(outputs, labels)
                batch_points = x.size(0)
                val_loss_total += loss.item() * batch_points
                total_val_points += batch_points

        val_loss = val_loss_total / total_val_points
        val_losses.append(val_loss)
        if scheduler:
            scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{hyperparams['num_epochs']}: Val Loss = {val_loss:.4f}")

        save_checkpoint(model, optimizer, epoch + 1, train_losses, val_losses,hyperparams['checkpoint_dir'])

    return train_losses, val_losses
