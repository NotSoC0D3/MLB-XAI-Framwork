import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation or test set for one epoch.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def train_model(model, trainloader, valloader, optimizer, criterion, scheduler, device, num_epochs, save_path, patience=5):
    """
    Full training loop incorporating early stopping, learning rate scheduling, and history tracking.
    """
    early_stopping = EarlyStopping(patience=patience)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_loss = float("inf")
    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, valloader, criterion, device)

        # Update scheduler
        if scheduler:
            scheduler.step(val_loss)

        curr_lr = optimizer.param_groups[0]["lr"]
        if curr_lr != prev_lr:
            print(f"LR reduced: {prev_lr:.2e} → {curr_lr:.2e}")
            prev_lr = curr_lr

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(curr_lr)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best model saved to {save_path}")
        
        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return history