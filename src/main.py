import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

# Import our custom modules
from src.models import SmallCNN
from src.data_preprocessing import get_dataloaders
from src.train import train_model
from src.evaluation import (
    get_predictions, 
    generate_classification_report, 
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_pr_curve
)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("\n--- Loading Dataset ---")
    trainloader, valloader, testloader, class_names = get_dataloaders(
        split_dir=args.data_dir, 
        batch_size=args.batch_size
    )
    print(f"Classes found: {class_names}")

    # 2. Initialize Model, Loss, and Optimizer
    print("\n--- Initializing Model ---")
    model = SmallCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # 3. Train Model
    if not args.evaluate_only:
        print("\n--- Starting Training ---")
        train_model(
            model=model, 
            trainloader=trainloader, 
            valloader=valloader, 
            optimizer=optimizer, 
            criterion=criterion, 
            scheduler=scheduler, 
            device=device, 
            num_epochs=args.epochs, 
            save_path=args.model_save_path,
            patience=5
        )

    # 4. Evaluate Model
    print("\n--- Evaluating Model on Test Set ---")
    # Load best weights for evaluation
    if os.path.exists(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    else:
        print("Warning: Model weights not found. Evaluating with initialized weights.")
        
    preds, labels, probs = get_predictions(model, testloader, device)
    
    generate_classification_report(labels, preds, class_names)
    plot_confusion_matrix(labels, preds, class_names, save_path="confusion_matrix.png")
    plot_roc_curve(labels, probs, class_names, save_path="roc_curve.png")
    plot_pr_curve(labels, probs, class_names, save_path="pr_curve.png")
    
    print("\nPipeline execution complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease CNN Execution Pipeline")
    parser.add_argument("--data_dir", type=str, default="data/dataset_balanced", help="Directory containing train/val/test splits")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoaders")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--model_save_path", type=str, default="smallcnn_best.pth", help="Path to save the best model weights")
    parser.add_argument("--evaluate_only", action="store_true", help="Skip training and only evaluate the saved model")
    
    args = parser.parse_args()
    main(args)