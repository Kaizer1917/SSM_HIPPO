import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mamba import SSM_HIPPO, ModelArgs
import matplotlib.pyplot as plt
import wandb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def get_scheduler(optimizer, model_args, train_loader):
    # Calculate total steps for all epochs
    total_steps = len(train_loader) * model_args.num_epochs
    
    # Warmup steps (10% of total steps)
    warmup_steps = int(0.1 * total_steps)
    
    # Create scheduler chain
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        # Linear warmup
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        ),
        # Cosine annealing with restarts
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_steps // 3,  # Restart every third of training
            T_mult=2,  # Double the restart interval each time
            eta_min=model_args.learning_rate * 0.01  # Minimum LR
        )
    ])
    
    return scheduler

def train(model_args, train_loader, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    wandb.init(project=model_args.project_name)
    wandb.config.update(vars(model_args))

    # Initialize model with optimized HiPPO configurations
    ssm_model = SSM_HIPPO(model_args).to(device)
    
    # Initialize layer-wise learning rates for better training
    param_groups = []
    for i, layer in enumerate(ssm_model.ssm_hippo_blocks):
        param_groups.append({
            'params': layer.parameters(),
            'lr': model_args.learning_rate * (0.9 ** i)  # Decrease learning rate for deeper layers
        })
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    
    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=model_args.learning_rate,
        epochs=model_args.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% warmup
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(model_args.num_epochs):
        ssm_model.train()
        running_loss = 0.0

        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            
            # Apply dynamic HiPPO transition scaling
            output = ssm_model(X_batch, epoch/model_args.num_epochs)
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient Clipping with dynamic threshold
            max_norm = 1.0 * (0.9 ** (epoch // 10))  # Reduce clipping threshold over time
            torch.nn.utils.clip_grad_norm_(ssm_model.parameters(), max_norm=max_norm)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                    "step": i + 1,
                    "gradient_norm": max_norm
                })

        # Validation phase
        ssm_model.eval()
        val_loss = 0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                pred = ssm_model(X_batch, epoch/model_args.num_epochs)
                all_preds.append(pred.cpu().numpy())
                all_true.append(y_batch.cpu().numpy())
                
                loss = mean_squared_error(y_batch.cpu().numpy(), pred.cpu().numpy(), squared=False)
                val_loss += loss

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': ssm_model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Calculate and log metrics
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        r2 = r2_score(all_true, all_preds)
        
        wandb.log({
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss,
            "r2_score": r2,
            "epoch": epoch + 1
        })

    wandb.finish()
    return ssm_model

# Main execution remains the same with additional arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the SSM-Hippo model.")
    parser.add_argument('--train_dataset', type=str, default="ssm-hippo-project", help='The training dataset.')
    parser.add_argument('--test_dataset', type=str, default="ssm-hippo-project", help='The test dataset.')
    parser.add_argument('--project_name', type=str, default="ssm-hippo-project", help='WandB project name.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--seq_len', type=int, required=True, help='Length of the input sequences.')
    parser.add_argument('--forecast_len', type=int, required=True, help='Length of the forecast sequences.')
    parser.add_argument('--input_dim', type=int, required=True, help='Input dimension for the model.')
    parser.add_argument('--hidden_dim', type=int, required=True, help='Hidden dimension for the model.')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of layers in the model.')
    
    args = parser.parse_args()

    # Create proper dataset objects first
    train_dataset = YourDatasetClass(args.train_dataset)  # Need to implement this
    test_dataset = YourDatasetClass(args.test_dataset)    # Need to implement this
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model_args = ModelArgs(
        seq_len=args.seq_len,
        forecast_len=args.forecast_len,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        project_name=args.project_name,  # Add training-related args to ModelArgs
    )

    train(model_args, train_loader, test_loader)
