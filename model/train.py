import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mamba import SSM_HIPPO, ModelArgs
import matplotlib.pyplot as plt
import wandb
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def ensemble_predict(ssm_model, catboost_model, X_batch, device):
    """Combine predictions from SSM-Hippo and CatBoost models"""
    with torch.no_grad():
        ssm_pred = ssm_model(X_batch.to(device)).cpu().numpy()
    catboost_pred = catboost_model.predict(X_batch.cpu().numpy())
    return 0.6 * ssm_pred + 0.4 * catboost_pred

def train(model_args, train_loader, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    wandb.init(project=model_args.project_name)
    wandb.config.update(vars(model_args))

    # Initialize models
    ssm_model = SSM_HIPPO(model_args).to(device)
    catboost_model = CatBoostRegressor(
        iterations=model_args.catboost_iterations,
        learning_rate=model_args.catboost_learning_rate,
        depth=model_args.catboost_depth,
        loss_function='RMSE',
        eval_metric='RMSE',
        early_stopping_rounds=50,
        verbose=100
    )

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(ssm_model.parameters(), lr=model_args.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=model_args.learning_rate,
        epochs=model_args.num_epochs,
        steps_per_epoch=len(train_loader)
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # K-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for epoch in range(model_args.num_epochs):
        ssm_model.train()
        running_loss = 0.0
        all_train_data = []
        all_train_labels = []

        print(f"\nEpoch [{epoch+1}/{model_args.num_epochs}]")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Training loop
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Store data for CatBoost
            all_train_data.append(X_batch.cpu().numpy())
            all_train_labels.append(y_batch.cpu().numpy())

            optimizer.zero_grad()
            output = ssm_model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(ssm_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                    "step": i + 1
                })

        # Train CatBoost model
        X_train = np.concatenate(all_train_data)
        y_train = np.concatenate(all_train_labels)
        
        # Cross-validation training for CatBoost
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            catboost_model.fit(
                X_fold_train, y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                silent=True
            )
            
            fold_pred = catboost_model.predict(X_fold_val)
            cv_scores.append(mean_squared_error(y_fold_val, fold_pred, squared=False))

        # Validation phase
        ssm_model.eval()
        val_loss = 0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # Ensemble prediction
                combined_pred = ensemble_predict(ssm_model, catboost_model, X_batch, device)
                all_preds.append(combined_pred)
                all_true.append(y_batch.cpu().numpy())
                
                loss = mean_squared_error(y_batch.cpu().numpy(), combined_pred, squared=False)
                val_loss += loss

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'ssm_model_state_dict': ssm_model.state_dict(),
                'catboost_model': catboost_model,
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'best_ensemble_model.pth')
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
            "cv_score_mean": np.mean(cv_scores),
            "cv_score_std": np.std(cv_scores),
            "epoch": epoch + 1
        })

    wandb.finish()
    return ssm_model, catboost_model

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
    parser.add_argument('--catboost_iterations', type=int, default=1000)
    parser.add_argument('--catboost_learning_rate', type=float, default=0.03)
    parser.add_argument('--catboost_depth', type=int, default=6)
    
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
        catboost_iterations=args.catboost_iterations,
        catboost_learning_rate=args.catboost_learning_rate,
        catboost_depth=args.catboost_depth
    )

    train(model_args, train_loader, test_loader)
