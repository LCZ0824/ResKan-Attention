import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from sklearn.metrics import roc_auc_score
from .model_architecture import ResKANUltraAttention

class ModelTrainer:
    def __init__(self, config, input_features, device='cuda'):
        self.config = config
        self.input_features = input_features
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def create_model(self):
        model = ResKANUltraAttention(
            input_features=self.input_features,
            config=self.config
        ).to(self.device)
        return model
        
    def train_model(self, fold_data):
        X_train_np = fold_data['X_train_processed']
        y_train_raw = fold_data['y_train_processed']
        
        if hasattr(y_train_raw, 'values'):
            y_train_np = y_train_raw.values.reshape(-1, 1)
        else:
            y_train_np = y_train_raw.reshape(-1, 1)
            
        X_val_np = fold_data['X_val_processed']
        y_val_np = fold_data['y_val_processed']
        
        X_train_th = torch.tensor(X_train_np, dtype=torch.float32).to(self.device)
        y_train_th = torch.tensor(y_train_np, dtype=torch.float32).to(self.device)
        X_val_th = torch.tensor(X_val_np, dtype=torch.float32).to(self.device)
        
        model = self.create_model()
        
        kan_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'kan' in name.lower():
                kan_params.append(param)
            else:
                other_params.append(param)
                
        optimizer = optim.AdamW([
            {'params': kan_params, 'lr': self.config['lr']},
            {'params': other_params, 'lr': self.config['lr'] * 0.8}
        ], weight_decay=self.config['weight_decay'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.7, verbose=False
        )
        
        loss_fn = nn.BCELoss()
        
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config['max_epochs']):
            model.train()
            train_dataset = torch.utils.data.TensorDataset(X_train_th, y_train_th)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.config['batch_size'], shuffle=True
            )
            
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_th)
                val_preds = val_outputs.cpu().numpy().flatten()
                val_auc = roc_auc_score(y_val_np, val_preds) if len(np.unique(y_val_np)) > 1 else 0.5
                
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if (epoch + 1) % 25 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch+1:3d}: Loss {epoch_loss/len(train_loader):.4f}, "
                      f"AUC {val_auc:.4f}, LR {current_lr:.6f}")
                      
            if patience_counter >= self.config['patience']:
                print(f"    Early stop at epoch {epoch+1}. Best: {best_val_auc:.4f}")
                break
                
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        return model, best_val_auc
        
    def evaluate_model(self, model, fold_data):
        X_val_np = fold_data['X_val_processed']
        y_val_np = fold_data['y_val_processed']
        X_val_th = torch.tensor(X_val_np, dtype=torch.float32).to(self.device)
        
        model.eval()
        with torch.no_grad():
            final_preds = model(X_val_th).cpu().numpy().flatten()
            
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        final_preds_class = (final_preds > 0.5).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y_val_np, final_preds),
            'accuracy': accuracy_score(y_val_np, final_preds_class),
            'precision': precision_score(y_val_np, final_preds_class, zero_division=0),
            'recall': recall_score(y_val_np, final_preds_class, zero_division=0),
            'f1': f1_score(y_val_np, final_preds_class, zero_division=0)
        }
        
        return metrics
        
    def save_model(self, model, fold_id, metrics, filename_prefix="model"):
        import pickle
        
        weights_path = f"{filename_prefix}_{fold_id}_weights.pth"
        info_path = f"{filename_prefix}_{fold_id}_info.pkl"
        
        torch.save(model.state_dict(), weights_path)
        
        info_to_save = {
            'fold_id': fold_id,
            'config': self.config,
            'input_features': self.input_features,
            'metrics': metrics,
        }
        
        with open(info_path, 'wb') as f:
            pickle.dump(info_to_save, f)
            
        return weights_path, info_path