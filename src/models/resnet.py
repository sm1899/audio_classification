import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os 
from ..utils.visualization import plot_training_metrics

class CustomHeadBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, dropout_rate=0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.block(x)

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.resnet = resnet18(pretrained=config.RESNET_CONFIG['pretrained'])
        # Modify first conv layer for single channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if config.RESNET_CONFIG['freeze_backbone']:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Remove the original FC layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add custom head block
        self.head_block = CustomHeadBlock(
            in_features=512,  # ResNet18's output features
            hidden_dim=config.RESNET_CONFIG.get('hidden_dim', 256)
        )
        
        # Final classification layer
        self.classifier = nn.Linear(
            config.RESNET_CONFIG.get('hidden_dim', 256) // 2,
            num_classes
        )
        
        # Initialize weights for custom layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.head_block.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize classifier
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        # Extract features through ResNet backbone
        x = self.resnet(x)
        # Flatten the features
        x = torch.flatten(x, 1)
        # Pass through custom head block
        x = self.head_block(x)
        # Final classification
        x = self.classifier(x)
        return x

    @staticmethod
    def train_model(model, train_loader, val_loader, config, device='cuda', window_type=None):
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Separate parameter groups for different learning rates
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'resnet' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Use different learning rates for backbone and head
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': config.RESNET_CONFIG['learning_rate'] * 0.1},
            {'params': head_params, 'lr': config.RESNET_CONFIG['learning_rate']}
        ])
        
        if config.RESNET_CONFIG['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=config.RESNET_CONFIG['num_epochs'])
        else:
            scheduler = None
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        best_acc = 0
        best_model = None
        
        for epoch in range(config.RESNET_CONFIG['num_epochs']):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f'{train_loss/(batch_idx+1):.3f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            if scheduler:
                scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc="Validating"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # Record metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = train_correct / train_total
            epoch_val_acc = val_correct / val_total
            
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)
            
            print(f"\nEpoch {epoch+1}:")
            print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {100.*epoch_train_acc:.2f}%")
            print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {100.*epoch_val_acc:.2f}%")
            
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                best_model = model.state_dict()
        
        model.load_state_dict(best_model)
        
        if window_type:
            results_dir = os.path.join(config.RESULTS_DIR, 'models', window_type)
            os.makedirs(results_dir, exist_ok=True)
            plot_training_metrics(history, results_dir, 'resnet')
            
        return model, history, best_acc