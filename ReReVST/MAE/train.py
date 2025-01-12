import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torch.cuda.amp import autocast, GradScaler
from models_mae import MaskedAutoencoderViT
from vit import ViT, Transformer  
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import ImageFolder
os.makedirs('model', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, _ = self.subset[idx]
        data = data.convert('RGB')
        if self.transform:
            data = self.transform(data)
        return data, 0 

def train(model, train_loader, optimizer, scaler, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training", unit="batch")
    for i, (inputs, _) in enumerate(pbar):
        inputs = inputs.to(device)
        with autocast():
            loss, _, _ = model(inputs)
            loss = loss / accumulation_steps  

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / (i + 1)
        pbar.set_postfix({"Mean loss": f"{avg_loss:.4f}"})

    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    pbar = tqdm(val_loader, desc="Validating", unit="batch")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)
            with autocast():
                loss, _, _ = model(inputs)
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            pbar.set_postfix({"Mean loss": f"{avg_loss:.4f}"})

    return total_loss / len(val_loader)

def main():
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    batch_size = 2
    learning_rate = 1e-4
    num_epochs = 20
    mask_ratio = 0.75

    encoder = ViT(
        image_size=(1024, 1024),  
        patch_size=(16, 16),   
        num_classes=1000,      
        dim=768,        
        depth=12,          
        num_heads=12,        
        mlp_dim=3072,         
        dim_per_head=64,        
        dropout=0.1,       
        pool='cls',            
        channels=3           
    )

    model = MaskedAutoencoderViT(
        encoder=encoder,
        decoder_dim=512,
        mask_ratio=mask_ratio,
        decoder_depth=8,
        num_decoder_heads=8,
        decoder_dim_per_head=64
    )
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05
    )

    scaler = GradScaler()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),  
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(),                 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],                  
            std=[0.229, 0.224, 0.225]                   
        )
    ])

    transform_val = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')), 
        transforms.Resize(256),                           
        transforms.CenterCrop(224),                      
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],                 
            std=[0.229, 0.224, 0.225]                  
        )
    ])
    try:
        full_dataset = datasets.Caltech101(
            root='./data',       
            download=True,
            transform=None         
        )
    except Exception as e:
        print(f"Error loading Caltech101 dataset: {e}")
        return
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    train_dataset = SubsetWithTransform(train_subset, transform=transform_train)
    val_dataset = SubsetWithTransform(val_subset, transform=transform_val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    if len(train_dataset) > 0:
        sample_image, _ = train_dataset[0]
        print(f"Sample image shape: {sample_image.shape}") 
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, scaler, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train loss: {train_loss:.4f}, "
              f"Test loss: {val_loss:.4f}, "
              f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'model/mae_best.pth')
            print("Saved Best Model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"model/mae_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved Checkpoint: {checkpoint_path}")

    print("Training Completed")

if __name__ == '__main__':
    main()
