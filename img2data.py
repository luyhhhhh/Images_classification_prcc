import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model_with_early_stopping(model, dataloaders, criterion, optimizer, scheduler, early_stopping, num_epochs=25,
                                    device='cpu'):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            phase_dataloader = tqdm(dataloaders[phase], desc=f"Epoch {epoch + 1} - {phase}")
            for inputs, labels in phase_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                phase_dataloader.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            writer.add_scalar(f'{phase} Loss', epoch_loss, epoch + 1)
            writer.add_scalar(f'{phase} Accuracy', epoch_acc, epoch + 1)

            if phase == 'val':
                scheduler.step()
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

        if early_stopping.early_stop:
            break

    writer.close()
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


def main():
    image_dir = "image"
    batch_size = 16
    num_epochs = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = prepare_data_loaders(image_dir, test_size=0.2, val_size=0.1,
                                                                 batch_size=batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader}

    model = models.resnet101(pretrained=True)
    model = fine_tune_model(model, num_classes=2)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    trained_model = train_model_with_early_stopping(model, dataloaders, criterion, optimizer, scheduler, early_stopping,
                                                    num_epochs=num_epochs, device=device)

    torch.save(trained_model.state_dict(), 'trained_resnet101.pth')


if __name__ == "__main__":
    main()








