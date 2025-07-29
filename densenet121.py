import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np


def main():
    # 设置随机种子（保证可重复性）
    torch.manual_seed(42)
    np.random.seed(42)

    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 数据集路径
    data_dir = r'C:\Users\chy04\PycharmProjects\PythonProject\densenet121'
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    }

    # 数据加载器
    batch_size = 16
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)
    }

    # 类别名称（3类）
    class_names = image_datasets['train'].classes
    print(f"Class names: {class_names}")

    # 检查CUDA是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练的DenseNet121
    model = models.densenet121(pretrained=True)

    # 冻结所有层（只训练最后的分类层）
    for param in model.parameters():
        param.requires_grad = False

    # 修改最后的分类层（输出3类）
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))  # 输出3类
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 训练函数（修改为记录训练和验证指标）
    def train_model(model, criterion, optimizer, num_epochs=23):
        # 记录训练过程指标
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # 训练阶段
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets['train'])
            epoch_acc = running_corrects.double() / len(image_datasets['train'])
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.cpu().numpy())

            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 测试阶段（每个epoch后评估）
            model.eval()
            test_loss = 0.0
            test_corrects = 0

            with torch.no_grad():
                for inputs, labels in dataloaders['test']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item() * inputs.size(0)
                    test_corrects += torch.sum(preds == labels.data)

            test_epoch_loss = test_loss / len(image_datasets['test'])
            test_epoch_acc = test_corrects.double() / len(image_datasets['test'])
            history['test_loss'].append(test_epoch_loss)
            history['test_acc'].append(test_epoch_acc.cpu().numpy())

            print(f'Test Loss: {test_epoch_loss:.4f} Acc: {test_epoch_acc:.4f}\n')

        return model, history

    # 训练模型并记录历史数据
    print("Training...")
    model, history = train_model(model, criterion, optimizer, num_epochs=23)

    # 绘制训练曲线
    def plot_training_curves(history):
        plt.figure(figsize=(12, 5))

        # Loss曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['test_loss'], label='Test Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['test_acc'], label='Test Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # 保存图像
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'results/training_curves_{timestamp}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Training curves saved to: {save_path}")

    # 绘制并保存训练曲线
    plot_training_curves(history)

    # 测试模型并绘制混淆矩阵
    def evaluate_model(model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 分类报告
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(cm)

        # 可视化并保存混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'results/confusion_matrix_{timestamp}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Confusion matrix saved to: {save_path}")

    # 在测试集上评估
    print("\nEvaluating on test set...")
    evaluate_model(model, dataloaders['test'])

    # 保存模型
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/shrimp_3class_model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()