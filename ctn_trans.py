import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch import optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=1):
        super(TCNBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                          stride=1, padding=padding, dilation=dilation))
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.downsample is not None:
            x = self.downsample(x)
        return out + x


class EEGNetTCNTransformer(nn.Module):
    def __init__(self, in_channels, num_classes, num_layers, hidden_dim, dropout_rate, num_tcn_blocks=3, kernel_size=3):
        super(EEGNetTCNTransformer, self).__init__()
        self.tcn_blocks = nn.ModuleList()
        tcn_channels = [64, 128, 256]  # 设定每个TCN块的输出通道数
        
        prev_channels = in_channels
        dilation_size = 1
        for idx in range(num_tcn_blocks):
            self.tcn_blocks.append(
                TCNBlock(prev_channels, tcn_channels[idx], kernel_size=kernel_size, dilation=dilation_size, padding=(kernel_size-1)*dilation_size//2)
            )
            prev_channels = tcn_channels[idx]
            dilation_size *= 2  # Increase dilation

        self.dropout = nn.Dropout(dropout_rate)  # 添加dropout层
        self.transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=num_layers, dim_feedforward=hidden_dim)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, src):
        out = src
        for block in self.tcn_blocks:
            out = block(out)
        
        out = out.permute(0, 2, 1)  # 调整维度顺序以适配Transformer
        out = self.dropout(out)  # 应用dropout层
        out = self.transformer(out, out)  # 将out作为源数据和目标数据传递给Transformer
        out = out.permute(0, 2, 1)  # 还原维度顺序
        out = torch.mean(out, dim=2)  # Global average pooling
        out = self.fc(out)
        return out


class EEGDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx, :-1].values.astype(float)
        label = self.data.iloc[idx, -1]
        # 调整数据维度顺序以匹配Conv1d的输入要求
        
        features = features.reshape(1, -1)  # 从 [sequence_length, in_channels] 调整为 [in_channels, sequence_length]
        return torch.Tensor(features), label


def train_model(model, dataloader, criterion, optimizer, device, num_epochs, model_save_path):
    model.train()  # 设置模型为训练模式

    for epoch in range(num_epochs):
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        # 使用tqdm显示进度条
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in progress_bar:
            # 获取输入数据并传送到设备
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device).long()  # 将标签转换为Long类型
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播、反向传播和优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 更新进度条的描述
            progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/(i+1):.4f}')
        
        # 每个epoch结束后计算并打印评价指标
        acc = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        
        print(f'Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}')

        # 每5个epoch保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'{model_save_path}_epoch_{epoch+1}.pth')
            print(f'Model saved at epoch: {epoch+1}')

    # 最后保存训练完成的模型
    torch.save(model.state_dict(), f'{model_save_path}_final.pth')
    print('Finished Training, final model saved.')


def main():
    # 定义超参数
    in_channels = 1
    num_classes = 2
    num_layers = 6
    hidden_dim = 2048
    batch_size = 256
    num_epochs = 20
    learning_rate = 0.000005
    save_interval = 5  # 每隔多少个epoch保存一次模型
    dropout_rate = 0.5  # 设置dropout概率

    # 创建数据加载器
    dataset = EEGDataset('chbmit_preprocessed_data.csv')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EEGNetTCNTransformer(in_channels=in_channels, num_classes=num_classes, num_layers=num_layers, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

    model.train()  # 设置模型为训练模式

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=0.00001)  # 使用Adam优化器

    # 定义训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型传送到合适的设备

    num_epochs = 50  # 定义训练的轮数

    train_model(model, dataloader, criterion, optimizer, device, num_epochs, "saved_model")

if __name__ == '__main__':
    main()
