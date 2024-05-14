import os
import pickle
from utils.data import DataProcessor
import numpy as np
from utils import logger
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('runs/experiment_name')

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        # 增加每个分支的深度和复杂度，使用层归一化
        self.text_branch = nn.Sequential(
            nn.Linear(100, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.audio_branch = nn.Sequential(
            nn.Linear(73, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.video_branch = nn.Sequential(
            nn.Linear(100, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        # 更复杂的合并策略，使用Dropout防止过拟合
        self.merge_layer = nn.Sequential(
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(192, 2)
        )

    def forward(self, text, audio, video, mask):
        # 应用掩码以处理序列中的padding
        mask = mask.unsqueeze(-1)
        text *= mask
        audio *= mask
        video *= mask
        
        text_features = self.text_branch(text)
        audio_features = self.audio_branch(audio)
        video_features = self.video_branch(video)

        concatenated = torch.cat((text_features, audio_features, video_features), dim=-1)
        output = self.merge_layer(concatenated)
        return output

#------
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for texts, audios, videos, labels, masks in train_loader:
        # texts: torch.Size([32, 63, 100])
        # audios: torch.Size([32, 63, 73])
        # videos: torch.Size([32, 63, 100])
        # labels: torch.Size([32, 63, 2])
        # masks: torch.Size([32, 63])
        texts, audios, videos, labels, masks = (d.to(device) for d in (texts, audios, videos, labels, masks))
        optimizer.zero_grad()
        outputs = model(texts, audios, videos, masks)
        #labels = torch.argmax(labels, dim=-1)
#  outputs输出的是每个batch每一句话（时间步）的正、负logits，labels输出的是每个batch每一句话（时间步）的正、负onehot标签，因为有两个类，所以是二维
        labels = labels.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
# 测试函数
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, audios, videos, labels, masks in test_loader:
            texts, audios, videos, labels, masks = texts.to(device), audios.to(device), videos.to(device), labels.to(device), masks.to(device)
            
            # Model prediction
            outputs = model(texts, audios, videos, masks)
            
            # Ensuring mask is correctly sized for applying to every element
            masks_expanded = masks.unsqueeze(-1).expand_as(labels)
            
            # Compute the loss only on the masked elements
            loss = criterion(outputs * masks_expanded, labels * masks_expanded)
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = torch.sigmoid(outputs) > 0.5  # Apply threshold to obtain binary predictions
            correct += ((pred == labels.byte()) * masks_expanded.byte()).sum().item()
            total += masks_expanded.sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0

    return avg_loss, accuracy
def create_data_loader(train_text, train_audio, train_video, train_label, train_mask, batch_size=32, shuffle=True):
    # 转换数据为 torch.Tensor
    train_text_tensor = torch.tensor(train_text, dtype=torch.float32)
    train_audio_tensor = torch.tensor(train_audio, dtype=torch.float32)
    train_video_tensor = torch.tensor(train_video, dtype=torch.float32)
    train_label_tensor = torch.tensor(train_label, dtype=torch.long)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.float32)

    # 创建数据集
    train_dataset = TensorDataset(train_text_tensor, train_audio_tensor, train_video_tensor, train_label_tensor, train_mask_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader


if __name__ == "__main__":
    #Batch size is 62, 63 is utterance(the actual figure is defined by len list), 100is vector(maximum)
    # train_text: (62, 63, 100)
    # train_label: (62, 63)
    # test_text: (31, 63, 100)
    # test_label: (31, 63)
    # max_utt_len: Not a tensor, the value is63
    # train_len: Not a tensor, the value is[14, 30, 24, 12, 14, 19, 39, 23, 26, 25, 33, 22, 30, 26, 29, 34, 22, 29, 18, 24, 25, 13, 12, 18, 14, 15, 17, 55, 32, 22, 11, 9, 28, 30, 21, 34, 25, 15, 33, 29, 19, 43, 15, 19, 30, 15, 14, 27, 31, 30, 10, 24, 14, 16, 21, 22, 18, 16, 30, 24, 23, 35]
    # test_len: Not a tensor, the value is[13, 25, 30, 63, 30, 25, 12, 31, 31, 31, 44, 31, 18, 21, 18, 39, 16, 20, 13, 32, 16, 22, 9, 34, 16, 24, 18, 16, 20, 12, 22]
    # train_audio: (62, 63, 73)
    # test_audio: (31, 63, 73)
    # train_video: (62, 63, 100)
    # test_video: (31, 63, 100)
    (
    train_text, train_label, test_text, test_label, max_utt_len,
    train_len, test_len, train_audio, test_audio,
    train_video, test_video
) = DataProcessor.load_data()
    train_label, test_label = DataProcessor.create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))
    train_mask, test_mask = DataProcessor.create_mask(train_text, test_text, train_len, test_len)
    #Divide Dataset for train and dev. It is not neccessary.
    # # 划分训练集和开发集 
    # train_text, dev_text = DataProcessor.split_dataset(train_text)
    # train_audio, dev_audio = DataProcessor.split_dataset(train_audio)
    # train_video, dev_video = DataProcessor.split_dataset(train_video)
    # train_label, dev_label = DataProcessor.split_dataset(train_label)
    # train_mask, dev_mask = DataProcessor.split_dataset(train_mask)

    print("Data loaded and processed.")
 # for mode in ['MMMU_BA', 'MMUU_SA', 'MU_SA', 'None']:
    #     train(mode)
    mode = "sdq"
    #train(mode)
    train_loader = create_data_loader(
    train_text=train_text,
    train_audio=train_audio,
    train_video=train_video,
    train_label=train_label,
    train_mask=train_mask
)
    test_loader = create_data_loader(
    train_text=test_text,
    train_audio=test_audio,
    train_video=test_video,
    train_label=test_label,
    train_mask=test_mask,
    shuffle=False  # 在测试时通常不需要打乱数据
)

    '''train_text: (62, 63, 100)
    train_audio: (62, 63, 73)
    train_video: (62, 63, 100)
    train_label: (62, 63, 2)
    train_mask: (62, 63)
    test_text: (31, 63, 100)
    test_audio: (31, 63, 73)
    test_video: (31, 63, 100)
    test_label: (31, 63, 2)
    test_mask: (31, 63)'''
    model = MultiModalModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
     #训练模型
    epochs = 1000  # 可以根据需要调整
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
                # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
    writer.close()  