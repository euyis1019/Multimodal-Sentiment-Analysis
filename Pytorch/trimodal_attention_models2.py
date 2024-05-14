import os
import pickle
from utils.data import DataProcessor
import numpy as np
from utils import logger
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import visdom
vis = visdom.Visdom()
assert vis.check_connection()
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter('runs/experiment_name')

class BiModalAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        # x, y should have dimensions [batch, seq_len, features]
        m1 = torch.bmm(x, y.transpose(1, 2))
        m2 = torch.bmm(y, x.transpose(1, 2))

        n1 = self.softmax(m1)
        n2 = self.softmax(m2)

        o1 = torch.bmm(n1, y)
        o2 = torch.bmm(n2, x)

        a1 = o1 * x
        a2 = o2 * y

        return torch.cat((a1, a2), dim=-1)

class TriModalAttention(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.tanh = nn.Tanh()
        self.dense_tv = nn.Linear(feature_size * 2, 100)
        self.dense_ta = nn.Linear(feature_size * 2, 100)
        self.dense_av = nn.Linear(feature_size * 2, 100)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, v, t, a):
        # Concatenate and pass through dense layers
        Ftv = self.tanh(self.dense_tv(torch.cat((t, v), dim=2)))
        Fta = self.tanh(self.dense_ta(torch.cat((t, a), dim=2)))
        Fav = self.tanh(self.dense_av(torch.cat((a, v), dim=2)))

        # Compute attention scores
        c1 = torch.bmm(a, Ftv.transpose(1, 2))
        c2 = torch.bmm(v, Fta.transpose(1, 2))
        c3 = torch.bmm(t, Fav.transpose(1, 2))

        p1 = self.softmax(c1)
        p2 = self.softmax(c2)
        p3 = self.softmax(c3)

        t1 = torch.bmm(p1, a)
        t2 = torch.bmm(p2, v)
        t3 = torch.bmm(p3, t)

        Oatv = t1 * Ftv
        Ovta = t2 * Fta
        Otav = t3 * Fav

        return torch.cat((Oatv, Ovta, Otav), dim=2)

class SelfAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m = torch.bmm(x, x.transpose(1, 2))
        n = self.softmax(m)
        o = torch.bmm(n, x)
        a = o * x
        return a
    
class ResidualGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        if input_size == 73 :
            input_size = 100
        self.projection = nn.Linear(2 * hidden_size, input_size)  # To match the input dimension for residual connection

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.projection(out)
        return F.relu(x + out)  # Applying residual connection

class ResidualAttention(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.attention_module = attention_module

    def forward(self, *inputs):
        attention_output = self.attention_module(*inputs)

        # 获取输入和注意力输出的维度
        batch_size, seq_len, features = inputs[0].shape
        _, _, attention_features = attention_output.shape
        
        # 检查attention_output的最后一维是否是inputs[0]的整数倍
        if attention_features % features == 0:
            scale_factor = attention_features // features
            # 扩展inputs[0]
            expanded_input = inputs[0].unsqueeze(-1).expand(batch_size, seq_len, features, scale_factor).reshape(batch_size, seq_len, attention_features)
        else:
            raise ValueError("attention_output的最后一维不是inputs[0]的整数倍，无法对齐")

        # 执行加法操作
        output = expanded_input + attention_output
        return F.relu(output)

class MultiModalModel(nn.Module):
    def __init__(self, dropout_rate=0.7):
        super().__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(100, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.LayerNorm(100),
            nn.ReLU()
        )
        self.audio_branch = nn.Sequential(
            nn.Linear(73, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.LayerNorm(100),
            nn.ReLU()
        )
        self.video_branch = nn.Sequential(
            nn.Linear(100, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.LayerNorm(100),
            nn.ReLU()
        )
        self.rnn_text = ResidualGRU(input_size=100, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.rnn_audio = ResidualGRU(input_size=100, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.rnn_video = ResidualGRU(input_size=100, hidden_size=300, num_layers=2, dropout_rate=dropout_rate)
        self.early_fuse = nn.Sequential(
            nn.Linear(300, 50),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.bi_modal_attention = ResidualAttention(BiModalAttention(size=600))
        self.tri_modal_attention = ResidualAttention(TriModalAttention(feature_size=100))
        self.self_attention = ResidualAttention(SelfAttention(size=600))

        self.output_layer =nn.Sequential(
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(300, 50),
            nn.Linear(50, 2),
        )
    def forward(self, text, audio, video, mask):
        text_embeddings = self.text_branch(text)
        audio_embeddings = self.audio_branch(audio)
        video_embeddings = self.video_branch(video)
        text_output = self.rnn_text(text_embeddings)
        audio_output = self.rnn_audio(audio_embeddings)
        video_output = self.rnn_video(video_embeddings)
        earyly_output = self.early_fuse(torch.cat((text_output, audio_output, video_output), dim=2)) #将三个outpu张量沿着最后一维拼接，900

        # Example of applying attention; you can adjust as needed
        bi_modal_output = self.bi_modal_attention(text_output, audio_output)
        tri_modal_output = self.tri_modal_attention(text_output, audio_output, video_output)
        self_attention_output = self.self_attention(text_output)

        # Example of combining features
        combined_features = torch.cat((bi_modal_output, tri_modal_output, self_attention_output), dim=2)

        logits = self.output_layer(combined_features)   
        return logits
#------
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    total_loss = 0
    correct = 0
    total = 0

    for texts, audios, videos, labels, masks in train_loader:
        # Move data to the device
        texts, audios, videos, labels, masks = (d.to(device) for d in (texts, audios, videos, labels, masks))

        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        outputs = model(texts, audios, videos, masks)

        # Convert labels to float for loss calculation (if required by your loss function)
        labels = labels.float()

        # Compute the loss
        loss = criterion(outputs, labels)
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        masks_expanded = masks.unsqueeze(-1).expand_as(labels)

        total_loss += loss.item()  # Accumulate the loss

        # Calculate accuracy
        pred = torch.sigmoid(outputs) > 0.5  # Apply threshold to obtain binary predictions
        correct += ((pred == labels.byte()) * masks_expanded.byte()).sum().item()
        total += masks_expanded.sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0  # Calculate accuracy as a percentage

    return avg_loss, accuracy
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
def create_data_loader(train_text, train_audio, train_video, train_label, train_mask, batch_size=7, shuffle=True):
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
    epochs = 100  # 可以根据需要调整
    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
                # Log to TensorBoard
    # 使用Visdom记录
        vis.line(X=[epoch], Y=[train_loss], win='train_loss', update='append', opts=dict(title='Train Loss'))
        vis.line(X=[epoch], Y=[train_accuracy], win='train_accuracy', update='append', opts=dict(title='Train Accuracy'))
        vis.line(X=[epoch], Y=[test_loss], win='test_loss', update='append', opts=dict(title='Test Loss'))
        vis.line(X=[epoch], Y=[test_accuracy], win='test_accuracy', update='append', opts=dict(title='Test Accuracy'))
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
