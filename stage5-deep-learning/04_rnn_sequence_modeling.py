"""
循环神经网络序列建模
学习目标：掌握RNN、LSTM、GRU的原理和在序列建模中的应用
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import math
import warnings
warnings.filterwarnings('ignore')

print("=== 循环神经网络序列建模 ===\n")

# 设备检查
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. RNN基础理论
print("1. RNN基础理论")
print("RNN核心思想：")
print("- 处理序列数据，具有记忆能力")
print("- 隐藏状态在时间步之间传递信息")
print("- 可以处理变长序列")
print("- 存在梯度消失/爆炸问题")

# 2. 简单RNN实现
print("\n2. 简单RNN实现")

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN前向传播
        out, _ = self.rnn(x, h0)
        
        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        
        return out

# 3. LSTM实现
print("\n3. LSTM实现")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 应用dropout
        out = self.dropout(out[:, -1, :])
        
        # 输出层
        out = self.fc(out)
        
        return out

# 4. GRU实现
print("\n4. GRU实现")

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU前向传播
        out, _ = self.gru(x, h0)
        
        # 应用dropout
        out = self.dropout(out[:, -1, :])
        
        # 输出层
        out = self.fc(out)
        
        return out

# 5. 生成时间序列数据
print("\n5. 生成时间序列数据")

def generate_sine_wave(seq_length=1000, num_features=1):
    """生成正弦波时间序列数据"""
    x = np.linspace(0, 4*np.pi, seq_length)
    
    if num_features == 1:
        data = np.sin(x) + 0.1 * np.random.randn(seq_length)
    else:
        # 多变量时间序列
        data = np.zeros((seq_length, num_features))
        for i in range(num_features):
            phase = i * 2 * np.pi / num_features
            data[:, i] = np.sin(x + phase) + 0.1 * np.random.randn(seq_length)
    
    return data

def generate_stock_like_data(seq_length=1000):
    """生成类似股票价格的时间序列数据"""
    # 使用几何布朗运动
    dt = 1/252  # 一年252个交易日
    mu = 0.1    # 年化收益率
    sigma = 0.2 # 年化波动率
    
    price = [100]  # 初始价格
    for _ in range(seq_length - 1):
        dS = price[-1] * (mu * dt + sigma * np.sqrt(dt) * np.random.randn())
        price.append(price[-1] + dS)
    
    return np.array(price)

# 生成数据
sine_data = generate_sine_wave(1000, 1)
stock_data = generate_stock_like_data(1000)
multi_sine_data = generate_sine_wave(1000, 3)

print(f"正弦波数据形状: {sine_data.shape}")
print(f"股价数据形状: {stock_data.shape}")
print(f"多变量正弦波数据形状: {multi_sine_data.shape}")

# 6. 序列数据集类
print("\n6. 序列数据集类")

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, prediction_length=1):
        self.data = data
        self.seq_length = seq_length
        self.prediction_length = prediction_length
        
        # 标准化数据
        self.scaler = MinMaxScaler()
        if len(data.shape) == 1:
            self.data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        else:
            self.data = self.scaler.fit_transform(data)
    
    def __len__(self):
        return len(self.data) - self.seq_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        # 获取输入序列
        if len(self.data.shape) == 1:
            x = self.data[idx:idx + self.seq_length]
            y = self.data[idx + self.seq_length:idx + self.seq_length + self.prediction_length]
            x = torch.FloatTensor(x).unsqueeze(-1)  # 添加特征维度
        else:
            x = self.data[idx:idx + self.seq_length]
            y = self.data[idx + self.seq_length:idx + self.seq_length + self.prediction_length]
            x = torch.FloatTensor(x)
        
        y = torch.FloatTensor(y)
        
        return x, y

# 7. 创建数据加载器
print("\n7. 创建数据加载器")

seq_length = 50
batch_size = 32

# 正弦波数据
sine_dataset = TimeSeriesDataset(sine_data, seq_length)
sine_train_size = int(0.8 * len(sine_dataset))
sine_test_size = len(sine_dataset) - sine_train_size
sine_train_dataset, sine_test_dataset = torch.utils.data.random_split(
    sine_dataset, [sine_train_size, sine_test_size])

sine_train_loader = DataLoader(sine_train_dataset, batch_size=batch_size, shuffle=True)
sine_test_loader = DataLoader(sine_test_dataset, batch_size=batch_size, shuffle=False)

# 股价数据
stock_dataset = TimeSeriesDataset(stock_data, seq_length)
stock_train_size = int(0.8 * len(stock_dataset))
stock_test_size = len(stock_dataset) - stock_train_size
stock_train_dataset, stock_test_dataset = torch.utils.data.random_split(
    stock_dataset, [stock_train_size, stock_test_size])

stock_train_loader = DataLoader(stock_train_dataset, batch_size=batch_size, shuffle=True)
stock_test_loader = DataLoader(stock_test_dataset, batch_size=batch_size, shuffle=False)

print(f"正弦波训练集大小: {sine_train_size}")
print(f"正弦波测试集大小: {sine_test_size}")
print(f"股价训练集大小: {stock_train_size}")
print(f"股价测试集大小: {stock_test_size}")

# 8. 训练函数
print("\n8. 训练函数")

def train_model(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 测试阶段
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    return train_losses, test_losses

# 9. 模型比较实验
print("\n9. 模型比较实验")

# 创建不同的模型
models = {
    'Simple RNN': SimpleRNN(input_size=1, hidden_size=64, output_size=1),
    'LSTM': LSTMModel(input_size=1, hidden_size=64, output_size=1, num_layers=2),
    'GRU': GRUModel(input_size=1, hidden_size=64, output_size=1, num_layers=2)
}

# 在正弦波数据上训练和测试
sine_results = {}
print("在正弦波数据上训练模型...")

for name, model in models.items():
    print(f"\n训练 {name}...")
    model = model.to(device)
    train_losses, test_losses = train_model(model, sine_train_loader, sine_test_loader, 
                                           num_epochs=50, learning_rate=0.001)
    sine_results[name] = {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_test_loss': test_losses[-1]
    }

# 10. 序列到序列模型
print("\n10. 序列到序列模型")

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_seq_len, num_layers=2):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        
        # 编码器
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 解码器
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, target_seq=None):
        batch_size = x.size(0)
        
        # 编码
        _, (hidden, cell) = self.encoder(x)
        
        # 解码
        outputs = []
        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)
        
        for t in range(self.output_seq_len):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)
            outputs.append(output)
            
            # 使用预测值作为下一个输入 (teacher forcing在训练时可选)
            if target_seq is not None and np.random.random() < 0.5:  # teacher forcing
                decoder_input = target_seq[:, t:t+1]
            else:
                decoder_input = output
        
        return torch.cat(outputs, dim=1)

# 11. 注意力机制LSTM
print("\n11. 注意力机制LSTM")

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # 注意力机制
        self.attention = nn.Linear(hidden_size, 1)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # 计算注意力权重
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_len, 1)
        
        # 加权求和
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)
        
        # 输出
        output = self.fc(context_vector)
        
        return output, attention_weights

# 12. 双向LSTM
print("\n12. 双向LSTM")

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM层
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=0.2, bidirectional=True)
        
        # 输出层 (隐藏大小要乘以2，因为是双向的)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        # 双向LSTM前向传播
        lstm_out, _ = self.bilstm(x)
        
        # 使用最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])
        
        return output

# 13. 测试高级模型
print("\n13. 测试高级模型")

# 创建高级模型
advanced_models = {
    'Attention LSTM': AttentionLSTM(input_size=1, hidden_size=64, output_size=1),
    'BiLSTM': BiLSTM(input_size=1, hidden_size=64, output_size=1)
}

# 训练注意力LSTM
attention_model = advanced_models['Attention LSTM'].to(device)
print("训练注意力LSTM...")
att_train_losses, att_test_losses = train_model(attention_model, sine_train_loader, 
                                               sine_test_loader, num_epochs=50)

# 14. 预测和可视化
print("\n14. 预测和可视化")

def make_predictions(model, test_loader, scaler):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            if isinstance(model, AttentionLSTM):
                outputs, _ = model(batch_x)
            else:
                outputs = model(batch_x)
            
            # 反标准化
            if hasattr(scaler, 'inverse_transform'):
                outputs_np = scaler.inverse_transform(outputs.cpu().numpy())
                batch_y_np = scaler.inverse_transform(batch_y.cpu().numpy())
            else:
                outputs_np = outputs.cpu().numpy()
                batch_y_np = batch_y.cpu().numpy()
            
            predictions.extend(outputs_np.flatten())
            actuals.extend(batch_y_np.flatten())
    
    return np.array(predictions), np.array(actuals)

# 获取最佳模型的预测结果
best_model_name = min(sine_results.keys(), key=lambda x: sine_results[x]['final_test_loss'])
best_model = sine_results[best_model_name]['model']

predictions, actuals = make_predictions(best_model, sine_test_loader, sine_dataset.scaler)

# 计算评估指标
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mse)

print(f"最佳模型 ({best_model_name}) 评估结果:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

# 15. 可视化结果
print("\n15. 可视化结果")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 15.1 训练损失对比
for name, result in sine_results.items():
    axes[0, 0].plot(result['train_losses'], label=f'{name} (train)')
    axes[0, 0].plot(result['test_losses'], label=f'{name} (test)', linestyle='--')

axes[0, 0].set_title('训练和测试损失')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 15.2 模型性能对比
model_names = list(sine_results.keys())
final_losses = [sine_results[name]['final_test_loss'] for name in model_names]

axes[0, 1].bar(model_names, final_losses, alpha=0.7)
axes[0, 1].set_title('模型最终测试损失对比')
axes[0, 1].set_ylabel('Test Loss')
axes[0, 1].tick_params(axis='x', rotation=45)

# 15.3 预测结果对比
sample_length = min(200, len(predictions))
axes[0, 2].plot(actuals[:sample_length], label='实际值', alpha=0.7)
axes[0, 2].plot(predictions[:sample_length], label='预测值', alpha=0.7)
axes[0, 2].set_title(f'预测结果对比 ({best_model_name})')
axes[0, 2].set_xlabel('时间步')
axes[0, 2].set_ylabel('值')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 15.4 原始时间序列数据
axes[1, 0].plot(sine_data[:500])
axes[1, 0].set_title('原始正弦波数据')
axes[1, 0].set_xlabel('时间步')
axes[1, 0].set_ylabel('值')
axes[1, 0].grid(True, alpha=0.3)

# 15.5 股价数据
axes[1, 1].plot(stock_data[:500])
axes[1, 1].set_title('模拟股价数据')
axes[1, 1].set_xlabel('时间步')
axes[1, 1].set_ylabel('价格')
axes[1, 1].grid(True, alpha=0.3)

# 15.6 预测误差分布
errors = predictions - actuals
axes[1, 2].hist(errors, bins=30, alpha=0.7, edgecolor='black')
axes[1, 2].set_title('预测误差分布')
axes[1, 2].set_xlabel('误差')
axes[1, 2].set_ylabel('频数')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/rnn_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 16. 注意力权重可视化
print("\n16. 注意力权重可视化")

if 'Attention LSTM' in locals():
    # 获取一个样本的注意力权重
    sample_batch = next(iter(sine_test_loader))
    sample_x, sample_y = sample_batch[0][:1].to(device), sample_batch[1][:1].to(device)
    
    with torch.no_grad():
        output, attention_weights = attention_model(sample_x)
    
    # 可视化注意力权重
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(sample_x[0, :, 0].cpu().numpy())
    plt.title('输入序列')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(attention_weights[0, :, 0].cpu().numpy())
    plt.title('注意力权重')
    plt.xlabel('时间步')
    plt.ylabel('权重')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/attention_weights.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("注意力权重可视化已保存")

# 17. 模型保存
print("\n17. 模型保存")

# 保存最佳模型
torch.save({
    'model_state_dict': best_model.state_dict(),
    'model_name': best_model_name,
    'seq_length': seq_length,
    'scaler': sine_dataset.scaler
}, '/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/best_rnn_model.pth')

print(f"最佳模型 ({best_model_name}) 已保存")

print("\n=== RNN序列建模总结 ===")
print("✅ 理解RNN、LSTM、GRU的原理和特点")
print("✅ 实现多种循环神经网络架构")
print("✅ 序列数据的预处理和数据集构建")
print("✅ 序列到序列模型实现")
print("✅ 注意力机制在RNN中的应用")
print("✅ 双向LSTM的实现和应用")
print("✅ 模型训练、评估和可视化")

print("\n关键技术点:")
print("1. 梯度裁剪防止梯度爆炸")
print("2. 序列数据的标准化处理")
print("3. Teacher Forcing训练策略")
print("4. 注意力机制提升模型性能")
print("5. 多步预测和序列到序列映射")

print("\n=== 练习任务 ===")
print("1. 实现Transformer架构")
print("2. 尝试不同的注意力机制")
print("3. 实现变长序列的处理")
print("4. 研究序列标注任务")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现文本生成模型")
print("2. 构建语音识别系统")
print("3. 实现机器翻译模型")
print("4. 研究时间序列异常检测")
print("5. 实现多变量时间序列预测")