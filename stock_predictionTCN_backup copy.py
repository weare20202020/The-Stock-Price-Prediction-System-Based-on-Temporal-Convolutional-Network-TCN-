import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        # 计算padding，确保输出序列长度与输入相同
        # 对于因果卷积，padding = (kernel_size-1) * dilation
        self.padding = (kernel_size-1) * dilation
        
        # 使用padding='same'模式，确保输出大小与输入相同
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # 如果输入输出通道数不同，则需要1x1卷积进行调整
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        residual = x
        # 应用第一个卷积
        out = self.conv1(x)
        # 由于padding='same'不能完全保证输出大小，手动裁剪确保大小匹配
        # 裁剪掉右侧多余的部分
        out = out[:, :, :x.size(2)]
        out = self.relu1(out)
        out = self.dropout(out)
        
        # 应用第二个卷积
        out = self.conv2(out)
        # 再次裁剪确保大小匹配
        out = out[:, :, :x.size(2)]
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        # 现在out和residual的形状应该完全一致
        out += residual
        return self.relu2(out)

# 定义TCN模型
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dilation_size))
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch, input_size, seq_len]
        out = self.network(x)
        # 取最后一个时间步的输出
        out = out[:, :, -1]
        out = self.dropout(out)
        out = self.linear(out)
        return out

# 数据预处理函数
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]  # 预测收盘价
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 加载数据
df = pd.read_csv('Shanghai Composite Historical Data.csv')

# 数据预处理
# 将日期转换为datetime格式
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# 按日期升序排序（因为原数据是倒序的）
df = df.sort_values('Date').reset_index(drop=True)

# 提取特征（新增Change %）
features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']

# 处理成交量数据，去除B后缀并转换为数值（B表示十亿）
def convert_volume(vol_str):
    # 如果已经是数字类型，直接返回
    if isinstance(vol_str, (int, float)):
        return vol_str
    # 如果是字符串，进行转换
    if isinstance(vol_str, str):
        if 'B' in vol_str:
            return float(vol_str.replace('B', '')) * 1000000000
        elif 'M' in vol_str:
            return float(vol_str.replace('M', '')) * 1000000
        elif 'K' in vol_str:
            return float(vol_str.replace('K', '')) * 1000
        else:
            return float(vol_str)
    return vol_str

df['Vol.'] = df['Vol.'].apply(convert_volume)

# 处理价格数据，去除逗号
for col in ['Price', 'Open', 'High', 'Low']:
    df[col] = df[col].str.replace(',', '').astype(float)

# 数据预处理：减去2000增强灵敏度
for col in ['Price', 'Open', 'High', 'Low']:
    df[col] = df[col] - 2000

# 处理Change %数据，去除%符号并转换为数值
df['Change %'] = df['Change %'].str.replace('%', '').astype(float)

# 检查并处理缺失值
print(f"缺失值统计：")
print(df[features].isnull().sum())

# 删除包含缺失值的行
df = df.dropna(subset=features)
print(f"处理后数据形状: {df.shape}")

# 选择特征
data = df[features].values

# 先按时间划分训练集索引，避免数据泄露
split_idx = int(len(data) * 0.8)

# 数据归一化：仅用训练集拟合，再对全量数据变换
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data[:split_idx])
data_scaled = scaler.transform(data)

# 创建序列数据
# 当前参数
seq_length = 30  # 使用30天的数据预测下一天

# 优化建议3：增加历史数据长度
seq_length = 60  # 使用60天的数据，捕获更长期的模式
X, y = create_sequences(data_scaled, seq_length)

# 划分训练/测试的序列数量，确保不跨越时间边界
train_size = split_idx - seq_length
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train).transpose(1, 2)  # 转换为[batch, features, seq_len]格式
X_test = torch.FloatTensor(X_test).transpose(1, 2)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
input_size = len(features)  # 特征数量
output_size = 1  # 预测收盘价

# 优化后的参数
num_channels = [128, 256, 128, 64]  # 增加层数和通道数
kernel_size = 5  # 增大卷积核
dropout = 0.3  # 适当增加dropout防止过拟合

# 创建模型实例
model = TCN(input_size, output_size, num_channels, kernel_size, dropout)

# 定义损失函数和优化器
criterion = nn.SmoothL1Loss()  # 使用Huber损失
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# 添加学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# 训练参数
num_epochs = 100  # 增加训练轮数
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # 降低学习率，添加权重衰减

# 添加学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# 训练模型
num_epochs = 50
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # 在测试集上评估
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    # 在训练循环的每个epoch结束后添加
    scheduler.step(test_loss)

# 在测试集上进行预测
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(batch_y.squeeze().tolist())

# 反归一化预测结果
predictions = np.array(predictions).reshape(-1, 1)
actuals = np.array(actuals).reshape(-1, 1)

# 创建一个只有一个特征（收盘价）的数组用于反归一化
pred_container = np.zeros((len(predictions), data.shape[1]))
pred_container[:, 0] = predictions.flatten()

actual_container = np.zeros((len(actuals), data.shape[1]))
actual_container[:, 0] = actuals.flatten()

# 反归一化
predictions_rescaled = scaler.inverse_transform(pred_container)[:, 0]
actuals_rescaled = scaler.inverse_transform(actual_container)[:, 0]

# 恢复原始价格范围（加回2000）
predictions_rescaled = predictions_rescaled + 2000
actuals_rescaled = actuals_rescaled + 2000

# 计算评估指标
mse = mean_squared_error(actuals_rescaled, predictions_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(actuals_rescaled, predictions_rescaled)

print(f'均方误差 (MSE): {mse:.4f}')
print(f'均方根误差 (RMSE): {rmse:.4f}')
print(f'决定系数 (R²): {r2:.4f}')
print(f'拟合百分比: {r2*100:.2f}%')

# 可视化训练过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(test_losses, label='测试损失')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('训练和测试损失')
plt.legend()

# 可视化预测结果
plt.subplot(1, 2, 2)

# 获取测试集对应的日期
test_dates = df['Date'].iloc[train_size+seq_length:train_size+seq_length+len(predictions_rescaled)]

plt.figure(figsize=(14, 7))
plt.plot(test_dates, actuals_rescaled, label='实际值', color='blue')
plt.plot(test_dates, predictions_rescaled, label='预测值', color='red', linestyle='--')
plt.xlabel('日期')
plt.ylabel('上证指数')
plt.title(f'上证指数预测 (拟合度: {r2*100:.2f}%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# 保存图像
plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
plt.show()


# 在特征工程部分添加技术指标
# import talib  # 需要安装：pip install TA-Lib

# 添加技术指标特征
def add_technical_indicators(df):
    # 移动平均线
    df['MA5'] = df['Price'].rolling(window=5).mean()
    df['MA20'] = df['Price'].rolling(window=20).mean()
    
    # RSI
    df['RSI'] = talib.RSI(df['Price'].values, timeperiod=14)
    
    # MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Price'].values)
    
    # 布林带
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Price'].values)
    
    # 去除NaN值
    df = df.dropna()
    return df

# 更新特征列表
features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 
           'MA5', 'MA20', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower']