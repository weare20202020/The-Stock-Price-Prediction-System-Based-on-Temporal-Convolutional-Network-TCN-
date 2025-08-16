# 📈 Stock Price Prediction System Based on Temporal Convolutional Network (TCN)
# 基于时间卷积网络(TCN)的股票价格预测系统

## 🎯 Project Overview | 项目概述

This project implements a high-precision stock price prediction system based on **Temporal Convolutional Network (TCN)**, specifically designed for Shanghai Composite Index prediction and analysis. Through deep learning technology and advanced time series modeling methods, it achieves **96.24% ultra-high fitting accuracy**.

本项目实现了一个基于**时间卷积网络(Temporal Convolutional Network, TCN)**的高精度股票价格预测系统，专门针对上证指数进行预测分析。通过深度学习技术和先进的时间序列建模方法，实现了**96.24%的超高拟合精度**。

## 🏆 Core Advantages & Highlights | 核心优势与亮点

### 📊 Excellent Prediction Performance | 卓越的预测性能
- **Coefficient of Determination (R²): 0.9624** - Achieving 96.24% fitting accuracy | **决定系数 (R²): 0.9624** - 达到96.24%的拟合精度
- **Root Mean Square Error (RMSE): 43.60** - Extremely low prediction error | **均方根误差 (RMSE): 43.60** - 极低的预测误差
- **Mean Square Error (MSE): 1900.94** - Stable prediction performance | **均方误差 (MSE): 1900.94** - 稳定的预测表现

### 🧠 Advanced Model Architecture | 先进的模型架构

#### 1. Temporal Convolutional Network (TCN) Core Technology | 时间卷积网络(TCN)核心技术
- **Residual Connections**: ResidualBlock design effectively solves gradient vanishing problem | **残差连接**: 采用ResidualBlock设计，有效解决梯度消失问题
- **Causal Convolution**: Ensures predictions depend only on historical data, avoiding future information leakage | **因果卷积**: 确保预测只依赖历史数据，避免未来信息泄露
- **Dilated Convolution**: Exponential dilation rates (2^i) expand receptive field, capturing long-term dependencies | **膨胀卷积**: 通过指数级膨胀率(2^i)扩大感受野，捕获长期依赖关系
- **Multi-level Feature Extraction**: Progressive feature learning from local to global pattern recognition | **多层级特征提取**: 渐进式特征学习，从局部到全局模式识别

#### 2. Intelligent Data Processing | 智能化数据处理
- **Multi-dimensional Feature Engineering**: Integrates 6 core features including price, volume, and change percentage | **多维特征工程**: 整合价格、成交量、涨跌幅等6个核心特征
- **Adaptive Data Preprocessing**: Intelligent processing of volume unit (B/M/K) conversion | **自适应数据预处理**: 智能处理成交量单位(B/M/K)转换
- **Data Augmentation Technology**: Enhances model sensitivity by subtracting baseline value (2000) | **数据增强技术**: 通过减去基准值(2000)提升模型敏感度
- **Sequential Modeling**: Uses sliding window technique to construct time series samples | **序列化建模**: 采用滑动窗口技术构建时间序列样本

### 🔧 Engineering Advantages | 工程化优势

#### 1. Robust Training Strategy | 鲁棒的训练策略
- **Huber Loss Function**: SmoothL1Loss more robust to outliers | **Huber损失函数**: 对异常值更加鲁棒的SmoothL1Loss
- **Adaptive Learning Rate**: ReduceLROnPlateau scheduler dynamically adjusts learning rate | **自适应学习率**: ReduceLROnPlateau调度器动态调整学习率
- **Regularization Techniques**: L2 weight decay and Dropout prevent overfitting | **正则化技术**: L2权重衰减和Dropout防止过拟合
- **Early Stopping Mechanism**: Intelligent training termination based on validation loss | **早停机制**: 基于验证损失的智能训练终止

#### 2. Comprehensive Evaluation System | 完整的评估体系
- **Multi-metric Evaluation**: Comprehensive performance assessment with MSE, RMSE, R² | **多指标评估**: MSE、RMSE、R²全方位性能评估
- **Visualization Analysis**: Training loss curves and prediction result comparison charts | **可视化分析**: 训练损失曲线和预测结果对比图
- **Time Series Visualization**: Timeline comparison of actual vs predicted values | **时间序列可视化**: 实际值与预测值的时间轴对比

## 🚀 Technical Features | 技术特性

### Core Algorithm | 核心算法
- **Deep Learning Framework**: PyTorch | **深度学习框架**: PyTorch
- **Network Architecture**: Temporal Convolutional Network (TCN) | **网络架构**: 时间卷积网络(TCN)
- **Optimization Algorithm**: Adam Optimizer | **优化算法**: Adam优化器
- **Loss Function**: Smooth L1 Loss (Huber Loss) | **损失函数**: Smooth L1 Loss (Huber Loss)

### Data Processing | 数据处理
- **Feature Normalization**: MinMaxScaler normalization | **特征标准化**: MinMaxScaler归一化
- **Sequence Length**: 30 time steps of historical data | **序列长度**: 30个时间步的历史数据
- **Train/Test Split**: 80%/20% ratio | **训练/测试分割**: 80%/20%比例
- **Batch Processing**: 32 samples batch size | **批处理**: 32样本批次大小

### Model Configuration | 模型配置
```python
# Network Parameters | 网络参数
input_size = 6          # Input feature dimension | 输入特征维度
output_size = 1         # Output dimension (closing price) | 输出维度(收盘价)
num_channels = [25, 25, 25, 25]  # Channel numbers for each layer | 各层通道数
kernel_size = 3         # Convolution kernel size | 卷积核大小
dropout = 0.2          # Dropout ratio | Dropout比例
```

## 📈 Experimental Results | 实验结果

### Performance Metrics | 性能指标
| Metric 指标 | Value 数值 | Description 说明 |
|------|------|------|
| R² Score | **0.9624** | 96.24% fitting accuracy \| 96.24%的拟合精度 |
| RMSE | **43.60** | Root Mean Square Error \| 均方根误差 |
| MSE | **1900.94** | Mean Square Error \| 均方误差 |

### Visualization Results | 可视化结果
- 📊 **Training Loss Curve**: Shows model convergence process | **训练损失曲线**: 展示模型收敛过程
- 📈 **Prediction Result Comparison**: Actual vs predicted time series chart | **预测结果对比**: 实际值vs预测值时间序列图
- 🎯 **High-precision Fitting**: Prediction curve highly matches actual trends | **高精度拟合**: 预测曲线与实际走势高度吻合

## 🛠️ Environment Requirements | 环境要求

```bash
# Core Dependencies | 核心依赖
pip install torch torchvision
pip install pandas numpy matplotlib
pip install scikit-learn

# Optional Technical Indicator Library | 可选技术指标库
pip install TA-Lib  # Technical Analysis Indicators | 技术分析指标
```

## 🚀 Quick Start | 快速开始

1. **Data Preparation | 数据准备**
   ```bash
   # Ensure data file exists | 确保数据文件存在
   Shanghai Composite Historical Data.csv
   ```

2. **Run Prediction | 运行预测**
   ```bash
   python stock_predictionTCN_backup\ copy.py
   ```

3. **View Results | 查看结果**
   - Console output performance metrics | 控制台输出性能指标
   - Generate `prediction_result.png` visualization chart | 生成`prediction_result.png`可视化图表
   - Save trained model weights | 保存训练好的模型权重

## 📁 Project Structure | 项目结构

```
├── stock_predictionTCN_backup copy.py  # Main program file | 主程序文件
├── Shanghai Composite Historical Data.csv  # Historical data | 历史数据
├── prediction_result.png               # Prediction result chart | 预测结果图
├── best_tcn_model.pth                 # Best model weights | 最佳模型权重
└── README.md                          # Project documentation | 项目说明
```

## 🔬 Technical Innovations | 技术创新点

1. **TCN Application**: First application of TCN to Chinese stock market prediction with remarkable results | **时间卷积网络应用**: 首次将TCN应用于中国股市预测，效果显著
2. **Multi-scale Feature Fusion**: Comprehensive modeling combining price, volume, and technical indicators | **多尺度特征融合**: 结合价格、成交量、技术指标的综合建模
3. **Causal Convolution Design**: Strictly ensures causality constraints in time series | **因果卷积设计**: 严格保证时间序列的因果性约束
4. **Residual Learning Mechanism**: Effectively alleviates gradient vanishing in deep networks | **残差学习机制**: 有效缓解深度网络的梯度消失问题
5. **Adaptive Training Strategy**: Dynamic learning rate adjustment and early stopping mechanism | **自适应训练策略**: 动态学习率调整和早停机制

## 📊 Application Scenarios | 应用场景

- 📈 **Quantitative Trading**: Provides price prediction signals for algorithmic trading | **量化交易**: 为算法交易提供价格预测信号
- 🏦 **Risk Management**: Assists financial institutions in risk assessment | **风险管理**: 辅助金融机构进行风险评估
- 📊 **Investment Decision**: Provides data-driven decision support for investors | **投资决策**: 为投资者提供数据驱动的决策支持
- 🔬 **Academic Research**: Time series prediction and deep learning research | **学术研究**: 时间序列预测和深度学习研究

## 🎯 Future Optimization Directions | 未来优化方向

- [ ] Integrate more technical indicator features | 集成更多技术指标特征
- [ ] Implement multi-step prediction functionality | 实现多步预测功能
- [ ] Add attention mechanism | 添加注意力机制
- [ ] Support multi-stock parallel prediction | 支持多股票并行预测
- [ ] Real-time data stream processing | 实时数据流处理

---

**Project Highlight | 项目亮点**: Through advanced Temporal Convolutional Network technology, achieved **96.24% ultra-high precision** stock price prediction, providing strong technical support for financial quantitative analysis. | 通过先进的时间卷积网络技术，实现了**96.24%超高精度**的股票价格预测，为金融量化分析提供了强有力的技术支撑。