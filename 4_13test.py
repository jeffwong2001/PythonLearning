import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 生成模拟流量数据 (Simulating Traffic Data)
# 假设特征：报文长度 (Packet Size) 和 持续时间 (Duration)
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)
X_scaled = StandardScaler().fit_transform(X)  # 归一化

# 转换为 PyTorch Tensor
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)


# 2. 定义 ANN 模型
class TrafficANN(nn.Module):
    def __init__(self):
        super(TrafficANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),  # 输入层
            nn.ReLU(),  # 激活函数
            nn.Linear(16, 8),  # 隐藏层
            nn.ReLU(),
            nn.Linear(8, 1),  # 输出层
            nn.Sigmoid()  # 二分类
        )

    def forward(self, x):
        return self.net(x)


model = TrafficANN()
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. 训练模型并记录 Loss
epochs = 100
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 验证集检测
    model.eval()
    with torch.no_grad():
        t_loss = criterion(model(X_test_tensor), y_test_tensor)
        test_losses.append(t_loss.item())

# 4. 绘图 (用于 PPT 展示)
plt.figure(figsize=(12, 5))

# 图 1: Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss', linestyle='--')
plt.title('ANN Training Loss Curve')
plt.xlabel('Epochs');
plt.ylabel('Loss');
plt.legend()

# 图 2: 决策边界 (展示分类效果)
plt.subplot(1, 2, 2)
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
probs = model(grid).reshape(xx.shape).detach().numpy()
plt.contourf(xx, yy, probs, alpha=0.8, cmap='RdYlBu')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', s=20)
plt.title('Traffic Classification Boundary')

plt.tight_layout()
plt.savefig('my_traffic_demo.png')
plt.show()