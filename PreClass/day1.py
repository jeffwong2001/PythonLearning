import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# 1. 设置设备：既然有 3090，就要充分利用显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#wguang

def create_dataset():
    x, y, coef = make_regression(
        n_samples=100, n_features=1, noise=12, bias=21, coef=True, random_state=3
    )
    # --- 修改 1: NumPy 转 Tensor，并统一为 float32 (显卡更喜欢的格式) ---
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float().view(-1, 1)  # 必须把 y 变成 (100, 1) 的形状
    return x, y, coef


def train(x, y, coef):
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # --- 修改 2: 将模型搬到显卡上 ---
    model = nn.Linear(1, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epoch_num = 100
    loss_list = []

    for epoch in range(epoch_num):
        total_loss = 0
        for train_x, train_y in dataloader:
            # --- 修改 3: 将训练数据也搬到显卡上 ---
            train_x, train_y = train_x.to(device), train_y.to(device)

            y_pred = model(train_x)
            loss = criterion(y_pred, train_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 记录每一轮的平均损失
        avg_loss = total_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f'轮数: {epoch}, 平均损失值: {avg_loss:.4f}')

    # --- 绘图部分优化 ---
    plt.figure(figsize=(10, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(epoch_num), loss_list)
    plt.title('Loss Curve')

    # 预测结果
    plt.subplot(1, 2, 2)
    plt.scatter(x.numpy(), y.numpy(), alpha=0.5, label='Actual Data')

    # 使用模型直接生成预测，不要用循环手动计算
    with torch.no_grad():  # 预测时关闭梯度计算
        # 模型在显卡上，数据也得传过去，算完再传回 CPU 绘图
        y_final_pred = model(x.to(device)).cpu()

    plt.plot(x.numpy(), y_final_pred.numpy(), color='red', label='Model Prediction')
    plt.legend()

    # 如果你在服务器上没有显示器，建议保存为图片
    plt.savefig('result.png')
    print("训练完成，结果已保存为 result.png")
    plt.show()


if __name__ == '__main__':
    x, y, coef = create_dataset()
    train(x, y, coef)
