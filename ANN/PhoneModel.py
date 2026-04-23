import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time

torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhoneModel(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.output(x)

        return x


def data_loader():
    data = pd.read_csv('./data/手机价格预测.csv')
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x = x.astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

def train(train_dataset, input_dim, output_dim):
    # 1. 创建数据加载器, 流程: 数据 -> 张量 -> 数据集 -> 数据加载器
    # 参1: 数据集对象(1600条), 参2: 每批次的数据条数, 参3: 是否打乱数据(训练集: 打乱, 测试集: 不打乱)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 2. 创建神经网络模型.
    model = PhoneModel(input_dim, output_dim)
    # 3. 定义损失函数, 因为是多分类, 这里用的是: 多分类交叉熵损失函数.
    criterion = nn.CrossEntropyLoss()
    # 4. 创建优化器对象.
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 5. 模型训练.
    # 5.1 定义变量, 记录训练的 总轮数.
    epochs = 50
    # 5.2 开始(每轮的)训练.
    for epoch in range(epochs):
        # 5.2.1 定义变量, 记录每次训练的损失值, 训练批次数.
        total_loss, batch_num = 0.0, 0
        # 5.2.2 定义变量, 表示训练开始的时间.
        start = time.time()
        # 5.2.3 开始本轮的 各个批次的训练.
        for x, y in train_loader:
            # 5.2.4 切换模型(状态)
            model.train()   # 训练模式.    model.eval()   # 测试模式.
            # 5.2.5 模型预测.
            y_pred = model(x)
            # 5.2.6 计算损失.
            loss = criterion(y_pred, y)
            # 5.2.7 梯度清零, 反向传播, 优化参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 5.2.8 累加损失值.
            total_loss += loss.item()
            batch_num += 1
        # 5.2.4 至此, 本轮训练结束, 打印训练信息.
        print(f'epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start:.2f}s')

    # 6. 走到这里, 说明多轮训练结束, 保存模型(参数)
    # 参1: 模型对象的参数(权重矩阵, 偏置矩阵)  参2: 模型保存的文件名.
    # print(f'\n\n模型的参数信息: {model.state_dict()}\n\n')
    torch.save(model.state_dict(), './model/phone.pth') # 后缀名用: pth, pkl, pickle均可.

# todo 4. 模型测试.
def evaluate(test_dataset, input_dim, output_dim):
    # 1. 创建神经网络分类对象.
    model = PhoneModel(input_dim, output_dim)
    # 2. 加载模型参数.
    model.load_state_dict(torch.load('./model/phone.pth'))
    # 3. 创建测试集的 数据加载器对象.
    # 参1: 数据集对象(400条), 参2: 每批次的数据条数, 参3: 是否打乱数据(训练集: 打乱, 测试集: 不打乱)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # 4. 定义变量, 记录预测正确的样本个数.
    correct = 0
    # 5. 从数据加载器中, 获取到每批次的数据.
    for x, y in test_loader:
        # 5.1 切换模型状态 -> 测试模式.
        model.eval()
        # 5.2 模型预测.
        y_pred = model(x)
        # print(f'y_pred: {y_pred}')  # [[0分类概率, 1分类概率, 2分类概率, 3分类概率], [...]...]

        # 5.3 根据加权求和, 得到类别, 用argmax()获取最大值对应的下标, 就是类别.
        y_pred = torch.argmax(y_pred, dim=1)    # dim=1 表示逐行处理.
        # print(f'y_pred: {y_pred}')  # [第1条数据的预测分类, ...]
        # print(f'y: {y}')

        # 5.4 统计预测正确的样本个数.
        # print(y_pred == y)          # tensor([ True,  True,  True,  True, False, False,  True, False])
        # print((y_pred == y).sum())  # True:1, False:0
        correct += (y_pred == y).sum()

    # 6.走到这里, 模型预测结束, 打印准确率即可.
    print(f'准确率(Accuracy): {correct / len(test_dataset):.4f}')


# todo 5. 测试
if __name__ == '__main__':
    # 1. 准备数据集.
    train_dataset, test_dataset, input_dim, output_dim = data_loader()
    # print(f'训练集 数据集对象: {train_dataset}')
    # print(f'测试集 数据集对象: {test_dataset}')
    # print(f'输入特征数: {input_dim}')    # 20
    # print(f'输出标签数: {output_dim}')   # 4

    # 2. 构建神经网络模型.
    # model = PhonePriceModel(input_dim, output_dim)
    # 计算模型参数
    # 参1: 模型对象. 参2: 输入数据的形状(批次大小, 输入特征数), 每批16条, 每条20列特征
    # summary(model, input_size=(16, input_dim))

    # 3. 模型训练
    # train(train_dataset, input_dim, output_dim)

    # 4. 模型测试.
    # evaluate(test_dataset, input_dim, output_dim)