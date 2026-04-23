import torch
import torch.nn as nn
from torchsummary import summary

# 确定运行设备：如果有显卡就用显卡，没有就用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备是: {device}")

class ModelDemo(nn.Module):
    def __init__(self):
        super().__init__()
        # 隐藏层及输出层定义
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2)

        # 参数初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.softmax(self.output(x), dim=-1)
        return x

def train():
    # 1. 创建模型对象，并移动到指定设备（GPU 或 CPU）
    my_model = ModelDemo().to(device) # 👈 关键点1: 将模型整体搬到 GPU

    # 2. 创建随机数据集，并移动到指定设备
    # 注意：输入数据的 shape 是 (3,) 或 (N, 3)，这里 summary 需要输入维度
    data = torch.randn(size=(5, 3)).to(device) # 👈 关键点2: 将输入数据搬到 GPU

    print(f'data.device: {data.device}')

    # 3. 前向传播
    output = my_model(data)
    print(f'output: {output}')
    print(f'output.device: {output.device}') # 此时输出也会在 GPU 上
    print('-' * 30)

    # 4. 查看模型结构
    # 注意：summary 内部会尝试创建一个 dummy 变量，
    # 在某些版本中它可能不会自动处理设备，建议传入已经 to(device) 的模型
    summary(my_model, input_size=(5,3)) # 这里 input_size 填特征数即可

    print('=================== 查看模型参数 ===================')
    for name, param in my_model.named_parameters():
        print(f'name: {name}')
        print(f'param: {param} \n')

# todo: 3.测试
if __name__ == '__main__':
    train()

