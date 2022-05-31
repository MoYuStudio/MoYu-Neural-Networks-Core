from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

torch.manual_seed(1)
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 4),
            nn.Sigmoid(),
            nn.Linear(4, 8),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)


def train():
    D = Discriminator()
    G = Generator()
    loss_func = nn.BCELoss() # - [p * log(q) + (1-p) * log(1-q)]
    optimiser_D = torch.optim.Adam(D.parameters(), lr=1e-3)
    optimiser_G = torch.optim.Adam(G.parameters(), lr=1e-3)
    output_list = []
    for i in range(1000):
        # 1 训练判别器
        # 1.1 真实数据real_data输入D,得到d_real
        real_data = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.float)
        d_real = D(real_data)
        
        # 1.2 生成数据的输出fake_data输入D,得到d_fake
        fake_data = G(torch.tensor([0.5])).detach()  # detach：只更新判别器的参数
        d_fake = D(fake_data)
        
        # 1.3 计算损失值 ，判别器学习使得d_real->1、d_fake->0
        loss_d_real = loss_func(d_real, torch.tensor([1.0]))
        loss_d_fack = loss_func(d_fake, torch.tensor([0.]))
        loss_d = loss_d_real + loss_d_fack
        
        # 1.4 反向传播更新参数
        optimiser_D.zero_grad()
        loss_d.backward()
        optimiser_D.step()

        # 2 训练生成器
        # 2.1 G的输出fake_data输入D，得到d_g_fake
        fake_data = G(torch.tensor([0.5]))  
        d_g_fake = D(fake_data)
        # 2.2 计算损失值，生成器学习使得d_g_fake->1
        loss_g = loss_func(d_g_fake, torch.tensor([1.0]))
        
        # 2.3 反向传播，优化
        optimiser_G.zero_grad()
        loss_g.backward()
        optimiser_G.step()
        
        if i % 100 == 0:
            output_list.append(fake_data.detach().numpy())
            print(fake_data.detach().numpy())
            
    plt.imshow(np.array(output_list), cmap='Blues')
    plt.colorbar()
    plt.xticks([])
    plt.ylabel('迭代次数X100')
    plt.show()


if __name__ == '__main__':
    train()
