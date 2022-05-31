import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)
np.random.seed(1)
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 512)
        )

    def forward(self, inputs):
        return self.model(inputs)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)


def normal_pdf(x, mu, sigma):
    ''' 正态分布概率密度函数'''
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


def draw(G, epoch, g_input_size):
    '''画目标分布和生成分布'''
    plt.clf()
    # 画出目标分布
    x = np.arange(-3, 9, 0.2)
    y = normal_pdf(x, 3, 1)
    plt.plot(x, y, 'r', linewidth=2)

    # 画出生成的分布
    test_data = torch.rand(1, g_input_size)
    data = G(test_data).detach().numpy()
    mean = data.mean()
    std = data.std()
    x = np.arange(np.floor(data.min()) - 5, np.ceil(data.max()) + 5, 0.2)
    y = normal_pdf(x, mean, std)
    plt.plot(x, y, 'orange', linewidth=2)
    plt.hist(data.flatten(), bins=20, color='y', alpha=0.5, rwidth=0.9, density=True)


    plt.legend(['目标分布', '生成分布'])
    plt.title('GAN：epoch' + str(epoch))
    plt.show()
    plt.pause(0.1)


def train():
    
    G_mean = []
    G_std = [] # 用于记录生成器生成的数据的均值和方差
    data_mean = 3
    data_std = 1  # 目标分布的均值和方差
    batch_size = 64
    g_input_size = 16
    g_output_size = 512
    
    epochs = 1001
    d_epoch = 1  # 每个epoch判别器的训练轮数

    # 初始化网络
    D = Discriminator()
    G = Generator()

    # 初始化优化器和损失函数
    d_learning_rate = 0.01
    g_learning_rate = 0.001
    loss_func = nn.BCELoss()  # - [p * log(q) + (1-p) * log(1-q)]
    optimiser_D = optim.Adam(D.parameters(), lr=d_learning_rate)
    optimiser_G = optim.Adam(G.parameters(), lr=g_learning_rate)

    plt.ion()
    for epoch in range(epochs):
        G.train()
        # 1 训练判别器d_steps次
        for _ in range(d_epoch):
            # 1.1 真实数据real_data输入D,得到d_real
            real_data = torch.tensor(np.random.normal(data_mean, data_std, (batch_size, g_output_size)),
                                     dtype=torch.float)
            d_real = D(real_data)
            # 1.2 生成数据的输出fake_data输入D,得到d_fake
            g_input = torch.rand(batch_size, g_input_size)
            fake_data = G(g_input).detach()  # detach：只更新判别器的参数
            d_fake = D(fake_data)

            # 1.3 计算损失值 ，判别器学习使得d_real->1、d_fake->0
            loss_d_real = loss_func(d_real, torch.ones([batch_size, 1]))
            loss_d_fake = loss_func(d_fake, torch.zeros([batch_size, 1]))
            d_loss = loss_d_real + loss_d_fake

            # 1.4 反向传播，优化
            optimiser_D.zero_grad()
            d_loss.backward()
            optimiser_D.step()

        # 2 训练生成器
        # 2.1 G输入g_input，输出fake_data。fake_data输入D，得到d_g_fake
        g_input = torch.rand(batch_size, g_input_size)
        fake_data = G(g_input)
        d_g_fake = D(fake_data)

        # 2.2 计算损失值，生成器学习使得d_g_fake->1
        loss_G = loss_func(d_g_fake, torch.ones([batch_size, 1]))

        # 2.3 反向传播，优化
        optimiser_G.zero_grad()
        loss_G.backward()
        optimiser_G.step()
        # 2.4 记录生成器输出的均值和方差
        G_mean.append(fake_data.mean().item())
        G_std.append(fake_data.std().item())

        if epoch % 10 == 0:
            print("Epoch: {}, 生成数据的均值: {}, 生成数据的标准差: {}".format(epoch, G_mean[-1], G_std[-1]))
            print('-' * 10)
            G.eval()
            draw(G, epoch, g_input_size)

    plt.ioff()
    plt.show()
    plt.plot(G_mean)
    plt.title('均值')
    plt.savefig('gan_mean.jpg')
    plt.show()

    plt.plot(G_std)
    plt.title('标准差')
    plt.savefig('gan_std.jpg')
    plt.show()

if __name__ == '__main__':
    train()
