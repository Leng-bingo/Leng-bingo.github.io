---
title: pytorch知识点
top: false
cover: false
toc: false
mathjax: true
tags:
  - 研究生
  - 代码
categories:
  - 代码
summary: pytorch知识点
abbrlink: 3b1e
date: 2022-08-17 15:31:35
password:
keywords:
description:
---

- dir():打开，看见
  - 能让我们知道工具箱以及工具箱中的分割区有什么东西。
- help（）：说明书
  - 能让我们知道每个工具是如何使用的，工具的使用方法。

---

- jupyter
  - shift+回车：快捷运行，以每一块作为运行整体

- 加载数据
  - Dataset：提供一种方式去获取。如何获取每个数据及其label，告诉我们总共有多少个数据
  - Dataloader：为后面的网络提供不同的数据形式

---

- tensorboard

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')

# writer.add_image()

# y = x
for i in range(100):
    writer.add_scalar('y=x', i, i)


writer.close()
```

- 使用`tensorboard --logdir=logs`，logs为目录
- 使用`tensorboard --logdir=logs --port=6007`，如果端口占用，可用可选端口

---

- 在项目文件夹中放入图片

```python
image_url = './dataset/练手数据集/val/bees/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg'
img = Image.open(image_url)
np_img = np.array(img)
writer.add_image('test', np_img, 2, dataformats='HWC')
```

---

- Transforms的使用

- 是一个工具箱，里面有很多函数

  - ToTensor，resize等等

- python中的用法 -》 tensor数据类型

- 通过transform.ToTensor去解决两个问题

  - 1、transform如何使用，需要先穿件函数 实例，因为里面有__call__方法，功能是把函数可以实例化为类使用。应先实例化，再调用。

  - ```python
    from PIL import Image
    from torchvision import transforms
    
    img_path = './dataset/练手数据集/train/ants_image/0013035.jpg'
    img = Image.open(img_path)
    # print(img)
    tensor_trans = transforms.ToTensor()
    tensor_img = tensor_trans(img)
    print(tensor_img)
    ```

  - 2、为什么需要Tensor数据类型

  - \__call__用法：

- 常见的Transforms

  - 输入：PIL，Image.open，tensor，narrays
  - 输出：

**忽略大小写，进行提示匹配**

- Compose method: It is need list, [item1, item 2],they must be transforms type

**关注函数的输入和输出类型，多看官方文档。关注方法需要的参数**

---

**神经网络股价骨架

- 先继承nn.model
- 卷积层：Conv2d
  - in_channel：同时输入的数据量
  - out_channel:卷积计算后的个数，有几个卷积核就有几个输出通道，卷积核一般不同

---

- 池化层 pooling layers
  - Max pooling，也被称作下采样，选kennal核最大的那个数字
  - dtype选择float才可以计算
  - 选取特征，减少大小

---

- 非线性激活层（举例ReLu）

---

- 线性层以及其他层
  - 正则化层，用的不是很多
  - `torch.flatten`把输入展开为一行

---

- sequential：

---

- 损失函数与反向传播
  - loss：1、计算输出和目标的差距。2、为我们更新输出提供一定的依据**（反向传播）**,利用梯度
    - MSE:平方平均。L1loss：加和平均
    - 交叉熵：C类
    - ![](https://leng-mypic.oss-cn-beijing.aliyuncs.com/mac-img/20220817100145.png)

```python
# 1-batch_size,1 channel, 1line3column
inputs = torch.reshape(inputs, (1, 1, 1, 3))
```

---

- 优化器

  - lr学习速率，learning rate
  - `optimizer.zero_grad()`每一次后清零梯度
  - 求梯度

- 训练简单步骤

- 

- ```python
  loss = nn.CrossEntropyLoss()
  tudui = Tudui()
  # 优化器
  optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
  for epoch in range(20):
      running_loss = 0.0
      for data in dataloader:
          imgs, targets = data
          outputs = tudui(imgs)
          result_loss = loss(outputs, targets)
          # 梯度为0
          optim.zero_grad()
          result_loss.backward()
          optim.step()
          running_loss = running_loss + result_loss
      print(running_loss)
  ```

---

- 模型使用与修改

  - 增加已有模型，增加一层

  - ```python
    vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
    ```

  - 修改已有的层级

  - ```python
    vgg16_false.classifier[6] = nn.Linear(4096, 10)
    ```

---

- 模型的保存和读取

- 减少内存，把tensor提取出数字

- ```python
  loss.item()
  ```

- ```python
  # 保存方式1,模型结构+模型参数
  torch.save(vgg16, "vgg16_method1.pth")
  model = torch.load("vgg16_method1.pth")
  # 保存方式2，模型参数（官方推荐）
  torch.save(vgg16.state_dict(), "vgg16_method2.pth")
  vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
  ```

---

- 完整模型训练步骤

  ```python
  import torchvision
  from torch.utils.tensorboard import SummaryWriter
  
  from model import *
  # 准备数据集
  from torch import nn
  from torch.utils.data import DataLoader
  
  train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                            download=True)
  test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                           download=True)
  
  # length 长度
  train_data_size = len(train_data)
  test_data_size = len(test_data)
  # 如果train_data_size=10, 训练数据集的长度为：10
  print("训练数据集的长度为：{}".format(train_data_size))
  print("测试数据集的长度为：{}".format(test_data_size))
  
  # 利用 DataLoader 来加载数据集
  train_dataloader = DataLoader(train_data, batch_size=64)
  test_dataloader = DataLoader(test_data, batch_size=64)
  
  # 创建网络模型
  tudui = Tudui()
  
  # 损失函数
  loss_fn = nn.CrossEntropyLoss()
  
  # 优化器
  # learning_rate = 0.01
  # 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
  learning_rate = 1e-2
  optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)
  
  # 设置训练网络的一些参数
  # 记录训练的次数
  total_train_step = 0
  # 记录测试的次数
  total_test_step = 0
  # 训练的轮数
  epoch = 10
  
  # 添加tensorboard
  writer = SummaryWriter("./logs")
  start_time = time.time()
  for i in range(epoch):
      print("-------第 {} 轮训练开始-------".format(i+1))
  
      # 训练步骤开始
      tudui.train()
      for data in train_dataloader:
          imgs, targets = data
          outputs = tudui(imgs)
          loss = loss_fn(outputs, targets)
  
          # 优化器优化模型
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  
          total_train_step = total_train_step + 1
          if total_train_step % 100 == 0:
            	end_time = time.time()
              print(end_time - start_time)
              print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
              writer.add_scalar("train_loss", loss.item(), total_train_step)
  
      # 测试步骤开始
      tudui.eval()
      total_test_loss = 0
      total_accuracy = 0
      with torch.no_grad():
          for data in test_dataloader:
              imgs, targets = data
              outputs = tudui(imgs)
              loss = loss_fn(outputs, targets)
              total_test_loss = total_test_loss + loss.item()
              accuracy = (outputs.argmax(1) == targets).sum()
              total_accuracy = total_accuracy + accuracy
  
      print("整体测试集上的Loss: {}".format(total_test_loss))
      print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
      writer.add_scalar("test_loss", total_test_loss, total_test_step)
      writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
      total_test_step = total_test_step + 1
  
      torch.save(tudui, "a_{}.pth".format(i))
      print("模型已保存")
  
  writer.close()
  ```

---

- GPU训练

- 

- ```python
  device = torch.device("cuda")
  tudui = tudui.to(device)
  ```

---

- 测试

---

- 参数设置