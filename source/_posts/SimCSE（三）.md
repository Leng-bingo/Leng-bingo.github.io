---
title: SimCSE（三）
top: false
cover: false
toc: false
mathjax: true
tags:
  - 论文
  - 研究生
  - 知识图谱
  - SimCSE
categories:
  - 论文
abbrlink: 3b4c
date: 2021-04-28 10:31:04
password:
keywords:
description:
summary: ICML2020
---
## Simple Contrastive Learning of Sentence Embeddings（三）

#### <font color = "red">还需学习的知识点</font>

- **Alignment**
- **Uniformity**

------

#### 对比表示学习（Contrastive Representation Learning）


对比学习的核心思想是将正样本和负样本在特征空间对比，学习样本的特征表示，难点在于如何构造正负样本。

> **Title**: 《Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere》
> **Author**:Tongzhou Wang ; Phillip Isola



[Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](ICML2020.pdf) （通过超球面上的对齐和一致性理解对比表示学习）指出了Contrastive representation learning的两个重要属性：

- **Alignment（计算正例对之间的向量距离的期望）:** two samples forming a positive pair should be mapped to nearby features, and thus be （mostly） invariant to unneeded noise factors. **Similar samples have similar features.** （正例之间表示保持较近距离）
  - 越相似的样例之间的alignment程度越高。因为alignment使用距离来衡量，所以距离越小，表示alignment的程度越高。
- **Uniformity（评估所有数据的向量均匀分布的程度，越均匀，保留的信息越多）:** feature vectors should be roughly uniformly distributed on the unit hypersphere, pre-serving as much information of the data as possible. **Illustration of alignment and uniformity of feature distributions on the output unit hypersphere.** （随机样例的表示应分散在超球面上）
  - 可以想象任意从表示空间中采样两个数据和, 希望他们的距离比较远。他们的距离越远，证明空间分布越uniform。所以uniformity的值也是越低越好。

![对比表示学习](对比表示学习.webp)

SimCSE也采用这两个指标来衡量生成的句子向量，并证明了文本的语义空间也满足：alignment值越低且uniformity值越低，向量表示的质量越高，在STS任务上的Spearman相关系数越高。


- 作者证明了现有的一些对比学习的算法正是较好地满足了这两条性质才取得了不错的效果。
- 作者提出了一个可优化的 metric (策略方法)来直接量化这两条属性。通过直接优化该loss（损失），也取得了较好的效果。

