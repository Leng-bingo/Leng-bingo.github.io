---
title: SimCSE（五）
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
summary: SimCSE Background
abbrlink: '8218'
date: 2021-04-29 14:59:06
password:
keywords:
description:
---
# Simple Contrastive Learning of Sentence Embeddings（五）

### 2 Background: Contrastive Learning

----

#### <font color = "red">还需学习的知识点</font>

- 交叉熵（cross-entropy）https://www.zhihu.com/question/65288314

- 批次内负采样 （in-batch negatives）

- 博客中的公式显示问题，以后都改为图片引用，不使用代码操作

---

> Contrastive learning aims to learn effective representation by pulling semantically close neighbors together and pushing apart non-neighbors. 

- 对比学习的目的是通过把语义相近的邻域拉近在一起，把非邻域分开来学习有效的表达。

> It assumes a set of paired examples $ \mathcal{D}=\left\{\left(x_{i}, x_{i}^{+}\right)\right\}_{i=1}^{m} $ , where $x_i$ and $x_{i}^{+}$ are semantically related. 

- 它假设一组成对的例子$\mathcal{D}=\left\{\left(x_{i}, x_{i}^{+}\right)\right\}_{i=1}^{m}$，$x_i$和$x_{i}^{+}$ 是语义相关的

> We follow the contrastive framework in Chen and take a cross-entropy objective with in-batch negatives : 

- 取一个批次内负采样 (in-batch negatives)作为的交叉熵（cross-entropy）目标

> let $h_i$ and $h_{i}^{+}$ denote the representations of $x_i$ and $x_{i}^{+}$, for a mini-batch with N pairs, the training objective for ($x_i,x_{i}^{+}$) is:

- 假设$h{i}$和$h{i}^{+}$表示$x{i}$和$x{i}^{+}$，对于N对的小批量， $(x_i,x_{i}^{+})$ 的训练目标是：

$$
\ell_{i}=\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{i}^{+}\right) / \tau}}{\sum_{j=1}^{N} e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{+}\right) / \tau}}
$$

> where $\mathcal{T}$ is a temperature hyperparameter and sim$(h_1,h_2)$ is the cosine similarity $\frac{\mathbf{h}_{1}^{\top} \mathbf{h}_{2}}{\left\|\mathbf{h}_{1}\right\| \cdot\left\|\mathbf{h}_{2}\right\|}$. 

- 其中$\mathcal{T}$是温度超参数,sim$(h_1，h_2)$是余弦相似性$\frac{\mathbf{h}_{1}^{\top} \mathbf{h}_{2}}{\left\|\mathbf{h}_{1}\right\| \cdot\left\|\mathbf{h}_{2}\right\|}$

> In this work, we encode input sentences using a pre-trained language model such as BERT (Devlin et al., 2019) or RoBERTa (Liu et al., 2019): $\mathbf{h}=f_{\theta}(x)$, 

- 在这项工作中，我们使用预先训练的语言模型（如BERT或RoBERTa）对输入句子进行编码：$\mathbf{h}=f_{\theta}(x)$

> and then fine-tune all the parameters using the contrastive learning objective (Eq. 1).

- 然后使用对比学习目标（等式1）微调所有参数。

---

#### Positive instances 正例

> One critical question in contrastive learning is how to construct  ($x_i,x_{i}^{+}$)  pairs. 

- 对比学习中的一个关键问题是如何构建（$x_i，x_{i}^{+}$）对。

> In visual representations, an effective solution is to take two random transformations of the same image (e.g., cropping, flipping, distortion and rotation) as $x_i$ and $x_{i}^{+}$.

- 在视觉表示中，一个有效的解决方案是对同一个图像进行两次随机变换（例如，裁剪、翻转、失真和旋转），分别为$x_i$和$x_{i}^{+}$。

> A similar approach has been recently adopted in language representations (Wu et al., 2020; Meng et al., 2021), by applying augmentation techniques such as word deletion, reordering, and substitution.

- 最近在语言表征中也采用了类似的方法，通过应用增广技术，如单词删除，重新排序和替代。

> However, data augmentation in NLP is inherently difficult because of its discrete nature.

- 然而，NLP中的数据扩充由于其离散性而具有固有的困难性。

> As we will see in §3, using standard dropout on intermediate representations outperforms these discrete operators.

- 我们将在§3中看到.在中间表示上使用dropout优于这些离散运算符。

> In NLP, a similar contrastive learning objective has been also explored in different contexts. 

- 在自然语言处理中，在不同的语境中也探讨了类似的对比学习目标。

> In these cases,$(x_i，x_{i}^{+})$are collected from supervised datasets such as mention-entity, or question-passage pairs. 

- 在这些情况下，$（x_i，x_{i}^{+}）$是从有监督的数据集（如提及实体或问题通道对）收集的。

> Because of the distinct nature of $x_i$ and $x_{i}^{+}$ by definition, these approaches always use a dual- encoder framework, i.e., using two independent encoders $f_{\theta_{1}}$ and $f_{\theta_{2}}$ for $x_i$ and $x_{i}^{+}$. 

- 由于定义上$x_{i}$和$x_{i}^{+}$的不同性质，这些方法总是使用双编码器框架，即使用两个独立的编码器$f{\theta{1}}$和$f{\theta{2}}$来表示$x_{i}$和$x_{i}^{+}$。

> For sentence embeddings, Logeswaran and Lee (2018) also use contrastive learning with a dual-encoder approach, by forming (current sentence, next sentence) as $（x_i，x_{i}^{+}）$. 

- 对于句子嵌入，Logeswaran和Lee（2018）也使用了双编码器方法的对比学习，将（当前句子，下一个句子）形成$（x_i，x_{i}^{+}）$。

> Zhang et al. (2020) consider global sentence representations and local token representations of the same sentence as positive instances.

- Zhang将同一句子的整体句子表征和局部标记表征视为正例。

#### Alignment and uniformity 对齐性和一致性

> Recently, Wang and Isola (2020) identify two key properties related to contrastive learning: alignment and uniformity and propose metrics to measure the quality of representations. 

- 最近，Wang和Isola确定了与对比学习相关的两个关键属性：对齐性和一致性并提出了衡量表征质量的指标。

> Given a distribution of positive pairs $p_{pos}$, alignment calculates expected distance between embeddings of the paired instances (assuming representations are already normalized),

- 给定正对分布$p_{pos}$，alignment计算成对实例的嵌入之间的预期距离（假设表示已经规范化），

$$
\ell_{\text {align }} \triangleq \underset{\left(x, x^{+}\right) \sim p_{\text {pos }}}{\mathbb{E}}\left\|f(x)-f\left(x^{+}\right)\right\|^{2}
$$

> On the other hand, uniformity measures how well the embeddings are uniformly distributed:

- 另一方面，uniformity衡量嵌入物均匀分布的程度：

$$
\ell_{\text {uniform }} \triangleq \log \quad \underset{\quad x, y^\stackrel{i . i . d .}{\sim}p_{data}}{\mathbb{E}} e^{-2\|f(x)-f(y)\|^{2}}
$$

> where $p_{data}$ denotes the data distribution. These two metrics are well aligned with the objective of contrastive learning: positive instances should stay close and embeddings for random instances should scatter on the hypersphere.

- 其中，$p_{data}$表示数据分布。这两个指标很好地符合对比学习的目标：正例应该保持紧密，随机实例的嵌入应该分散在超球体上。

> In the following sections, we will also use the two metrics to justify the inner workings of our approaches.

- 在下面的部分中，我们还将使用这两个度量来证明我们的方法的内部工作。







