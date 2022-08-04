---
title: SimCSE（六）
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
summary: Unsupervised and supervised SimCSE
abbrlink: ca01
date: 2021-04-30 13:39:24
password:
keywords:
description:
---
# Simple Contrastive Learning of Sentence Embeddings（六）

### 3 Unsupervised SimCSE and supervised SimCSE

---

#### <font color = "red">还需学习的知识点</font>

- dropout mask，mask是什么意思
- STS-B（Semantic Textual Similarity Benchmark）：语义文本相似度基准

- write one sentence that is absolutely true (entailment), one that might be true (neutral), and one that is definitely false (contradiction). 

---

> In this section, we describe our unsupervised SimCSE model. 

- 在本节中，我们将介绍我们的无监督SimCSE模型。

> The idea is extremely simple: we take a collection of sentences $\left\{x_{i}\right\}_{i=1}^{m}$ and use $x_i^+=x_i$. 

- 这个想法非常简单：我们取一组句子$\left\{x_{i}\right\}_{i=1}^{m}$，然后使$x_i^+=x_i$。

> The key ingredient to get this to work with identical positive pairs is through the use of independently sampled *dropout masks*. 

- 关键的因素是通过使用独立取样的“dropout masks”，使这项工作与相同的正对。

> In standard training of Transformers (Vaswani et al., 2017), there is a dropout mask placed on fully-connected layers as well as attention probabilities (default p = 0.1). 

- 在Transformers的标准训练中，在完全连接的层上以及注意概率（默认p=0.1）上放置了一个dropout mask。

> We denote $\mathbf{h}_{i}^{z}=f_{\theta}\left(x_{i}, z\right)$ where $z$ is a random mask for dropout. 

- 我们用$\mathbf{h}_{i}^{z}=f_{\theta}\left(x_{i}, z\right)$表示，$z$是一个随机的dropout mask

> We simply feed the same input to the encoder twice by applying different dropout masks  $z, z^{\prime}$ and the training objective becomes:

- 我们只需通过应用不同的dropout mask，$z，z^{\prime}$将相同的输入输入输入编码器两次，训练目标就变成：

> for a mini-batch with N sentences. 

- 对于少量的句子N

> Note that z is just the standard dropout mask in Transformers and we do not add any additional dropout.

- 注意，z只是转变中的标准dropout mask，我们没有添加任何额外的dropout。

#### Dropout noise as data augmentation 作为数据增强的dropout噪声

> We view this approach as a minimal form of data augmentation: the positive pair takes exactly the same sentence, and their embeddings only differ in dropout masks. 

- 我们将这种方法视为数据扩充的一种最小形式：正例对采用完全相同的句子，它们的嵌入只在dropout mask上有所不同。

> We compare this approach to common augmentation techniques and other training objectives on the STS-B development set (Cer et al., 2017).

- 我们将这种方法与常见的增强技术和STS-B开发集上的其他训练目标进行了比较。

> We use $N=512$ and $m = 10^6$ sentences randomly drawn from English Wikipedia in these experiments. 

在这些实验中，我们使用从英文维基百科中随机抽取的$N=512$ 和 $m = 10^6$ 个句子。

> **Table 2** compares our approach to common data augmentation techniques such as crop, word deletion and replacement, which can be viewed as $\mathbf{h}=f_{\theta}(g(x), z)$ and $g$ is a (random) discrete operator on $x$. 

- **表2**将我们的方法与常见的数据扩充技术（如裁剪、字删除和替换）进行了比较，这些技术可以看作是$\mathbf{h}=f_{\theta}(g(x), z)$ ，而$g$是$x$上的（随机）离散运算符。

  <img src="Table 2.png" alt="Table 2" style="zoom:50%;" />



> We find that even deleting one word would hurt performance and none of the discrete augmentations outperforms basic dropout noise.

- 我们发现，即使删除一个单词也会影响性能，并且没有一个离散增强比基本的丢失噪声更好。

> We also compare this self-prediction training objective to next-sentence objective used in Logeswaran and Lee (2018), taking either one encoder or two independent encoders. 

- 我们还将这个自我预测训练目标与Logeswaran和Lee中使用的下一个句子目标进行比较，选择一个编码器或两个独立的编码器。

> As shown in Table 3, we find that SimCSE performs much better than the next-sentence objectives (79.1 vs 69.7 on STS-B) and using one encoder instead of two makes a significant difference in our approach.

- 如表3所示，我们发现SimCSE比下一个句子目标（79.1 vs STS-B为69.7）的表现要好得多，并且使用一个编码器而不是两个编码器使我们的方法有显著的不同。

  <img src="Table 3.png" alt="Table 3" style="zoom:50%;" />

---

#### Why does it work?

> To further understand the role of dropout noise in unsupervised SimCSE, we try out different dropout rates in Table 4 and observe that all the variants underperform the default dropout probability p = 0.1 from Transformers. 

- 为了进一步了解在无监督SimCSE中丢失噪声的作用，我们在表4中尝试了不同的丢失率，并观察到所有变体都低于Transformers的默认丢失概率p=0.1。

<img src="Table 4.png" alt="Table 4" style="zoom:50%;" />

> We find two extreme cases particularly interesting: “no dropout” (p = 0) and “fixed 0.1” (using default dropout p = 0.1 but the same dropout masks for the pair).

- 我们发现两个极端的情况特别有趣：“无dropout”（p=0）和“固定0.1”（使用默认的dropoutp=0.1，但对这两种情况使用相同的dropout mask）。





 In both cases, the resulting embeddings for the pair are exactly the same, and it leads to a dramatic performance degradation. 

- 在这两种情况下，生成的嵌入对完全相同，这会导致性能急剧下降。

> We take the checkpoints of these models every 10 steps during training and visualize the alignment and uniformity metrics2 in Figure 2, along with a simple data augmentation model “delete one word”. 

- 在训练过程中，我们每10步对这些模型进行一次检查点检查，并在图2中可视化对齐和一致性度量2，以及一个简单的数据处理模型“删除一个单词”。

<img src="Figure 2.png" alt="Figure 2" style="zoom:50%;" />

> As is clearly shown, all models largely improve the uniformity. 

- 如图所示，所有模型在很大程度上改善了均匀性。

> However, the alignment of the two special variants also degrades drastically, while our unsupervised SimCSE keeps a steady alignment, thanks to the use of dropout noise. 

- 然而，这两种特殊变体的对齐也会急剧下降，而我们的无监督SimCSE由于使用了衰减噪声，保持了稳定的对齐。

> On the other hand, although “delete one word” slightly improves the alignment, it has a smaller gain on the uniformity, and eventually underperforms unsupervised SimCSE.

- 另一方面，虽然“删除一个单词”稍微提高了对齐度，但它在一致性方面的增益较小，最终表现不如无监督SimCSE。

------

### 4 Supervised SimCSE

> We have demonstrated that adding dropout noise is able to learn a good alignment for positive pairs $\left(x, x^{+}\right) \sim p_{\text {pos }}$. 

- 我们已经证明，对于正对$\left(x, x^{+}\right) \sim p_{\text {pos }}$，添加dropout噪声能够学习良好的对齐性。

> In this section, we study whether we can leverage supervised datasets to provide better training signals for improving alignment of our approach.

- 在本节中，我们将研究是否可以利用监督数据集来提供更好的训练信号，以改进方法的一致性。

> Prior work (Conneau et al., 2017; Reimers and Gurevych, 2019) has demonstrated that supervised natural language inference (NLI) datasets (Bowman et al., 2015; Williams et al., 2018) are effective for learning sentence embeddings, by predicting whether the relationship between two sentences is *entailment*, *neutral* or *contradiction*. 

- 前期工作已经证明了有监督的自然语言推理（NLI）数据集通过预测两个句子之间的关系是蕴涵、中性还是矛盾，来有效地学习句子嵌入。

> In our contrastive learning framework, we instead directly take $(x_i,x_i^+)$pairs from supervised datasets and use them to optimize Eq. 1.

- 在我们的对比学习框架中，我们直接从有监督的数据集中提取$(x_i,x_i^+)$对，并使用它们来优化等式1。

$$
\ell_{i}=\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{i}^{+}\right) / \tau}}{\sum_{j=1}^{N} e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{+}\right) / \tau}}
$$

---

#### Exploiting supervised data利用监督数据

> We first explore which annotated datasets are especially suitable for constructing positive pairs $(x_i,x_i^+)$. 

- 我们首先探讨哪些带注释的数据集特别适合于构造正对$(x_i,x_i^+)$。

> We exiperiment with a number of datasets with sentence-pair examples, including QQP4: Quora question pairs; Flickr30k (Young et al., 2014): each image is annotated with 5 human-written captions and we consider any two captions of the same image as a positive pair; ParaNMT (Wieting and Gimpel, 2018): a large-scale back-translation paraphrase dataset; and finally NLI datasets: SNLI (Bowman et al., 2015) and MNLI (Williams et al., 2018).

- 我们用一些句子对的例子进行了实验，包括QQP问题对；Flickr：每幅图像都有5个人类书写的字幕注释，我们将同一幅图像的任意两个字幕视为正对；ParaNMT：大规模反译释义数据集；最后是NLI数据集：SNLI和MNLI。

> We train the contrastive learning model (Eq. 1) with different datasets and compare the results in Table 5 （for a fair comparison, we also run experiments with the same # of training pairs）. 

- 我们用不同的数据集训练对比学习模型（等式1），并比较表5中的结果（为了公平比较，我们还用相同的训练对进行实验）。

<img src="Table 5.png" alt="Table 5" style="zoom:50%;" />

> We find that most of these models using supervised datasets outperform our unsupervised approach, showing a clear benefit from supervised signals. 

- 我们发现，大多数使用监督数据集的模型都比我们的无监督方法有更好的性能，显示出监督信号的明显优势。

> Among all the options, using entailment pairs from the NLI (SNLI + MNLI) datasets perform the best. 

在所有选项中，使用NLI（SNLI+MNLI）数据集entailment对表现最好。

> We think this is reasonable, as the NLI datasets consist of high-quality and crowd-sourced pairs, and human annotators are expected to write the hypotheses manually based on the premises, and hence two sentences tend to have less lexical overlap. 

- 我们认为这是合理的，因为NLI数据集由高质量和众包的成对数据组成，并且人类注释者需要根据前提手工编写假设，因此两句话的词汇重叠较少。

> For instance, we find that the lexical overlap (F1 measured between two bags of words) for the entailment pairs (SNLI + MNLI) is 39%, while they are 60% and 55% for QQP and ParaNMT.

- 例如，我们发现蕴涵对（SNLI+MNLI）的词汇重叠（两袋词之间的F1）为39%，而QQP和ParaNMT分别为60%和55%。

---

#### Contradiction as hard negatives否定的矛盾

> Finally, we further take the advantage of the NLI datasets by using its contradiction pairs as hard negatives. 

- 最后，我们进一步利用NLI数据集的矛盾对作为硬否定。

> In NLI datasets, given one premise, annotators are required to manually write one sentence that is absolutely true (entailment), one that might be true (neutral), and one that is definitely false (contradiction). 

- 在NLI数据集中，给定一个前提，注释者需要手动编写一个绝对正确的句子（蕴涵），一个可能正确的句子（中性），一个绝对错误的句子（矛盾）。

> Thus for each premise and its entailment hypothesis, there is an accompanying contradiction hypothesis7 (see Figure 1 for an example).

- 因此，对于每个前提及其蕴涵假设，都有一个伴随的矛盾假设（参见图1中的示例）。

> Formally, we extend $(x_i,x_i^+)$ to $(x_i,x_i^+,x_i^-)$ where $x_i$ is the premise, $x_i^+$and $x_i^-$ are entailment and contradiction hypotheses. 

- 形式上，我们将$（x_i，x_i^+）$扩展到$(x_i,x_i^+,x_i^-)$，其中$x_i$是前提，$x_i^+$和$x_i^-$是蕴涵和矛盾假设

> The training objective $\ell_{i}$ is then defined by (N is the mini-batch size):

- 训练目标$\ell{i}$由（N是最小批量大小）定义：

$$
-\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{i}^{+}\right) / \tau}}{\sum_{j=1}^{N}\left(e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{+}\right) / \tau}+e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{-}\right) / \tau}\right)}
$$

> As shown in Table 5, adding hard negatives can further improve performance (84.9 $\rightarrow$ 86.2) and this is our final supervised SimCSE. 

- 如表5所示，添加消极可以进一步提高性能（84.9$\rightarrow$86.2），这是我们的最终有监督SimCSE。

> We also tried to add the ANLI dataset (Nie et al., 2020) or combine it with our unsupervised SimCSE approach, but didn’t find a meaningful improvement. 

- 我们还尝试添加ANLI数据集或将其与我们的无监督SimCSE方法相结合，但没有发现有意义的改进。

> We also considered a dual encoder framework in supervised SimCSE and it hurt performance (86.2$\rightarrow$ 84.2).

- 我们还考虑了监督SimCSE中的双编码器框架，它会影响性能（86.2$\rightarrow$84.2）。

