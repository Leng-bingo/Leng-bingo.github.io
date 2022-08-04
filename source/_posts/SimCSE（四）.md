---
title: SimCSE（四）
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
abbrlink: 40a6
date: 2021-04-28 12:38:25
password:
keywords:
description:
summary: SimCSE Introduction
---

# Simple Contrastive Learning of Sentence Embeddings（四）

**SimCSE**有两个变体：**Unsupervised SimCSE**和**Supervised SimCSE**，主要不同在于对比学习的正负例的构造。下面详细介绍下他们的构造方式。

----

####  <font color = "red">还需学习的知识点</font>

- **in-batch和negatives方法还不懂，需要学习**

- **正例和负例具体指什么数据**

- **自然语言推理（NLI）数据集**

- **句子嵌入是什么意思以及如何实现**

- **有监督和无监督是什么意思**

-----

## 无监督SimCSE

> Learning universal sentence embeddings is a fundamental problem in natural language processing and has been studied extensively in the literature.

- 通用句嵌入的学习是自然语言处理中的一个基本问题，已有文献对此进行了广泛的研究。

> In this work, we advance state-of-the-art sentence embedding methods and demonstrate that a contrastive objective can be extremely effective in learning sentence embeddings, coupled with pre-trained language models such as BERT and RoBERTa. 

- 在这项工作中，我们提出了最先进的句子嵌入方法，并证明对比目标在学习句子嵌入时是非常有效的，再加上预先训练的语言模型，如BERT和RoBERTa。

> We present SimCSE, a simple contrastive sentence embedding framework, which can be used to produce superior sentence embeddings, from either unlabeled or labeled data.

- 我们提出了一个简单的对比句子嵌入框架SimCSE，它可以用来从未标记或标记的数据中产生更好的句子嵌入。



> Our *unsupervised* SimCSE simply predicts the input sentence itself, with only *dropout* (Srivastava et al., 2014) used as noise （Figure 1（a））.

- 我们的无监督SimCSE只是预测输入句子本身，只有*dropout*用作噪声（图1（a））。

<img src="Unsupervised SimCSE.png" alt="Unsupervised SimCSE" style="zoom:50%;" />

> In other words, we pass the same input sentence to the pretrained encoder twice and obtain two embeddings as “positive pairs”, by applying independently sampled dropout masks. 

- 换言之，我们将相同的输入语句传递给预训练的编码器两次，并通过应用独立采样的dropout掩码获得两个作为“正对”的嵌入。

- 对于**Unsupervised SimCSE**，核心在于如何生成dropout mask。因为BERT内部每次dropout都随机会生成一个不同的dropout mask。所以SimCSL不需要改变原始BERT，只需要将同一个句子喂给模型两次，得到的两个向量就是应用两次不同dropout mask的结果。然后将两个向量作为正例对。（真的simple）

> Although it may appear strikingly simple, we find that this approach largely outperforms training objectives such as predicting next sentences and common data augmentation techniques, e.g., word deletion and replacement.

- 虽然它看起来非常简单，但是我们发现这种方法在很大程度上优于训练目标，例如预测下一句话和常用的数据增强技术，例如单词删除和替换。

> More surprisingly, this unsupervised embedding method already matches all the previous supervised approaches. 

- 更令人惊讶的是，这种无监督的嵌入方法已经匹配了所有以前的监督方法。

> Through careful analysis, we find that dropout essentially acts as minimal data augmentation, while removing it leads to a representation collapse.

- 通过仔细的分析，我们发现退dropout本质上是作为最小的数据扩充，而删除它会导致表示崩溃。

**Unsupervised SimCSE** 引入dropout给输入加噪声，假设加噪后的输入仍与原始输入在语义空间距离相近。其正负例的构造方式如下：

> 正例：给定输入，用预训练语言模型编码两次得到的两个向量和作为正例对。我理解的正例就是大于平均值的数据，负例就是低于平均值的数据

> 负例：使用in-batch negatives的方式，即随机采样一个batch中另一个输入作为的负例。

----

## 有监督SimCSE

**Supervised SimCSE**，利用标注数据来构造对比学习的正负例子。为探究哪种标注数据更有利于句子向量的学习，文中在多种数据集上做了实验，最后发现NLI数据最有利于学习句子表示。下面以NLI数据为例介绍Supervised SimCSE的流程。

Supervised SimCSE 引入了NLI任务来监督对比学习过程。该模型假设如果两个句子存在蕴含关系，那么它们之间的句子向量距离应该较近；如果两个句子存在矛盾关系，那么它们的距离应该较远。因此NLI中的蕴含句对和矛盾句对分别对应对比学习中的正例对和负例对。所以在Supervised SimCSE中，正负例的构造方式如下:

> 正例：NLI中entailment关系样例对。负例：a) in-batch negatives b)NLI中关系为contradiction的样例对。

---

> In our *supervised* SimCSE, we build upon the recent success of leveraging natural language inference （NLI） datasets for sentence embeddings （Conneau et al., 2017; Reimers and Gurevych, 2019） and incorporate supervised sentence pairs in contrastive learning （Figure 1(b)）. 

- 在我们的有监督SimCSE中，我们建立在利用自然语言推理（NLI）数据集进行句子嵌入的最新成功基础上，并将有监督句对纳入对比学习（图1（b））。

<img src="Supervised SimCSE.png" alt="Supervised SimCSE" style="zoom:50%;" />

> Unlike previous work that casts it as a 3-way classification task （entailment/neutral/contradiction）, we take advantage of the fact that **entailment** pairs can be naturally used as positive instances.

- 与以前的工作不同的是，我们将它作为一个三向分类任务（蕴涵/中立/矛盾），我们利用了蕴涵对可以自然地用作正实例这一事实。



> We also find that adding corresponding contradiction pairs as hard negatives further improves performance. 

- 我们还发现，添加相应的矛盾对作为硬否定进一步提高了性能。



> This simple use of NLI datasets achieves a greater performance compared to prior methods using the same datasets. 

- 与以前使用相同数据集的方法相比，NLI数据集的这种简单使用实现了更高的性能。

> We also compare to other (annotated) sentence-pair datasets and find that NLI datasets are especially effective for learning sentence embeddings.

- 我们还比较了其他（带注释的）句子对数据集，发现NLI数据集对于学习句子嵌入特别有效。





> To better understand the superior performance of SimCSE, we borrow the analysis tool from Wang and Isola (2020), which takes *alignment* between semantically-related positive pairs and *uniformity* of the whole representation space to measure the quality of learned embeddings. 

- 为了更好地理解SimCSE的优越性能，我们借用了Wang和Isola（2020）的分析工具，它采用语义相关正对之间的对齐度和整个表示空间的一致度来衡量学习嵌入的质量。



> We prove that theoretically the contrastive learning objective “flattens” the singular value distribution of the sentence embedding space, hence improving the uniformity. 

- 我们从理论上证明了对比学习目标“平坦”了句子嵌入空间的奇异值分布，从而提高了一致性。





> We also draw a connection to the recent findings that pre-trained word embeddings suffer from anisotropy (Ethayarajh, 2019; Li et al., 2020).

- 我们还与最近的研究结果相联系，即预先训练的单词嵌入会受到各向异性的影响。





> We find that our unsupervised SimCSE essentially improves uniformity while avoiding degenerated alignment via dropout noise, thus greatly improves the expressiveness of the representations. 

- 我们发现，我们的无监督SimCSE本质上改善了一致性，同时避免了通过丢失噪声退化对齐，从而大大提高了表示的表达能力。



> We also demonstrate that the NLI training signal can further improve alignment between positive pairs and hence produce better sentence embeddings.

- 我们还证明，NLI训练信号可以进一步改善正对之间的对齐，从而产生更好的句子嵌入。







第三段



> We conduct a comprehensive evaluation of SimCSE, along with previous state-of-the-art models on 7 semantic textual similarity (STS) tasks and 7 transfer tasks.

- 我们对SimCSE进行了综合评价，并对7个语义-文本相似度（STS）任务和7个迁移任务进行了分析。



> On STS tasks, we show that our unsupervised and supervised models achieve a 74.5% and 81.6% averaged Spearman’s correlation respectively using BERTbase , largely outperforming previous best (Table 1). 

- 在STS任务中，我们发现我们的无监督和有监督模型使用BERTbase分别达到了74.5%和81.6%的平均Spearman相关性，在很大程度上优于以前的最佳（表1）。



> We also achieve competitive performance on the transfer tasks. Additionally, we identify an incoherent evaluation issue in existing work and consolidate results of different evaluation settings for future research.

- 我们在转移任务上也取得了有竞争力的表现。此外，我们在现有工作中发现了一个不连贯的评估问题，并将不同评估设置的结果进行了整合，以备将来研究之用。




------



