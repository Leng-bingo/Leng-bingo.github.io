---
title: SimCSE（二）
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
summary: SimCSE Abstract
abbrlink: 221e
date: 2021-04-27 14:07:32
password:
keywords:
description:
---
## Simple Contrastive Learning of Sentence Embeddings（二）

#### 单词预览

- contrastive，对比
- state-of-the-art，最先进的
- unsupervise，无监督
- predict，预测
- objective，目标
- contrastive，对比
- on par with，与……同一水平
- represent，代表，表示
- inspiration，灵感
- incorporate，包含
- annotate，注释
- pairs from，来自
- entailment，蕴含
- semantic，语义
- correlate，有关系的
- respective，分别，各自

-----

> 两端对齐并调整行间距: style="text-align:justify";line-height:1.8rem
>
> html斜体使用i标签

#### 英文摘要Abstract

<div style="text-align:justify;line-height:1.8rem">This paper presents SimCSE, a simple contrastive learning framework that greatly advances the state-of-the-art sentence embeddings. We first describe an unsupervised approach, which takes an input sentence and predicts <i>itself</i> in a contrastive objective, with only standard dropout used as noise. This simple method works surprisingly well, performing on par with previous supervised counterparts. We hypothesize that dropout acts as minimal data augmentation and removing it leads to a representation collapse. Then, we draw inspiration from the recent success of learning sentence embeddings from natural language inference（NLI）datasets and incorporate annotated pairs from NLI datasets into contrastive learning by using “entailment” pairs as pos- itives and “contradiction” pairs as hard negatives. We evaluate SimCSE on standard semantic textual similarity（STS）tasks, and our unsupervised and supervised models using BERTbase achieve an average of 74.5% and 81.6% Spearman’s correlation respectively, a 7.9 and 4.6 points improvement compared to previous best results. We also show that contrastive learning theoretically regularizes pretrained embeddings’ anisotropic space to be more uniform, and it better aligns positive pairs when supervised signals are available.</div>

-----

#### 详细翻译中文摘要

<div style="text-align:justify;line-height:1.8rem">文章主要介绍了SimCSE，一种简单对比学习框架，SimCSE对比学习框架极大的提高了最先进的句子嵌入技术。我们首先描述了一种无监督的方法，这种方法采用一个输入语句，并根据一个对比目标进行预测，仅使用标准的dropout作为噪声。这种简单方法非常好，表现的与以前的可监督方法水平不相上下。我们假设，dropout做为最小的数据增加和删除，它会导致表示崩溃。然后，我们从最近的自然语言推理（NLI）数据集的成功经验获得灵感，将来自NLI数据集中的注释合并到对比学习中，并使用“蕴含”作为正向，“矛盾”作为负向。我们在标准的语义-文本相似度（STS）任务中对SimSCE进行了评估，使用Bear的无监督和有监督模型的相关度平均达到74.5%和81.6%的Spearman相关度，与以前的最佳结果分别提高了7.9和4.6个点。我们也展示了相对学习理论使预训练嵌入的各向异性空间更加均匀，并在有监督信号的情况下可以更好的对其正向嵌入。</div>

