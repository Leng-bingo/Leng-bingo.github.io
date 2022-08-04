---
title: SimCSE（七）
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
summary: Connection to Anisotropy
abbrlink: f664
date: 2021-05-04 15:30:08
password:
keywords:
description:
---
# Simple Contrastive Learning of Sentence Embeddings（七）

### 5 Connection to Anisotropy 各向异性连接

#### <font color = "red">还需学习的知识点</font>

- anisotropy（各向异性）是什么意思
- isotropic distribution 各向同性分布
- 奇异值singular是什么东西
- 这些公式都要再看一下

---



> Recent work identifies an anisotropy problem in language representations (Ethayarajh, 2019; Li et al., 2020), i.e., the learned embeddings occupy a narrow cone in the vector space, which largely limits their expressiveness. 

- 最近的研究发现了语言表征中的“各向异性”问题，即学习到的嵌入在向量空间中占据了一个狭窄的圆锥体，这在很大程度上限制了它们的表达能力。

> Gao et al. (2019) term it as a *representation degeneration* problem and demonstrate that language models trained with tied input/output embeddings lead to anisotropic word embeddings, and this is further observed by Ethayarajh (2019) in pretrained contextual embeddings. 

- Gao将其称为“表征退化”问题，并证明使用捆绑输入/输出嵌入训练的语言模型会导致各向异性的单词嵌入，Ethayarajh在预先训练的上下文嵌入中进一步观察到了这一点。

> Wang et al. (2020) show that the singular values of the word embedding matrix decay drastically. In other words, except for a few dominating singular values, all others are close to zero.

- Wang证明单词嵌入矩阵的奇异值急剧衰减。换言之，除了少数占主导地位的奇异值外，其他所有奇异值都接近于零。

> A simple way to alleviate the problem is postprocessing, either to eliminate the dominant principal components (Arora et al., 2017; Mu and Viswanath, 2018), or to map embeddings to an isotropic distribution (Li et al., 2020; Su et al., 2021). 

- 缓解问题的一个简单方法是后处理，即消除主要的主成分，或将嵌入映射到各向同性分布。

> Alternatively, one can add regularization during training (Gao et al., 2019; Wang et al., 2020). 

- 或者，可以在训练期间添加正则化。

> In this section, we show that the contrastive objective can inherently “flatten” the singular value distribution of the sentence-embedding matrix.

- 在这一节中，我们证明了对比目标可以内在地“平坦”句子嵌入矩阵的奇异值分布。

> Following Wang and Isola (2020), the asymptotics of the contrastive learning objective can be expressed by the following equation when the number of negative instances approaches infinity (assuming f(x) is normalized):

- 在Wang和Isola的基础上，对比学习目标的渐近性可以用以下等式表示，当负实例的数量接近无穷大时（假设f（x）是标准化的）：

$$
\begin{array}{l}
-\frac{1}{\tau} \underset{\left(x, x^{+}\right) \sim p_{\mathrm{pos}}}{\mathbb{E}}\left[f(x)^{\top} f\left(x^{+}\right)\right] 
+\underset{x \sim p_{\text {data }}}{\mathbb{E}}\left[\log \underset{x^{-} \sim p_{\text {data }}}{\mathbb{E}}\left[e^{f(x)^{\top} f\left(x^{-}\right) / \tau}\right]\right]
\end{array}
$$

> where the first term keeps positive instances similar and the second pushes negative pairs apart. When $p_{data}$ is uniform over finite samples $\{x_i\}_{i=1}^m$ , with $h_i=f(x)$, we can derive the following formula from the second term with Jensen’s inequality:

- 其中第一项保持正的实例相似，第二项将负的对分开。当$p_{data}$在有限样本$\{x_i\}_{i=1}^m$上是一致的，且$h_i=f（x）$，我们可以从第二项和詹森不等式导出以下公式：

$$
\begin{aligned}
& \underset{x \sim p_{\text {data }}}{\mathbb{E}}\left[\log \underset{x^{-} \sim p_{\text {data }}}{\mathbb{E}}\left[e^{f(x)^{\top} f\left(x^{-}\right) / \tau}\right]\right] \\
=& \frac{1}{m} \sum_{i=1}^{m} \log \left(\frac{1}{m} \sum_{j=1}^{m} e^{\mathbf{h}_{i}^{\top} \mathbf{h}_{j} / \tau}\right) \\
\geq & \frac{1}{\tau m^{2}} \sum_{i=1}^{m} \sum_{j=1}^{m} \mathbf{h}_{i}^{\top} \mathbf{h}_{j}
\end{aligned}
$$

> Let $W$ be the sentence embedding matrix corresponding to $\{x_i\}_{i=1}^m$, i.e., the i-th row of $W$ is $h_i$. 

- 设$W$为$\{x_i\}_{i=1}^m$对应的句子嵌入矩阵，即W的第i行为$h_i$。

> Ignoring the constant terms, optimizing the second term in Eq essentially minimizes an upper bound of the summation of all elements in $WW^T$, i.e., $\operatorname{Sum}\left(\mathbf{W} \mathbf{W}^{\top}\right)=\sum_{i=1}^{m} \sum_{j=1}^{m} \mathbf{h}_{i}^{\top} \mathbf{h}_{j}$

- 忽略常量项，优化了等式1中的第二项实质上最小化了$WW^T$中所有元素之和的上限，即$\operatorname{Sum}\left(\mathbf{W} \mathbf{W}^{\top}\right)=\sum_{i=1}^{m} \sum_{j=1}^{m} \mathbf{h}_{i}^{\top} \mathbf{h}_{j}$



> Since we normalize $h_i$, all elements on the diagonal of $WW^T$ are 1 and then $tr(WW^T)$, also the sum of all eigenvalues, is a constant. 

- 因为我们规范化了$h_i$，所以$WW^T$对角线上的所有元素都是1，$tr（WW^T）$，也是所有特征值的总和，是一个常数。

> According to Merikoski (1984), if all elements in $WW^T$ are positive, which is the case in most times from Gao et al. (2019), then $Sum(WW^T)$ is an upper bound for the largest eigenvalue of $WW^T$. 

- 根据Merikoski，如果$WW^T$中的所有元素都是正的，这在Gao的大多数情况下都是这样，那么$Sum（WW^T）$是最大特征值$WW^T$的上界。

> Therefore, when minimizing the second term in Eq1, we are reducing the top eigenvalue of $WW^T$ and inherently “flattening” the singular spectrum of the embedding space.

- 因此，当最小化Eq1中的第二项时，我们减少了$WW^T$的顶部特征值，并且固有地“平坦”了嵌入空间的奇异谱。

> Hence contrastive learning can potentially tackle the representation degeneration problem and improve the uniformity.

- 因此，对比学习有可能解决表征退化问题，提高一致性。

> Compared to postprocessing methods in Li et al. (2020); Su et al. (2021), which only aim to encourage isotropic representations, contrastive learning also optimizes for aligning positive pairs by the first term in Eq. 6, which is the key to the success of SimCSE (a quantitative analysis is given in §7).

- 与Li等人（2020）的后处理方法相比；Su等人（2021年）只致力于鼓励各向同性表征，对比学习还优化了等式1中第一项的正对对齐，这是SimCSE成功的关键（定量分析见第七章表1）.

---



