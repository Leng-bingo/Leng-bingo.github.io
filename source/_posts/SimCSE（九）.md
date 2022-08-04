---
title: SimCSE（九）
top: false
cover: false
toc: false
mathjax: true
tags:
  - 论文
  - 知识图谱
  - SimCSE
  - 代码
categories:
  - 论文
summary: 初步代码运行
abbrlink: 1e5f
date: 2021-05-05 15:41:25
password:
keywords:
description:
---
# SimCSE运行

# 运行结果

<img src="训练结果.png" alt="训练结果" style="zoom:67%;" />

---

- 先安装好torch和所需库

- 先下载评估数据，并且要安装wegt以便运行bash语句

- ```sh
  cd SentEval/data/downstream/
  bash download_dataset.sh
  ```

- **经向作者发邮件咨询如何运行以及参数调试**

- ```python
  python evaluation.py --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased
  ```

- 运行语句设置

- ```python
  python evaluation.py \
      --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
      --pooler cls \
      --task_set sts \
      --mode test
  ```

## Evaluation

Arguments for the evaluation script are as follows,

- `--model_name_or_path`: The name or path of a `transformers`-based pre-trained checkpoint. You can directly use the models in the above table, e.g., `princeton-nlp/sup-simcse-bert-base-uncased`.
- `--pooler`: Pooling method. Now we support
  - `cls` (default): Use the representation of `[CLS]` token. A linear+activation layer is applied after the representation (it's in the standard BERT implementation). If you use SimCSE, you should use this option.
  - `cls_before_pooler`: Use the representation of `[CLS]` token without the extra linear+activation.
  - `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
  - `avg_top2`: Average embeddings of the last two layers.
  - `avg_first_last`: Average embeddings of the first and last layers. If you use vanilla BERT or RoBERTa, this works the best.
- `--mode`: Evaluation mode
  - `test` (default): The default test mode. To faithfully reproduce our results, you should use this option.
  - `dev`: Report the development set results. Note that in STS tasks, only `STS-B` and `SICK-R` have development sets, so we only report their numbers. It also takes a fast mode for transfer tasks, so the running time is much shorter than the `test` mode (though numbers are slightly lower).
  - `fasttest`: It is the same as `test`, but with a fast mode so the running time is much shorter, but the reported numbers may be lower (only for transfer tasks).
- `--task_set`: What set of tasks to evaluate on (if set, it will override `--tasks`)
  - `sts` (default): Evaluate on STS tasks, including `STS 12~16`, `STS-B` and `SICK-R`. This is the most commonly-used set of tasks to evaluate the quality of sentence embeddings.
  - `transfer`: Evaluate on transfer tasks.
  - `full`: Evaluate on both STS and transfer tasks.
  - `na`: Manually set tasks by `--tasks`.
- `--tasks`: Specify which dataset(s) to evaluate on. Will be overridden if `--task_set` is not `na`. See the code for a full list of tasks.

