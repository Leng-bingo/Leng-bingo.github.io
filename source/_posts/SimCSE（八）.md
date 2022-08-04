---
title: SimCSE（八）
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
summary: 代码环境配置
abbrlink: '4201'
date: 2021-05-04 15:35:14
password:
keywords:
description:
---
# SimCSE代码Mac配置

- 安装conda虚拟环境https://github.com/conda-forge/miniforge/#download

- ```shell
  bash Miniforge3-MacOSX-arm64.sh
  ```

- 配置环境变量

- ```shell
  vim ~/.bash_profile
  export PATH="/Users/leng/miniforge3/bin:$PATH"
  #刷新变量
  source $HOME/.bash_profile
  ```

- 创建python虚拟环境

- ```shell
  conda create -n py38 python=3.8
  # 激活环境
  source activate
  # 打开虚拟环境
  conda activate py38
  # 退出环境
  source deactivate
  ```

- 创造Virtualenv虚拟环境

- ```shell
  # 在项目目录下生成venv目录
  python -m venv venv
  # 将下载好的安装脚本放在venv同级目录下，安装
  bash download_and_install.sh
  # 输入虚拟环境的路径，注意要以venv结尾
  ```

  

