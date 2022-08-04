---
title: tensorflow（一）
top: false
cover: false
toc: false
mathjax: true
tags:
  - 研究生
  - tensorflow
  - NLP
  - 神经网络
categories:
  - tensorflow
summary: tensorflow模型建立训练及预测
abbrlink: bd21
date: 2021-05-07 19:12:57
password:
keywords:
description:
---
# tensorflow入门（一）

数据集和py文件 [训练模型及使用.py](https://lengblog.oss-cn-hangzhou.aliyuncs.com/blog%E6%95%B0%E6%8D%AE%E6%96%87%E4%BB%B6/tensorflow%EF%BC%88%E4%B8%80%EF%BC%89/%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E5%8F%8A%E4%BD%BF%E7%94%A8.py) ， [Sarcasm_Headlines_Dataset.json](https://lengblog.oss-cn-hangzhou.aliyuncs.com/blog%E6%95%B0%E6%8D%AE%E6%96%87%E4%BB%B6/tensorflow%EF%BC%88%E4%B8%80%EF%BC%89/Sarcasm_Headlines_Dataset.json)  ，[sarcasm.json](https://lengblog.oss-cn-hangzhou.aliyuncs.com/blog%E6%95%B0%E6%8D%AE%E6%96%87%E4%BB%B6/tensorflow%EF%BC%88%E4%B8%80%EF%BC%89/sarcasm.json) 

----

1. 初始定义参数

   ```python
   # 分词保留的最大单词数
   vocab_size = 10000
   # 每个词创建的向量维度
   embedding_dim = 16
   # 数据最大长度
   max_length = 100
   # 数据长度大于所设定句子长度maxlen，可加入截断参数，truncating='post'，
   trunc_type = 'post'
   # 在句子后边补齐
   padding_type = 'post'
   # oov_token属性，将语料库中没有的单词标记出来
   oov_tok = "<OOV>"
   # 两万训练数据
   training_size = 20000
   ```


2. 从本地读取json文件

- json.load()和json.loads()都实现了反序列化，变量内容从序列化的对象重新读到内存里称之为反序列化，反序列化是流转换为对象。也就是由json（双引号）转变为dict（键值对用单引号包起来的就是dict）字典可供python操作。

  - load：针对文件句柄，将json格式的字符转换为dict，从文件中读取（将string转换为dict）

    `a_json = json.load(open('demo.json','r'))`

  - loads：针对内存对象，将string转化为dict

    `a = json.loads('{'a':'1111','b':'2222'}')`

- json.dump()和json.dumps()都实现了序列化，变量从内存中变成可存储或传输的过程称之为序列化，序列化是将对象状态转化为可保存或可传输格式的过程。

  - dump：将dict类型转化为json字符串格式，写入到文件**（易存储）**

  ```python
  a_dict = {'a':'1111','b':'2222'}
  json.dump(a_dict, open('demo.json', 'w'))
  ```

  - dumps：将dict转化为string**（易传输）**

  ```python
  a_dict = {'a':'1111','b':'2222'}
  a_str = json.dumps(a_dict)
  ```

  

- 方法一：适用于少量数据规整，标准json格式

  ```python
  with open("data/sarcasm.json", 'r') as f:
    	# json.load将json转化为python可操作的dict字典
  		datastore = json.load(f)
        
  # 定义空数组      
  sentences = []
  labels = []
  urls = []
  # json数据样式
  # {"article_link": "https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5", "headline": "former versace store clerk sues over secret 'black code' for minority shoppers", "is_sarcastic": 0},
  # 循环将数据插入格式化
  for item in datastore:
    	#append，字符串插入操作
    	sentences.append(item['headline'])
    	labels.append(item['is_sarcastic'])
  		urls.append(item['article_link'])
  ```

- 方法二：适用于大量冗余数据

  ```python
  # 先定义打开文件操作，这个文件不标准，不是标准json，中间没有逗号
  file = open("data/Sarcasm_Headlines_Dataset.json", 'r')
  
  # 定义空数组      
  sentences = []
  labels = []
  urls = []
  
  # json数据样式
  # {"article_link": "https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5", "headline": "former versace store clerk sues over secret 'black code' for minority shoppers", "is_sarcastic": 0},
  
  # 调用readlines()方法逐行扫描json并插入数组，规范化数据
  for line in file.readlines():
    	# 将每次扫描到的那一行转化为dict数据，然后再插入到数组中
      datastore = json.loads(line)
      sentences.append(datastore['headline'])
      labels.append(datastore['is_sarcastic'])
      # urls.append(datastore['article_link'])
  ```

2. 进行分片，使训练集和测试集互不影响

   ```python
   training_sentences = sentences[0:training_size]
   testing_sentences = sentences[training_size:]
   training_labels = labels[0:training_size]
   testing_labels = labels[training_size:]
   ```

3. 实例化分词器，然后用训练集training_sentences创建字典并对应编号。再生成每个训练集句子的对应数字编号所组成的每个句子的token序列，然后设置训练集数据格式，最大长度，位数不够时补零方式以及位数超出时截断方式

   ```python
   # 要保证创建的神经网络只见过训练集，没有见过训练集
   # 分词保留的最大单词数，为了显示语料库中没有的单词，利用oov_token属性，将其设置为语料库中无法识别的内容显示为1
   # 实例化分词器
   tokenizer = Tokenizer(num_words=vocab_size,
                         oov_token=oov_tok)
   # 用这个训练集创建token字典
   tokenizer.fit_on_texts(training_sentences)
   # 分词并编号
   word_index = tokenizer.word_index
   
   # 每个句子的token序列
   training_sequences = tokenizer.texts_to_sequences(training_sentences)
   # 填充序列变为同样长度，以最长的为基准
   # 如果添加参数 padding='post'，代表在token序列后方加入0补齐
   # 也可以加入参数maxlen，指定所需句子长度，如 maxlen=5
   # 如果数据长度大于所设定句子长度maxlen，可加入截断参数，post从后面截断，默认是pre，truncating='post'，
   training_padded = pad_sequences(training_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
   
   testing_sequences =
   tokenizer.texts_to_sequences(testing_sentences)
   testing_padded = pad_sequences(testing_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
   ```

   

4. 引入科学计算库numpy，训练模型中的训练数据只支持numpy数组

   ```python
   # numpy一个科学计算库
   import numpy as np
   
   # np.array将数组等其他形式转化为numpy数组，以便进行矩阵等运算
   training_padded_after = np.array(training_padded)
   training_labels_after = np.array(training_labels)
   testing_padded_after = np.array(testing_padded)
   testing_labels_after = np.array(testing_labels)
   ```

   

5. 建立模型

   ```python
   # 建立模型
   model = tf.keras.Sequential([
       # 每个单词的情感方向都会被一次又一次的学习
       tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
       # 将向量进行相加
       tf.keras.layers.GlobalAveragePooling1D(),
   
       # 然后嵌入到一个普通的神经网络中，24输出维度大小units（该层有多少个神经元），activation激活函数
       # 输入层和隐层，输出层等等,下面Dense就是神经网络的两层
       # https://blog.csdn.net/ybdesire/article/details/85217688
       tf.keras.layers.Dense(24, activation='relu'),
       tf.keras.layers.Dense(1, activation='sigmoid')
   ])
   # 查看https://blog.csdn.net/chaojishuai123/article/details/114580892
   # loss目标函数，'binary_crossentropy'对数损失；metrics评价函数（acc和val_acc就是通过定义评价函数得到的）
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   
   # 通过model.summary()输出模型各层的参数情况
   model.summary()
   ```

   

6. 输入训练数据以及测试数据进行模型的训练以及评估

   ```python
   # 30次迭代训练
   num_epochs = 30
   # 训练以及检测
   # training_padded训练集输入特征，training_labels训练集标签
   # epochs迭代次数，validation_data = (测试集的输入特征，测试集的标签），verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
   # https://blog.csdn.net/LuYi_WeiLin/article/details/88555813
   # numpy格式数组
   history = model.fit(training_padded_after, training_labels_after,epochs=num_epochs,validation_data=(testing_padded_after, testing_labels_after),verbose=2)
   
   #训练以及评估结果
   Epoch 30/30
   625/625 - 0s - loss: 0.0199 - accuracy: 0.9945 - val_loss: 1.1787 - val_accuracy: 0.8109
   ```

   ![模型情况](模型情况.png)

7. matlabplot画图

   ```python
   import matplotlib.pyplot as plt
   
   # 画图
   def plot_graphs(history, string):
       plt.plot(history.history[string])
       plt.plot(history.history['val_' + string])
       plt.xlabel("Epochs")
       plt.ylabel(string)
       plt.legend([string, 'val_' + string])
       plt.show()
   
   plot_graphs(history, "accuracy")
   plot_graphs(history, "loss")
   ```

   ![accuracy](下载1.png)

   ![loss](下载2.png)

8. 转化key和value，返回该层的权重，有多少个神经元，可以不写代码里

   ```python
   # 将value和key转化，可以互相反查
   reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
   
   def decode_sentence(text):
       # 字典dict.get,返回指定键i的值，如果没有返回'？'。
       # join用法，把？，用空格连接
       return ' '.join([reverse_word_index.get(i, '?') for i in text])
   
   # 测试一下对不对
   print('training_padded[0]为：',training_padded[0])
   print(decode_sentence(training_padded[0]))
   print(training_sentences[2])
   print(labels[2])
   
   e = model.layers[0]
   # 返回该层的权重
   weights = e.get_weights()[0]
   print(weights.shape)  # shape: (vocab_size, embedding_dim)
   ```

   

9. 利用训练好的模型，传入新句子，判断情感色彩

   ```python
   # 利用神经网络，判断新句子的情感色彩
   new_sentence = ["granny starting to fear spiders in the garden might be real",
                   "game of thrones season finale showing this sunday night"]
   # 求出新句子的token序列，分词编号用的是训练集
   new_sequences = tokenizer.texts_to_sequences(new_sentence)
   # 规范化序列
   padded = pad_sequences(new_sequences, maxlen=max_length,
                          padding=padding_type,
                          truncating=trunc_type)
   # 输出预测结果
   print(model.predict(padded))
   # 结果 [[9.8276615e-01] [3.8519531e-04]]，表示第一句话负面性概率为98.2%，第二句负面性为38.5%
   ```

   

