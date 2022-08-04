---
title: SIFT代码（一）
top: false
cover: false
toc: false
mathjax: true
tags:
  - 论文
  - 图像处理
  - 大津法
  - SIFT
categories:
  - 代码
summary: 手写rgb转灰度，大津法灰度转二值图像
abbrlink: b1d3
date: 2021-08-31 18:25:41
password:
keywords:
description:
---
## SIFT代码（一）

#### 不调用库，手写rbg转灰度，利用大津法灰度转二值图像

![原始图像](https://leng-mypic.oss-cn-beijing.aliyuncs.com/%E7%AC%94%E8%BF%B9%E5%9B%BE%E5%83%8F.png)

![灰度图像](https://leng-mypic.oss-cn-beijing.aliyuncs.com/20210831182941.png)

![大津法二值图像](https://leng-mypic.oss-cn-beijing.aliyuncs.com/20210831183030.png)


```python
import cv2
import numpy as np

class SIFT(object):
    def __init__(self, img):
        """
        :param img:  图像路径
        """
        self.img = img
        self.rgbToray(img)
        # self.openConvertImg(img)

    # 将彩色图片转化为灰度图片,调用库
    def openConvertImg(self, img):
        L = img.convert('L')
        L.show()
        return L

    # 将RGB转化为灰度图片，手写
    def rgbToray(self, img):
        """

        :param img:
        :return: gray_img，灰度图像
        astype是numpy中类型转换的方式，括号中是要变为的格式
        读入图片的格式是uint8，进行计算处理时转化为float32，算完了再转回来
        """
        img_float = cv2.imread(img).astype(np.float32)
        b = img_float[:, :, 0].copy()
        g = img_float[:, :, 1].copy()
        r = img_float[:, :, 2].copy()
        gray_img = 0.2126 * r + 0.7152 * g + 0.0722 * b
        self.otus(gray_img)
        return gray_img

    # 显示图片
    def showImg(self, img):
        cv2.imshow('img', img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    # 大津法二值化图像
    def otus(self, gray_img):
        """

        h: 图片高度
        w：图片宽度
        threshold_t： 阈值t
        max_g：最大类间方差
        n0: 小于阈值t的像素，前景
        n1：大于阈值t的像素，背景
        n0 + n1 = h * w ：总像素个数

        w0：前景像素数量占总像素数量的比例，w0 = n0 / (h * w)
        w1：背景像素数量占总像素数量的比例，w1 = n1 / (h * w)
        w0 + w1 == 1

        u0：前景平均灰度，u0 = n0灰度累加和 / n0
        u1：背景平均灰度，u1 = n1灰度累加和 / n1

        u：平均灰度， u = (n0灰度累加和 + n1灰度累加和) / (h * w)
        u = w0 * u0 + w1 * u1

        g：类间方差（那个灰度的g最大，哪个灰度就是需要的阈值t）
        g = w0 * (u0 - u)^2 + w1 * (u1 - u)^2
        根据上面的关系，可以推出：（这个一步一步推导就可以得到）
        g = w0 * w1 * (u0 - u1) ^ 2
        """

        h = gray_img.shape[0]
        w = gray_img.shape[1]
        threshold_t = 0
        max_g = 0

        # 遍历每一个灰度层
        for t in range(255):
            n0 = gray_img[np.where(gray_img < t)]
            n1 = gray_img[np.where(gray_img >= t)]
            w0 = len(n0) / (h * w)
            w1 = len(n1) / (h * w)
            u0 = np.mean(n0) if len(n0) > 0 else 0
            u1 = np.mean(n1) if len(n1) > 0 else 0

            g = w0 * w1 * (u0 - u1) ** 2
            if g > max_g:
                max_g = g
                threshold_t = t
        print('类间方差最大阈值：', threshold_t)
        gray_img[gray_img < threshold_t] = 0
        gray_img[gray_img > threshold_t] = 255
        # 显示图像
        self.showImg(gray_img)
        return gray_img


imgUrl = '笔迹图像.png'
SIFT(imgUrl)

```


