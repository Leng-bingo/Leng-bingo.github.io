---
title: 爬虫数据库bug
top: false
cover: false
toc: false
mathjax: true
tags:
  - bug
  - 爬虫
  - 项目
categories:
  - bug
summary: 爬虫数据存入数据库bug修改
abbrlink: 5fd7
date: 2021-05-17 13:32:24
password:
keywords:
description:
---
# 爬虫数据存入数据库bug修改

- 问题：微博所有数据爬取正常，但存入数据库的数据很少，有时还会重复获取同一时间段的文章，导致微博ID一样，无法存入数据库
- 解决：修改search.py的parse_by_hour为：

```python
    def parse_by_hour(self, response):
        """以小时为单位筛选"""
        keyword = response.meta.get('keyword')
        is_empty = response.xpath(
            '//div[@class="card card-no-result s-pt20b40"]')
        if is_empty:
            print('当前页面搜索结果为空')
        else:
            # 解析当前页面
            for weibo in self.parse_weibo(response):
                self.check_environment()
                yield weibo
            next_url = response.xpath(
                '//a[@class="next"]/@href').extract_first()
            if next_url:
                next_url = self.base_url + next_url
                yield scrapy.Request(url=next_url,
                                     callback=self.parse_page,
                                     meta={'keyword': keyword})
```

