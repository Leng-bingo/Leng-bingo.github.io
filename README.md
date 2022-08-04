- Hexo博客代码源码

```git
git remote add origin git@github.com:Leng-bingo/Leng-Blog.git
需要使用的如下三行
git add .
git commit -m "你的注释...." 
git push -u origin main
```

- 测试
- 测试一次
- 111

- git基本操作

  ```
  git init                                                //初始化本地仓库
  git add README.md                                       //添加刚刚创建的README文档
  git commit -m "你的注释...."                             //提交到本地仓库，并写一些注释
  git remote add origin git@github.com:Leng-bingo/myBlog.git  //连接远程仓库并建了一个名叫：origin的别名，当然可以为其他名字，但是origin一看就知道是别名，youname记得替换成你的用户名
  git push -u origin master                             //将本地仓库的文件提交到别名为origin的地址的main分支下，-u为第一次提交，需要创建master分支，下次就不需要了
  git push origin master
  ```

- 分支操作

- ```
  git branch test //创建分支test
  git branch //查看分支都有什么
  git checkout test //切换到分支test
  git add .
  git commit -m "提交信息备注"
  git status //查看状态
  git checkout master
  git merge test //将分支上的改动合并到主分支上
  git push -u origin master
  git branch -D test //删除分支
  ```

- 正常操作

- ```
  git add .
  git commit -m "注释"
  git push origin master
  ```

- 删除远程分支

- ```
  git push origin --delete temp
  ```

  