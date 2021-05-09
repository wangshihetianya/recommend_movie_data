# 电影推荐系统(大数据部分)

- HDFS保存离线训练所需训练样本和特征
- Spark进行批量的特征处理，例如one-hot编码和Embedding

## 环境搭建

这部分需要的环境有：

- Java 8

- Scala 2.11

- hadoop 2.7.1

- spark 2.3.0

​        为了模拟分布式的情景，我们选择`docker-compose`来搭建spark-hadoop集群，从而尽可能"真实地"分布式地存储和处理数据(因此还需要`docker desktop`)。

集群的搭建具体参考了[这篇文章](https://cloud.tencent.com/developer/article/1438344)。