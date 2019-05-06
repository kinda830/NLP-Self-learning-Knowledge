---
typora-root-url: E:\电子书\小象AI班\1. 基础知识\LDA
---

# LDA 知识点

​	LDA：Latent Dirichlet Allocation，topic model

![topic model](/../../../../../LDA/topic model.png)

1. 由 Dirichlet 分布生成文档的 topic 分布，即 \Theta
2. Z 则是一个隐含变量，对应文档中每个单词属于哪个topic，即从 \theta （topic分布）采样生成；
3. \beta 是对应每一个 topic 分布下词的概率分布；
4. 最后先由 Z 采用当前词的 topic，然后再通过 \beta 中获取对应 topic 中的词的分布，最后采样生成对应单词；