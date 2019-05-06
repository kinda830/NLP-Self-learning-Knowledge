# seq2seq 模型知识点

## RNN 结构

​	根据输出和输入序列不同数量 RNN 可以有多种不同的结构，不同结构自然就有不同的引用场合。如下图：

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170901184911057?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FuZ3lhbmd6aGl6aG91/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- **one to one 结构**：仅仅只是简单的给一个输入得到一个输出，此处并未体现序列的特征，例如图像分类场景
- **one to many 结构**：给一个输入得到一系列输出，这种结构可用于生产图片描述的场景
- **many to one 结构**：给一系列输入得到一个输出，这种结构可用于文本情感分析，对一系列的文本输入进行分类，看是消极还是积极情感
- **many to many 结构**：给一系列输入得到一系列输出，这种结构可用于翻译或聊天对话场景，对输入的文本转换成另外一系列文本
- **同步 many to many 结构**：它是经典的 RNN 结构，前一输入的状态会带到下一个状态中，而且每一个输入都会对应一个输出，我们最熟悉的就是用于字符预测了，同样也可以用于视频分类，对视频的帧打标签

## seq2seq 定义

​	在上述 many to many 的两种模型中，上图可以看到第四种和第五种是有差异的，经典的 RNN 结构的输入和输出序列必须要是等长，它的应用场景也比较有限。而第四种它可以是输入和输出序列不等长，这种模型便是 seq2seq 模型，即Sequence to Sequence。经典的 RNN 模型固定了输入序列和输出序列的大小，而 seq2seq 模型则突破了该限制。

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170827161652677?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FuZ3lhbmd6aGl6aG91/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170905150828217?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FuZ3lhbmd6aGl6aG91/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

> 备注：其实对于 seq2seq 的 decoder，它在训练阶段和预测阶段对 RNN 的输出处理可能是不一样的，比如在训练阶段可能对 RNN 的输出不处理，直接用 target 的序列作为下一时刻的输入，如上图一；而预测阶段会将 RNN 的输出当成是下一时刻的输入，因为此时已经没有 target 序列可以作为输入了，如上图二。

## Encoder - Decoder 结构

​	seq2seq 属于 encoder - decoder 结构的一种，基本思想就是利用两个 RNN，一个 RNN 作为 encoder，另一个 RNN 作为 decoder，如下图所示：

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20171201205617129?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZ3ptZnh5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

Encoder：负责将输入序列压缩成指定长度的向量，这个向量就可以看成是这个序列的语义，这个过程称为编码；

Decoder：负责根据语义向量生成制定的序列，这个过程也称为解码。其中 decoder 有两种实现方式：

1. 将 encoder 得到的语义变量作为初始状态输入到 decoder 的 RNN 中，得到输出序列：

   ![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170905153318191?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FuZ3lhbmd6aGl6aG91/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

2. 将 encoder 得到的语义变量语义向量C参与序列所有时刻的运算，如下图，上一时刻的输出仍然作为当前时刻的输入，但语义向量C会参与所有时刻的运算：

   ![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170905153734454?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2FuZ3lhbmd6aGl6aG91/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)