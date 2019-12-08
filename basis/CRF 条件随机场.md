# CRF 条件随机场

## 1. 概念

​	CRF （条件随机场）主要应用于序列标注问题，可以简单理解为是给序列中的每一帧都进行分类。那么我们接下来通过两个实现方式来了解 CRF 的原理。

## 2. 模型对比

### 2.1. 逐帧 softmax

​	这种实现方式为将这个序列用 CNN 或 RNN 进行编码后，对序列的每一帧的输出都通过一个 softmax 函数来激活，如下图所示：

![1527042482806](CRF\1527042482806.png)

从上图可以看到，逐帧 softmax 的实现方式并没有直接考虑输出的上下文关联。

### 2.2. 条件随机场

​	但是，当我们设计标签时，比如用s、b、m、e 的 4 个标签来做标注法的分词，目标是输出序列本身会带有一些上下文关联，比如 s 后面就不能接 m 和 e，等等。逐标签 softmax 并没有考虑这种输出层面的上下文关联，所以它意味着把这些关联放到了编码层面，希望模型能自己学到这些内容，但有时候会“强模型所难”。

​	而 CRF 则更直接一点，它将输出层面的关联分离了出来，这使得模型在学习上更为“从容”：

![1527042479585](CRF\1527042479585.png)

CRF 在输出端显示地考虑了上下文关联。

## 3. CRF 的原理

​	如果仅仅是引入输出的关联，还不仅仅是 CRF 的全部，CRF 的真正精巧的地方，是它以路径为单位，考虑的是路径的概率。

## 4. 模型概要

​	假如一个输入有 n 帧，每一帧的标签有 k 中可能性，那么理论上就有  k^n  中不同的输入。我们可以将它用如下的网络图进行简单的可视化。在下图中，每个点代表一个标签的可能性，点之间的连线表示标签之间的关联，而每一种标注结果，都对应着图上的一条完整的路径。

![1527042479907](CRF\1527042479907.png)

​                                                                        4 tag分词模型中输出网络图

​	而在序列标注任务中，我们的正确答案一般是唯一的。比如“今天天气不错”，如果对应的分析结果是“今天/天气/不/错”，那么目标输出序列就是 bebess ，除此之外别的路径都不符合要求。

​	换言之，在序列标注任务中，我们的研究的基本单位应该是路径，我们要做的事情，是从 k * n 条路径选出正确的一条，那就意味着，如果将它视为一个分类问题，那么将是 k^n 条路径选出正确的一条，那就意味着，如果将它视为一个分类问题，那么将是  k^n  类中选一类的分类问题。

​	这就是逐帧 softmax 和 CRF 的根本不同了：前者将序列标注看成是 n 个 k 分类问题，后者将序列标注看成是 1 个  k^n 分类问题。

​	具体来讲，在 CRF 的序列标注问题中，我们要计算的是条件概率：
$$
P(y_1,...,y_n|x_1,...,x_n) = P(y_1,...,y_n|X), x = (x_1,...,x_n)（1）
$$

## 5. CRF 的两个假设

​	为了得到上述概率的估计，CRF 做了两个假设：

**假设一：该分布式指数族分布：**

​	这个假设意味着存在函数 $$ f(y_1,...,y_n) $$，使得：
$$
P、(y_1,...,y_n|X)=\frac1{Z(x)}exp(f(y_1,\cdots,y_n;x))（2）
$$

​	其中 Z(x) 是归一化因子，因为这个是条件分布，所以归一化因子跟 x 有关。这个 f 函数可以视为一个打分函数，打分函数取指数并归一化后就得到概率分布。

**假设二：输出之间的关联仅发生在相邻位置，并且关联是指数加性的。**

​	这个假设意味着 $$ f(y_1,\cdots,y_n;x) $$ 可以更进一步简化为：
$$
f(y_1,\cdots,y_n;x) = h(y_1;x)+g(y_1,y_2;x)+h(y_2;x)+\cdots+g(y_{n-1},y_n;x) + h(y_n;x)   （3）
$$
​	这也就是说，现在我们只需要对每一个标签和每一个相邻标签对分别打分，然后将所有打分结果求和得到总分。

### 5.1. 线性链CRF

​	尽管已经做了大量简化，但一般来说，（3）式所表示的概率模型还是过于复杂，难以求解。于是考虑到当前深度学习模型中。RNN 或者层叠 CNN 等模型已经能够比较充分捕捉各个 y 与输出 x 的联系，因此，我们不妨考虑函数 g 跟 x 无关，那么：
$$
f(y_1,\cdots,y_n;x) = h(y_1;x)+g(y_1,y_2)+h(y_2;x)+\cdots+g(y_{n-1},y_n)+h(y_n;x)（4）
$$
这时候 g 实际上就是一个有限的、待训练的参数矩阵而已，而单标签的打分函数 $$ h(y_i;x) $$ 我们可以通过 RNN 或者 CNN 来建模。因此，该模型是可以建立的，其中概率分布变为：
$$
P(y_1,\cdots,y_n|x)=\frac1{Z(x)}exp(h(y_1;x)+\sum^{n-1}_{k=1}g(y_k, y_{k+1})+h(y_{k+1};x)) （5）
$$
这就是线性链 CRF 的概念。

### 5.2. 归一化因子

​	为了训练 CRF 模型，我们用最大似然方法，也就是用：
$$
-logP(y_1,\cdots,y_n;x) （6）
$$
作为损失函数，可以算出它等于：
$$
-(h(y_1;x)+\sum^{n-1}_{k=1}g(y_k,y_{k+1})+h(y_{k+1};x))+logZ(x)（7）
$$
其中第一项是原来概率式的分子的对数，它目标的序列的打分，虽然它看上去挺迂回的，但是不难计算。真正的难度在于分母的对数 logZ(x) 这一项。

归一化因子，在物理上也叫配分函数，在这里它需要我们对所有可能的路径的打分进行指数求和，而我们前面已经说到，这样的路径数是指数量级的（k^n ) ，因此直接来算几乎是不可能的。

事实上，归一化因子难算，几乎是所有概率图模型的公共难题。幸运的是，在 CRF 模型中，由于我们只考虑了临近标签的联系（马尔可夫假设），因此我们可以递归地算出归一化因子，这使得原来是指数级的计算量降低为线性级别。

具体来说，我们将计算到时刻 t 的归一化因子记为 Zt，并将它分为 k 个部分：
$$
Z_t = Z_t^{(1)} + Z_t^{(2)}+\cdots+Z_t^{(k)}（8）
$$
其中
$$
Z_t^{(1)},\cdots,Z_t^{(k)}
$$
分别是截止到当前时刻 t 中，以标签 1, ... , k 为终点的所有路径的得分指数和。那么，我们可以递归地计算：
$$
Z_{t+1}^{(1)} = (Z_t^{(1)}G_{11}+Z_t^{(2)}G_{21}+\cdots+Z_t^{(k)}G_{k1})h_{t+1}(1|x)\\
Z_{t+1}^{(2)} = (Z_t^{(1)}G_{12}+Z_t^{(2)}G_{22}+\cdots+Z_t^{(k)}G_{k2})h_{t+1}(2|x)\\
\vdots\\
Z_{t+1}^{(k)} = (Z_t^{(1)}G_{1k}+Z_t^{(2)}G_{2k}+\cdots+Z_t^{(k)}G_{kk})h_{t+1}(k|x)\\
(9)
$$
它可以简单写为矩阵形式：
$$
Z_{t+1}=Z_tG\otimes H(y_{t+1}|x)  （10）
$$
其中
$$
Z_t=[Z_t^{(1)},\cdots,Z_t^{(k)}]
$$
而 G 是对 g（yi, yj ) 各个元素取指数后的矩阵，即 $$ G = e^{g(yi,yj)} $$。而 $$ H(y_{t+1}|x) $$ 是编码模型。

（CNN、RNN等）对位置 t+1 的各个标签的打分的指数，即
$$
H(y_{t+1}|x)=e^{h(y_{t+1}|x)}
$$
也是一个向量。式（10）中，Zt G 这一步是矩阵乘法，得到一个向量，而 $$ \otimes $$ 是两个向量的逐位对应相乘。

![1527042481987](CRF\1527042481987.png)

归一化因子的递归计算图示。从 t 到 t+1 时刻的计算，包括转移概率和 j+1 节点本身的概率。

如果不熟悉的读者，可能一下子比较难接受（10）式。读者可以把 n=1, n=2, n=3 时的归一化因子写出来，试着找它们的递归关系，慢慢地就可以理解（10）式了。

## 6. 动态规划

写出损失函数$$ -logP(y_1, \cdots,y_n|x) $$后，就可以完成模型的训练了。因为目前的深度学习框架都已经带有自动求导的功能，只要我们能写出可导的 $$ loss $$，就可以帮我们完成优化过程了。

那么剩下的最后一步，就是模型训练完成后，如何根据输入最优路径来。跟前面一样，这也是一个从 k^n 条路径中选最优的问题，而同样地，因为马尔可夫假设的存在，它可以转化为一个动态规划问题，用 viterbi 算法解决，计算量正比于 n。

**动态规划的递归思想就是：一条最优路径切成两段，那么每一段都是一条（局部）最优路径。**

## 7. 纯 Keras 实现的 CRF 层

```python
from keras.layers import Layer
import keras.backend as K


class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)
    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        states = K.expand_dims(states[0], 2) # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0) # (1, output_dim, output_dim)
        output = K.logsumexp(states+trans, 1) # (batch_size, output_dim)
        return output+inputs, [output+inputs]
    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True) # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans*labels, [2,3]), 1, keepdims=True)
        return point_score+trans_score # 两部分得分之和
    def call(self, inputs): # CRF本身不改变输出，它只是一个loss
        return inputs
    def loss(self, y_true, y_pred): # 目标y_pred需要是one hot形式
        mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_states = [y_pred[:,0]] # 初始状态
        log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        return log_norm - path_score # 即log(分子/分母)
    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)
```

