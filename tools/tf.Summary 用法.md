# tf.Summary 用法

## tf.summary.scalar

​	用来显示标量信息，其格式如下：

```
tf.summary.scalar(
    name,
    tensor,
    collections=None,
    family=None
)

```

参数说明：

- **name**: A name for the generated node. Will also serve as the series name in TensorBoard.
- **tensor**: A real numeric Tensor containing a single value.
- **collections**: Optional list of graph collections keys. The new summary op is added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
- **family**: Optional; if provided, used as the prefix of the summary tag name, which controls the tab name used for display on Tensorboard.

应用说明：

​	例如：tf.summary.scalar('mean', mean)

​	一般在画loss,accuary时会用到这个函数。

