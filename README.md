11.09

Image classification with separable conv and attention mechanism.

TensorFlow 2.4

tensorflow中的模型训练太集成，无法方便地获取输入形状等信息，这种情况下可以向model中添加层：

```python
def print_shape(x):
    print(x.shape)
    return x
layers.Lambda(print_shape)	# add to model
```

