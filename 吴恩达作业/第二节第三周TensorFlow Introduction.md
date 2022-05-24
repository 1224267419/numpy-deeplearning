```python
X = np.random.randn(3, 1)
W = np.random.randn(4, 3)
b = np.random.randn(4, 1)
Y = tf.add(tf.matmul(W, X), b)
```

numpy随机化矩阵，并对其

``` python
z=tf.cast(z, tf.float32)
a=tf.keras.activations.sigmoid(z)
```

如上，先转换z的数据类型，再使用sigmoid一键转换

