# DQN-Noisy-Network-with-Tensorflow-2.0.
## Env:
  **Python = 3.7.10**  
  **Tensorflow = 2.7.0**  
  **Gym = 0.19.0**  
  
## Result:  
![Contrast](https://user-images.githubusercontent.com/102845636/172040788-f1d7d6a2-2c49-487b-a43b-5ca2dd2e8d3c.png)

## Noisy Dense:
```
class NoisyDense(Model):
    def __init__(self, units=32,activation=None):
        super().__init__()
        self.units = units
        self.f_p = None
        self.f_q = None
        self.activation = activation

    def f(self, x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

    def build(self, input_shape):
        self.w_mu = tf.Variable(
            initial_value=tf.random.normal(shape=(input_shape[1], self.units),dtype=tf.float64),
            trainable=True
        )
        self.w_sigma = tf.Variable(
            initial_value=tf.random.normal(shape=(input_shape[1], self.units),dtype=tf.float64)
            ,trainable=True
        )
        self.b_mu = tf.Variable(
            initial_value=tf.random.normal(shape=(self.units,), dtype=tf.float64),
            trainable=True
        )
        self.b_sigma = tf.Variable(
            initial_value=tf.random.normal(shape=(self.units,), dtype=tf.float64),
            trainable=True
        )

    def call(self, inputs, training=True):
        if training:
            p = tf.random.normal((inputs.shape[1], 1))
            q = tf.random.normal((1, self.units))
            self.f_p = self.f(p)
            self.f_q = self.f(q)

        w_epsilon = tf.cast(self.f_p * self.f_q, dtype=tf.float64)
        b_epsilon = tf.cast(self.f_q, dtype=tf.float64)

        # w = w_mu + w_sigma*w_epsilon
        self.w = self.w_mu + tf.multiply(self.w_sigma, w_epsilon)
        inputs = tf.cast(inputs,dtype=tf.float64)
        ret = tf.matmul(inputs, self.w)

        # b = b_mu + b_sigma*b_epsilon
        self.b = self.b_mu + tf.multiply(self.b_sigma, b_epsilon)

        if self.activation is not None:
            return self.activation(ret + self.b)
        else:
            return ret + self.b
```
