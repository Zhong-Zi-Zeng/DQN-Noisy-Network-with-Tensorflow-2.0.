from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Build a original DQN model
def build_ori_dqn(lr, n_actions, input_shape):
    input = Input(shape=(input_shape,))

    h1 = Dense(1024, activation='relu')(input)
    h2 = Dense(512, activation='relu')(h1)
    h3 = Dense(256, activation='relu')(h2)
    h4 = Dense(128, activation='relu')(h3)

    output = Dense(n_actions, activation='linear')(h4)
    model = Model(inputs=[input], outputs=[output])
    model.compile(Adam(learning_rate=lr), loss='mse')
    model.summary()

    return model

# Build a Noisy DQN model
class NoisyModel(Model):
    def __init__(self,state_shape,n_actions):
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.l1 = Dense(256, activation='relu', input_shape=(self.state_shape,))
        self.n1 = NoisyDense(128,activation=tf.keras.activations.relu)
        self.n2 = NoisyDense(self.n_actions,activation=None)

    def call(self, inputs):
        output = self.l1(inputs)
        output = self.n1(output)
        output = self.n2(output)

        return output

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

# Replay memory
class ReplayBuffer():
    def __init__(self, max_mem, n_action, input_shape):
        self.max_mem = max_mem

        self.state_memory = np.zeros((self.max_mem, input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.max_mem, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.max_mem, n_action), dtype=np.int8)
        self.reward_memory = np.zeros(self.max_mem)
        self.terminal_memory = np.zeros(self.max_mem, dtype=np.float32)
        self.mem_counter = 0

    def store_transition(self, state, action, reward, next_state, done):
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1
        self.action_memory[self.mem_counter] = actions
        self.state_memory[self.mem_counter] = state
        self.reward_memory[self.mem_counter] = reward
        self.next_state_memory[self.mem_counter] = next_state
        self.terminal_memory[self.mem_counter] = 1 - int(done)

        self.mem_counter += 1
        if (self.mem_counter == self.max_mem):
            self.mem_counter = 0

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.max_mem)
        batch = np.random.choice(max_mem, batch_size)

        state = self.state_memory[batch]
        next_state = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return state, actions, rewards, next_state, terminal

class Agent():
    def __init__(self
                 , alpha
                 , gamma
                 , n_actions
                 , epsilon
                 , batch_size
                 , epsilon_end
                 , mem_size
                 , epsilon_dec
                 , input_shape
                 , use_noisy
                 , iteration=200):

        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.iteration = iteration
        self.iteration_counter = 0
        self.use_noisy = use_noisy

        self.memory = ReplayBuffer(max_mem=mem_size, n_action=n_actions, input_shape=input_shape)

        # Build model
        if self.use_noisy:
            self.q_eval = NoisyModel(state_shape=input_shape,n_actions=n_actions)
            self.q_target_net = NoisyModel(state_shape=input_shape,n_actions=n_actions)
        else:
            self.q_eval = build_ori_dqn(lr=alpha, n_actions=n_actions, input_shape=input_shape)
            self.q_target_net = build_ori_dqn(lr=alpha, n_actions=n_actions, input_shape=input_shape)
            self.q_target_net.set_weights(self.q_eval.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()

        if self.use_noisy:
            state = tf.convert_to_tensor(state)
            actions = self.q_eval(state)
            actions = tf.squeeze(actions).numpy()
            action = np.argmax(actions)
        else:
            if (rand < self.epsilon):
                action = np.random.choice(self.action_space)
            else:
                actions = self.q_eval.predict(state)
                action = np.argmax(actions)

        return action

    def learn(self):
        if (self.memory.mem_counter < self.batch_size):
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)

        if self.use_noisy:
            opt = Adam(learning_rate=self.alpha)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            state = tf.convert_to_tensor(state)
            next_state = tf.convert_to_tensor(next_state)
            q_target_pre = self.q_target_net(next_state).numpy()

            with tf.GradientTape() as tape:
                q_eval = self.q_eval(state).numpy()
                q_target = q_eval.copy()
                q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_target_pre, axis=1) * done
                q_target = tf.convert_to_tensor(q_target)
                q_eval = self.q_eval(state,training=False)
                loss = (q_eval - q_target) ** 2

            grads = tape.gradient(loss, self.q_eval.trainable_variables)
            opt.apply_gradients(zip(grads, self.q_eval.trainable_variables))

        else:
            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end
            q_eval = self.q_eval.predict(state)
            q_target_pre = self.q_target_net.predict(next_state)

            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_target_pre, axis=1) * done
            _ = self.q_eval.fit(state, q_target, verbose=0)

        self.iteration_counter += 1
        if (self.iteration_counter == self.iteration):
            self.q_target_net.set_weights(self.q_eval.get_weights())
            self.iteration_counter = 0

