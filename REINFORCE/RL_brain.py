import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, optimizers, ops

# 设置随机种子保证结果可复现
np.random.seed(1)
tf.random.set_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.95,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        # 存储回合数据的列表
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # 构建网络
        self.model = self._build_net()
        self.optimizer = optimizers.Adam(learning_rate=self.lr)

    def _build_net(self):
        """使用 Keras 3 Functional API 构建模型"""
        inputs = layers.Input(shape=(self.n_features,), name="observations")

        # 第一层全连接
        layer = layers.Dense(
            units=64,
            activation="tanh",
            kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.3),
            bias_initializer=keras.initializers.Constant(0.1),
            name="fc1"
        )(inputs)

        # 输出层 (Softmax 得到动作概率)
        all_act_prob = layers.Dense(
            units=self.n_actions,
            activation="softmax",
            kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.3),
            bias_initializer=keras.initializers.Constant(0.1),
            name="fc2"
        )(layer)

        model = keras.Model(inputs=inputs, outputs=all_act_prob)
        return model

    def choose_action(self, observation):
        """根据概率分布选择动作"""
        # observation 形状调整为 (1, n_features)
        observation = observation[np.newaxis, :]

        # 预测概率
        prob_weights = self.model(observation).numpy()

        # 根据概率随机选择动作
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        """存储单步数据"""
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        """学习过程：计算梯度并更新参数"""
        # 1. 折扣奖励并归一化
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # 准备训练数据
        obs_stack = np.vstack(self.ep_obs)
        acts_stack = np.array(self.ep_as)
        vt_stack = discounted_ep_rs_norm

        # 2. 使用 GradientTape 计算梯度
        with tf.GradientTape() as tape:
            # 前向传播获取所有动作的概率
            all_prob = self.model(obs_stack, training=True)

            # 交叉熵损失的思想：我们希望增大获得高 Reward 的动作概率
            # 这里的 loss = -log(prob) * vt
            # 计算被选中动作的 log 概率
            neg_log_prob = keras.losses.sparse_categorical_crossentropy(
                y_true=acts_stack, y_pred=all_prob
            )

            # 损失函数 = 负对数概率 * 累积奖励
            loss = tf.reduce_mean(neg_log_prob * vt_stack)

        # 3. 更新权重
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # 清空当前回合数据
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        """计算回合的折扣奖励"""
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 归一化奖励（均值为0，方差为1），有利于梯度收敛
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /=np.std(discounted_ep_rs)
        return discounted_ep_rs