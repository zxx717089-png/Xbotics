from collections import deque
import numpy as np
import cv2
import sys
import random

sys.path.append("game/")
import wrapped_flappy_bird as game

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 1000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1


def createNetwork():
    """创建深度Q网络"""
    model = keras.Sequential([
        layers.Input(shape=(80, 80, 4)),
        layers.Conv2D(32, (8, 8), strides=4, activation='relu', padding='same',
                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(0.01)),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

        layers.Conv2D(64, (4, 4), strides=2, activation='relu', padding='same',
                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(0.01)),

        layers.Conv2D(64, (3, 3), strides=1, activation='relu', padding='same',
                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(0.01)),

        layers.Flatten(),
        layers.Dense(512, activation='relu',
                     kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                     bias_initializer=keras.initializers.Constant(0.01)),
        layers.Dense(ACTIONS,
                     kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                     bias_initializer=keras.initializers.Constant(0.01))
    ])

    return model


def trainNetwork(model):
    """训练网络"""
    optimizer = keras.optimizers.Adam(learning_rate=1e-6)

    game_state = game.GameState()
    D = deque(maxlen=REPLAY_MEMORY)

    # 创建日志文件
    import os
    os.makedirs(f"logs_{GAME}", exist_ok=True)
    a_file = open(f"logs_{GAME}/readout.txt", "w")
    h_file = open(f"logs_{GAME}/hidden.txt", "w")

    # 初始化游戏
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    # 预处理图像
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 加载检查点
    checkpoint_dir = f'saved_networks_{GAME}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")

    epsilon = INITIAL_EPSILON
    t = 0

    while True:  # 修正原代码的 while 条件
        # 选择动作
        s_t_input = np.expand_dims(s_t, axis=0)
        readout_t = model(s_t_input, training=False).numpy()[0]

        a_t = np.zeros([ACTIONS])
        action_index = 0

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("--------------Random Action--------------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # 修正原代码的 a_t[o]

        # 降低 epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 执行动作
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)  # 修正 COLOR_BRG2GRAY
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 存储经验
        D.append((s_t, a_t, r_t, s_t1, terminal))

        # 训练
        if t > OBSERVE:
            # 采样 minibatch
            minibatch = random.sample(D, min(BATCH, len(D)))

            s_j_batch = np.array([d[0] for d in minibatch])
            a_batch = np.array([d[1] for d in minibatch])
            r_batch = np.array([d[2] for d in minibatch])
            s_j1_batch = np.array([d[3] for d in minibatch])

            # 计算目标 Q 值
            readout_j1_batch = model(s_j1_batch, training=False).numpy()
            y_batch = []

            for i in range(len(minibatch)):
                terminal_batch = minibatch[i][4]
                if terminal_batch:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            y_batch = np.array(y_batch)

            # 训练步骤
            with tf.GradientTape() as tape:
                readout_batch = model(s_j_batch, training=True)
                readout_action = tf.reduce_sum(tf.multiply(readout_batch, a_batch), axis=1)
                cost = tf.reduce_mean(tf.square(y_batch - readout_action))

            gradients = tape.gradient(cost, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        s_t = s_t1
        t += 1

        # 保存模型
        if t % 10000 == 0:
            manager.save()
            print(f"Saved checkpoint at timestep {t}")

        # 确定状态
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:  # 修正语法错误
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))


def playGame():
    """开始游戏"""
    model = createNetwork()
    trainNetwork(model)


def main():
    playGame()


if __name__ == "__main__":  # 修正语法错误
    main()