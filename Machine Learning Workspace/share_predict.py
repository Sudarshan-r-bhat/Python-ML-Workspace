# here we will be using only the equity prices and going to check how well did our model predict the share prices.
# sp500 is like NSE in India, it list the overall prices of 500 companies in the US.
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("D:\\ML_datasets\\ML_share_dataset\\SP500_stocks_dataset-master\\SP500_stocks_dataset.csv")

data = data.drop(["DATE"], 1)

n = data.shape[0]
p = data.shape[1]

data = np.array(data)

train_start = 0
train_end = int(np.floor(0.8 * n))
test_start = train_end + 1
test_end = n

# now its time to get our data ready for tensor operations.
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# i think we are taking the sp500 value as the y coordinate and other cols as Xcord
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# no of stocks in the training data
n_stocks = X_train.shape[1]

# neurons
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# session
net = tf.InteractiveSession()

# placeholders . but for what? used as a handle to feed values.
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# hidden weights and bias.
w_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
w_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
w_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
w_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

w_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# adding hidden layers
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, w_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, w_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, w_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, w_hidden_4), bias_hidden_4))

out = tf.transpose(tf.add(tf.matmul(hidden_4, w_out), bias_out))

# cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))  # mse = mean squared error. msa = mean abs error.

# optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# init
net.run(tf.global_variables_initializer())

# setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

# fit nn
batch_size = 256
mse_train = []
mse_test = []

# run
epochs = 2

for e in range(epochs):

    # shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    for i in range(0, len(y_train) // batch_size):

        start = i * batch_size
        batch_x = X_train[start: start + batch_size]
        batch_y = y_train[start: start + batch_size]

        # run optimizer with the batch.
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})
        plt.show()
        # show progress
        """if np.mod(i, 50) == 0:
            # mse train & test.
            mse_train.append(net.run(mse, feed_dict={X: batch_x, Y: batch_y}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))

            print('mse_test = ', mse_test)
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch' + str(e) + ', Batch' + str(i))
            plt.pause(0.01)"""




