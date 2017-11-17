# This file is part of the program for paper 'Simultaneous prediction
# of protein secondary structure population and intrinsic disorder
# using multi-task deep learning'.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import tensorflow as tf
import evaluation

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

class deepSingle:

    def __init__(self, VEC_SIZE=23, WINDOW_SIZE=9, OUTPUT_SIZE=2, LR_ALPHA=1e-4,
                 loss_function=evaluation.loss_cross_entropy, is_train=True,
                 is_context=False):

        input_size = VEC_SIZE*(WINDOW_SIZE*2+1)

        self._input = tf.placeholder(tf.float32, shape=[None, input_size])
        self._label = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

        raw = tf.reshape(self._input, shape=[-1, 1, (WINDOW_SIZE * 2 + 1), VEC_SIZE])

        w_initial = tf.random_normal_initializer(stddev=0.1)
        b_initial = tf.constant_initializer(0.1)

        # First layer: convolution
        W_conv1 = tf.get_variable("W_conv1", [1, 5, VEC_SIZE, VEC_SIZE*2],
                                  initializer=w_initial)  # input feature map size = 1, output feature map size = 32
        b_conv1 = tf.get_variable("b_conv1", [VEC_SIZE*2], initializer=b_initial)
        h_conv1 = tf.nn.relu(conv2d(raw, W_conv1) + b_conv1)  # convolution layer with window 5x5, strides len = 1, padding = 'SAME'
        h_pool1 = max_pool_2x2(h_conv1)  # max_pooing with window = 2x2

        # Second layer: convolution
        W_conv2 = tf.get_variable("W_conv2", [1, 5, VEC_SIZE*2, VEC_SIZE*4], initializer=w_initial)
        b_conv2 = tf.get_variable("b_conv2", [VEC_SIZE*4], initializer=b_initial)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Third layer: convolution
        W_conv3 = tf.get_variable("W_conv3", [1, 3, VEC_SIZE*4, VEC_SIZE*8], initializer=w_initial)
        b_conv3 = tf.get_variable("b_conv3", [VEC_SIZE*8], initializer=b_initial)
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        # Fully connected layer
        W_fc1 = tf.get_variable("W_fc1", [1 * 3 * VEC_SIZE*8, 512], initializer=w_initial)
        b_fc1 = tf.get_variable("b_fc1", [512], initializer=b_initial)
        h_pool_flat = tf.reshape(h_pool3, [-1, 1 * 3 * VEC_SIZE*8])

        # if is_context:
        #     W_ctxt = tf.get_variable('W_ctxt', [100, 512], initializer=w_initial)
        #     self._context = tf.placeholder(tf.float32, shape=[None, 100])
        #     h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + tf.matmul(self._context, W_ctxt) + b_fc1)
        # else:
        #     h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        # Dropout
        self._keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)


        W_fc2 = tf.get_variable("W_fc2", [512, OUTPUT_SIZE], initializer=w_initial)
        b_fc2 = tf.get_variable("b_fc2", [OUTPUT_SIZE], initializer=b_initial)

        if is_context:
            W_ctxt = tf.get_variable('W_ctxt', [100, OUTPUT_SIZE], initializer=w_initial)
            self._context = tf.placeholder(tf.float32, shape=[None, 100])
            self._output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + tf.matmul(self._context, W_ctxt) + b_fc2)
        else:
            self._output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        # self._output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        if not is_train:
            return

        self._loss = loss_function(self._label, self._output)

        # Train_op
        self._train_step = tf.train.AdamOptimizer(LR_ALPHA).minimize(self._loss)

    @property
    def input(self):
        return self._input

    @property
    def label(self):
        return self._label

    @property
    def output(self):
        return self._output

    @property
    def train_step(self):
        return self._train_step

    @property
    def loss(self):
        return self._loss

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def context(self):
        return self._context

if __name__ == '__main__':
    model = deepSingle(WINDOW_SIZE=9)


