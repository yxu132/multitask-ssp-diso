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

class deepCrossStitch:

    def __init__(self, VEC_SIZE=23, WINDOW_SIZE=9,
                 OUTPUT_SIZE=2,
                 OUTPUT_SIZE_2=3,
                 LR_ALPHA=1e-4,
                 loss_function=evaluation.loss_cross_entropy,
                 loss_function_2=evaluation.loss_mean_squared_error,
                 is_train=True,
                 is_context=False,
                 cst_init=[[0.8, 0.2], [0.2, 0.8]]):

        input_size = VEC_SIZE*(WINDOW_SIZE*2+1)
        cst_initial = tf.constant(cst_init)

        self._input = tf.placeholder(tf.float32, shape=[None, input_size])
        self._label = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
        self._label_2 = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE_2])

        raw = tf.reshape(self._input, shape=[-1, 1, (WINDOW_SIZE * 2 + 1), VEC_SIZE])

        w_initial = tf.random_normal_initializer(stddev=0.1)
        b_initial = tf.constant_initializer(0.1)

        # First layer: convolution
        W_conv1 = tf.get_variable("W_conv1", [1, 5, VEC_SIZE, VEC_SIZE*2],initializer=w_initial)  # input feature map size = 1, output feature map size = 32
        b_conv1 = tf.get_variable("b_conv1", [VEC_SIZE*2], initializer=b_initial)
        h_conv1 = tf.nn.relu(conv2d(raw, W_conv1) + b_conv1)  # convolution layer with window 5x5, strides len = 1, padding = 'SAME'
        h_pool1 = max_pool_2x2(h_conv1)  # max_pooing with window = 2x2

        W_conv1_2 = tf.get_variable("W_conv1_2", [1, 5, VEC_SIZE, VEC_SIZE * 2],initializer=w_initial)  # input feature map size = 1, output feature map size = 32
        b_conv1_2 = tf.get_variable("b_conv1_2", [VEC_SIZE * 2], initializer=b_initial)
        h_conv1_2 = tf.nn.relu(conv2d(raw, W_conv1_2) + b_conv1_2)  # convolution layer with window 5x5, strides len = 1, padding = 'SAME'
        h_pool1_2 = max_pool_2x2(h_conv1_2)  # max_pooing with window = 2x2

        self.cross_stitch_unit1 = tf.get_variable("cross_stitch_unit1", initializer=cst_initial)
        cst_pool1 = tf.mul(h_pool1, tf.slice(self.cross_stitch_unit1, [0, 0], [1, 1])) \
                           + tf.mul(h_pool1_2, tf.slice(self.cross_stitch_unit1, [0, 1], [1, 1]))
        cst_pool1_2 = tf.mul(h_pool1, tf.slice(self.cross_stitch_unit1, [1, 0], [1, 1])) \
                             + tf.mul(h_pool1_2, tf.slice(self.cross_stitch_unit1, [1, 1], [1, 1]))


        # Second layer: convolution
        W_conv2 = tf.get_variable("W_conv2", [1, 5, VEC_SIZE*2, VEC_SIZE*4], initializer=w_initial)
        b_conv2 = tf.get_variable("b_conv2", [VEC_SIZE*4], initializer=b_initial)
        h_conv2 = tf.nn.relu(conv2d(cst_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv2_2 = tf.get_variable("W_conv2_2", [1, 5, VEC_SIZE * 2, VEC_SIZE * 4], initializer=w_initial)
        b_conv2_2 = tf.get_variable("b_conv2_2", [VEC_SIZE * 4], initializer=b_initial)
        h_conv2_2 = tf.nn.relu(conv2d(cst_pool1_2, W_conv2_2) + b_conv2_2)
        h_pool2_2 = max_pool_2x2(h_conv2_2)

        self.cross_stitch_unit2 = tf.get_variable("cross_stitch_unit2", initializer=cst_initial)
        cst_pool2 = tf.mul(h_pool2, tf.slice(self.cross_stitch_unit2, [0, 0], [1, 1])) \
                           + tf.mul(h_pool2_2, tf.slice(self.cross_stitch_unit2, [0, 1], [1, 1]))
        cst_pool2_2 = tf.mul(h_pool2, tf.slice(self.cross_stitch_unit2, [1, 0], [1, 1])) \
                             + tf.mul(h_pool2_2, tf.slice(self.cross_stitch_unit2, [1, 1], [1, 1]))

        # Third layer: convolution
        W_conv3 = tf.get_variable("W_conv3", [1, 3, VEC_SIZE*4, VEC_SIZE*8], initializer=w_initial)
        b_conv3 = tf.get_variable("b_conv3", [VEC_SIZE*8], initializer=b_initial)
        h_conv3 = tf.nn.relu(conv2d(cst_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        W_conv3_2 = tf.get_variable("W_conv3_2", [1, 3, VEC_SIZE * 4, VEC_SIZE * 8], initializer=w_initial)
        b_conv3_2 = tf.get_variable("b_conv3_2", [VEC_SIZE * 8], initializer=b_initial)
        h_conv3_2 = tf.nn.relu(conv2d(cst_pool2_2, W_conv3_2) + b_conv3_2)
        h_pool3_2 = max_pool_2x2(h_conv3_2)

        self.cross_stitch_unit3 = tf.get_variable("cross_stitch_unit3", initializer=cst_initial)
        cst_pool3 = tf.mul(h_pool3, tf.slice(self.cross_stitch_unit3, [0, 0], [1, 1])) \
                    + tf.mul(h_pool3_2, tf.slice(self.cross_stitch_unit3, [0, 1], [1, 1]))
        cst_pool3_2 = tf.mul(h_pool3, tf.slice(self.cross_stitch_unit3, [1, 0], [1, 1])) \
                      + tf.mul(h_pool3_2, tf.slice(self.cross_stitch_unit3, [1, 1], [1, 1]))

        ########################## Full connected layer ################


        # Task1
        W_fc1 = tf.get_variable("W_fc1", [1 * 3 * VEC_SIZE * 8, 512], initializer=w_initial)
        b_fc1 = tf.get_variable("b_fc1", [512], initializer=b_initial)
        h_pool_flat = tf.reshape(cst_pool3, [-1, 1 * 3 * VEC_SIZE * 8])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

        # Task2
        W_fc1_2 = tf.get_variable("W_fc1_2", [1 * 3 * VEC_SIZE * 8, 512], initializer=w_initial)
        b_fc1_2 = tf.get_variable("b_fc1_2", [512], initializer=b_initial)
        h_pool_flat_2 = tf.reshape(cst_pool3_2, [-1, 1 * 3 * VEC_SIZE * 8])
        h_fc1_2 = tf.nn.relu(tf.matmul(h_pool_flat_2, W_fc1_2) + b_fc1_2)

        self.cross_stitch_unit4 = tf.get_variable("cross_stitch_unit4", initializer=cst_initial)
        cst_fc1 = tf.mul(h_fc1, tf.slice(self.cross_stitch_unit4, [0, 0], [1, 1])) \
                    + tf.mul(h_fc1_2, tf.slice(self.cross_stitch_unit4, [0, 1], [1, 1]))
        cst_fc1_2 = tf.mul(h_fc1, tf.slice(self.cross_stitch_unit4, [1, 0], [1, 1])) \
                      + tf.mul(h_fc1_2, tf.slice(self.cross_stitch_unit4, [1, 1], [1, 1]))


        #################### Dropout layer ##########################

        if is_context:
            self._context = tf.placeholder(tf.float32, shape=[None, 100])

        # Dropout
        self._keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(cst_fc1, self._keep_prob)

        W_fc2 = tf.get_variable("W_fc2", [512, OUTPUT_SIZE], initializer=w_initial)
        b_fc2 = tf.get_variable("b_fc2", [OUTPUT_SIZE], initializer=b_initial)

        if is_context:
            W_ctxt = tf.get_variable('W_ctxt', [100, OUTPUT_SIZE], initializer=w_initial)
            self._output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + tf.matmul(self._context, W_ctxt) + b_fc2)
        else:
            self._output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # Dropout
        self._keep_prob_2 = tf.placeholder(tf.float32)
        h_fc1_drop_2 = tf.nn.dropout(cst_fc1_2, self._keep_prob_2)

        W_fc2_2 = tf.get_variable("W_fc2_2", [512, OUTPUT_SIZE_2], initializer=w_initial)
        b_fc2_2 = tf.get_variable("b_fc2_2", [OUTPUT_SIZE_2], initializer=b_initial)

        if is_context:
            W_ctxt_2 = tf.get_variable('W_ctxt_2', [100, OUTPUT_SIZE_2], initializer=w_initial)
            self._output_2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop_2, W_fc2_2) + tf.matmul(self._context, W_ctxt_2) + b_fc2_2)
        else:
            self._output_2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop_2, W_fc2_2) + b_fc2_2)

        ############### end of layers ####################


        if not is_train:
            return

        self._loss = loss_function(self._label, self._output)
        self._loss_2 = loss_function_2(self._label_2, self._output_2)

        # Train_op
        self._train_step = tf.train.AdamOptimizer(LR_ALPHA).minimize(self._loss+self._loss_2)
        self._train_step_1 = tf.train.AdamOptimizer(LR_ALPHA).minimize(self._loss)
        self._train_step_2 = tf.train.AdamOptimizer(LR_ALPHA).minimize(self._loss_2)

    @property
    def input(self):
        return self._input

    @property
    def label(self):
        return self._label

    @property
    def label_2(self):
        return self._label_2

    @property
    def output_1(self):
        return self._output

    @property
    def output_2(self):
        return self._output_2

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def keep_prob_2(self):
        return self._keep_prob_2

    @property
    def loss_1(self):
        return self._loss

    @property
    def loss_2(self):
        return self._loss_2

    @property
    def train_step(self):
        return self._train_step

    @property
    def train_step_1(self):
        return self._train_step_1

    @property
    def train_step_2(self):
        return self._train_step_2

    @property
    def context(self):
        return self._context

    @property
    def cross_stitch_units(self):
        return self.cross_stitch_unit1, self.cross_stitch_unit2, self.cross_stitch_unit3, self.cross_stitch_unit4

if __name__ == '__main__':
    model = deepCrossStitch(WINDOW_SIZE=9)


