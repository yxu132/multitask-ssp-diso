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

def loss_mean_squared_error(y, y_output):
    return tf.reduce_mean(tf.square(y-y_output))

def mean_absolute_error(y, y_output):
    return tf.reduce_mean(tf.abs(y-y_output))

def loss_cross_entropy(y_, y_output):
    return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_output,1e-10,1.0)), reduction_indices=[1]))

def accuracy(y_, y_output):
    correct_predictions = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y_output, 1))
    acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return acc

def R(y, y_output):
    y_mean, y_var = tf.nn.moments(y, [0])
    y_output_mean, y_output_var = tf.nn.moments(y_output, [0])
    return tf.reduce_mean((y - y_mean) * (y_output - y_output_mean)) / (tf.sqrt(y_var)*tf.sqrt(y_output_var))

def pssp_eval(y_, y_conv):
    return R(y_[:, 0], y_conv[:, 0]), R(y_[:, 1], y_conv[:, 1]), R(y_[:, 2], y_conv[:, 2]), \
           loss_mean_squared_error(y_[:, 0], y_conv[:, 0]), loss_mean_squared_error(y_[:, 1], y_conv[:, 1]), loss_mean_squared_error(y_[:, 2], y_conv[:, 2]), \
           mean_absolute_error(y_[:, 0], y_conv[:, 0]), mean_absolute_error(y_[:, 1], y_conv[:, 1]), mean_absolute_error(y_[:, 2], y_conv[:, 2])
