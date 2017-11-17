# Copyright 2016 Monash University
#
# Author: Ying Xu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy
import tensorflow as tf

class CrossValidation(object):

  def __init__(self, features, labels, test_features=None, test_labels=None, contexts=None, test_contexts=None, disos=None, test_disos=None):
    self._features = features
    self._labels = labels
    self._test_features = test_features
    self._test_labels = test_labels
    self._contexts = contexts
    self._test_contexts = test_contexts
    self._disos = disos
    self._test_disos = test_disos
    self._fold = 0
    self._numFold = 10

  @property
  def contexts(self):
    return self._contexts

  def nextFold(self):
    train_feature = self._features[0:self._fold]+self._features[self._fold+1:]
    train_label = self._labels[0:self._fold]+self._labels[self._fold+1:]
    test_feature = self._test_features[self._fold]
    test_label = self._test_labels[self._fold]

    train_features = train_feature[0]
    for x in train_feature[1:]:
      train_features = numpy.concatenate((train_features, x), axis=0)

    train_labels = train_label[0]
    for x in train_label[1:]:
      train_labels = numpy.concatenate((train_labels, x), axis=0)

    if self._contexts != None:
        train_context = self._contexts[0:self._fold] + self._contexts[self._fold + 1:]
        train_contexts = train_context[0]
        for x in train_context[1:]:
            train_contexts = numpy.concatenate((train_contexts, x), axis=0)
        test_context = self._test_contexts[self._fold]
        self._fold += 1
        return DataSet(train_features, train_labels, contexts=train_contexts), DataSet(test_feature, test_label, contexts=test_context, is_test=True)
    elif self._disos != None:
        train_diso = self._disos[0:self._fold] + self._disos[self._fold+1:]
        train_disos = train_diso[0]
        for x in train_diso[1:]:
            train_disos = numpy.concatenate((train_disos, x), axis=0)
        test_diso = self._test_disos[self._fold]
        self._fold += 1
        return DataSet(train_features, train_labels, disos=train_disos), DataSet(test_feature, test_label, disos=test_diso, is_test=True)
    else:
      self._fold += 1
      return DataSet(train_features, train_labels), DataSet(test_feature, test_label, is_test=True)

  def get_i(self, i):
    if self._test_features == None:
      if self._contexts != None: 
          return DataSet(self._features[i], self._labels[i], contexts=self._contexts[i], is_test=True)
      else: 
          return DataSet(self._features[i], self._labels[i], is_test=True)
    else:
      fold = self._fold - 1
      index = i
      if i >= fold:
          index = i+1
      if self._test_contexts != None:
          return DataSet(self._test_features[index], self._test_labels[index], contexts=self._test_contexts[index], is_test=True)
      elif self._test_disos != None:
          return DataSet(self._test_features[index], self._test_labels[index], disos=self._test_disos[index], is_test=True)
      else:
          return DataSet(self._test_features[index], self._test_labels[index], is_test=True)

  def all(self):
    train_feature = self._features
    train_label = self._labels
    train_features = train_feature[0]
    for x in train_feature[1:]:
      train_features = numpy.concatenate((train_features, x), axis=0)
    train_labels = train_label[0]
    for x in train_label[1:]:
      train_labels = numpy.concatenate((train_labels, x), axis=0)
    if self._contexts != None:
      train_contexts = self._contexts[0]
      for x in self._contexts[1:]:
          train_contexts = numpy.concatenate((train_contexts, x), axis=0)
      return DataSet(train_features, train_labels, contexts=train_contexts, is_test=True)
    elif self._disos != None:
      disos = self._disos[0]
      for x in self._disos[1:]:
        disos = numpy.concatenate((disos, x), axis=0)
      return DataSet(train_features, train_labels, disos=disos, is_test=True)
    else:
      return DataSet(train_features, train_labels, is_test=True)


class DataSet(object):

  def __init__(self, features, labels, contexts=None, disos=None, dtype=tf.float32, is_test=False):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    assert features.shape[0] == labels.shape[0], ('features.shape: %s labels.shape: %s' % (features.shape,labels.shape))
    self._num_examples = features.shape[0]
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])
    if dtype == tf.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      features = features.astype(numpy.float32)
      features = numpy.multiply(features, 1.0 / 255.0)
    self._features = features
    self._labels = labels
    self._contexts = contexts
    self._disos = disos
    self._epochs_completed = 0
    self._index_in_epoch = 0

    # randomise at initialisation
    if not is_test:
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._features = self._features[perm]
      self._labels = self._labels[perm]
      if self._contexts != None:
        self._contexts = self._contexts[perm]
      if self._disos != None:
        self._disos = self._disos[perm]

  @property
  def features(self):
    return self._features

  @property
  def labels(self):
    return self._labels

  @property
  def disos(self):
    return self._disos

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed


  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._features = self._features[perm]
      self._labels = self._labels[perm]
      if self._contexts != None:
        self._contexts = self._contexts[perm]
      if self._disos != None:
        self._disos = self._disos[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    if self._contexts != None:
      return self._features[start:end], self._labels[start:end], self._contexts[start:end]
    elif self._disos != None:
      return self._features[start:end], self._labels[start:end], self._disos[start:end]
    else:
      return self._features[start:end], self._labels[start:end], None

  def all(self):
    if self._contexts != None:
      return self._features[:], self._labels[:], self._contexts[:]
    elif self._disos != None:
      return self._features[:], self._labels[:], self._disos[:]
    else:
      return self._features[:], self._labels[:], None

# def read_test_set(dir, testname, win):
#   features = extractFeatures(dir+testname+'_feature_win'+str(win)+'.csv', win_size=win*2+1)
#   labels = extractOneLabels(dir+testname+'_label_win'+str(win)+'.csv')
#   return DataSet(features, labels)

#features = extractFeatures('../DATA/cross_validation_10_clustered/Fold_0_feature_win11.csv', win_size=11)
#features = extractFeatures('../DATA/test_IDP/Fold_0_feature_win11.csv', win_size=11)

#test = read_test_set('../DATA/test/', 'IDR', win=9).all()
