// Copyright (c) 2019 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

import tensorflow as tf

def gen_aug_bits(batch_size, aug_level, max_aug_level):
    """Creates a batch of random bits for augmentation
          Args:
            batch_size: batch size
            aug_level: a scalar tensor to index the current augmentation level. aug_bits have value zero beyond the current augmentation level
            max_aug_level: the maximum augmentation level in order to initialize the PA relevant variables once at the beginning of training, avoiding run-time tf.graph changing.
          Returns:
            aug_bits: augmentation bits (batch size, max aug levels)
            flag_odd: checksum of the augmentation bits (batch_size,)
    """
    # a batch of random bits following uniform distribution
    aug_bits  = tf.random.categorical(tf.zeros((batch_size, 2)), max_aug_level, dtype=tf.int32)
    # based on the current augmentation level, create a mask to null out aug bits beyond the current level
    mask      = tf.one_hot(aug_level, max_aug_level + 1, dtype=tf.int32)
    cum_mask  = tf.reshape(tf.cumsum(mask[1:], reverse=True), [1, max_aug_level])
    aug_bits  = aug_bits * cum_mask
    # compute the checksum of the generated batch of aug bits, 0: even, 1: odd
    flag_odd  = tf.floormod(tf.reduce_sum(aug_bits, axis=1),2)
    return aug_bits, flag_odd

def pa_conv2d(input_, output_dim, aug_bits, aug_level, max_aug_level=0, k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', bias_on=True, name='paconv2d'):
    """Creates an augmented 2D convolutional (conv.) layer.
      Args:
        input_: input tensor of the conv. layer (batch size, feat_h, feat_w, feat_c).
        output_dim: number of features in the output layer.
        aug_bits: augmentation bits that are concatenated with input_ (batch size, max aug levels)
        aug_level: a scalar tensor to index the current augmentation level.
        max_aug_level: the maximum augmentation level in order to initialize the PA relevant variables once at the beginning of training, avoiding run-time tf.graph changing.
        k_h: conv. kernel height.
        k_w: conv. kernel width.
        d_h: conv. height stride.
        d_w: conv. width stride.
        padding: padding type of conv. layer
        bias_on: if using bias
        name: Optional, variable scope to put the layer's parameters into
      Returns:
        conv: output tensor of the conv. layer
    """
    with tf.variable_scope(name, values=[input_, aug_bits], reuse=tf.AUTO_REUSE):

        input_shape = input_.get_shape().as_list()
        # create an augmented input
        # broadcast the bit value over (height, width)
        aug_bit_ch = tf.tile(tf.cast(aug_bits, input_.dtype), [1, input_shape[1], input_shape[2], 1])
        aug_input  = tf.concat([input_, aug_bit_ch], axis=3)

        # conv. kernel for data feats
        w_1   = tf.get_variable('w_aug_1', [k_h, k_w, input_shape[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
        # conv. kernel for aug bits
        w_2   = tf.get_variable('w_aug_2', [k_h, k_w, max_aug_level, output_dim], initializer=tf.zeros_initializer())
        w_aug = tf.convert_to_tensor(tf.concat([w_1, w_2], axis=2), name="w_aug")

        # depending on the current augmentation level, w_2 are masked
        # such that the weights beyond the current augmentation level will not affect the current training
        # Note: aug_level has the value range [0, max_aug_level], where $0$ indicates no augmentation bits
        mask     = tf.one_hot(aug_level, max_aug_level+1, dtype=tf.int32)
        cum_mask = tf.reshape(tf.cumsum(mask[1:], reverse=True), [1, 1, max_aug_level, 1])
        mask4w   = tf.tile(tf.concat([tf.ones((1, 1, input_shape[-1], 1), dtype=tf.int32), cum_mask], axis=2), [k_h, k_w, 1, output_dim])
        mask_w_aug = tf.where(tf.equal(mask4w, tf.constant(1, tf.int32)), w_aug, tf.zeros_like(w_aug))

        ##if spectral normalization is in use, mask_w_aug is normalized by the iterative power method before convolution
        ##based on "Spectral Normalization for Generative Adversarial Networks" [Miyato et al., 2018]
        conv = tf.nn.conv2d(aug_input, mask_w_aug, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.zeros_initializer(), trainable=bias_on)
        conv = tf.nn.bias_add(conv, biases)

        return conv
