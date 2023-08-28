# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""phased Gated Recurrent Unit layer."""

import uuid

import tensorflow as tf

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine import base_layer
from keras.engine.input_spec import InputSpec
from keras.layers.rnn import gru_lstm_utils
from keras.layers.rnn import rnn_utils
from keras.layers.rnn.base_rnn import RNN
from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.utils import tf_utils
from numpy import float32
import logging
# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


def phi_fast(time, s, tau):
    x = time - s
    tau += 1
    x = tf.math.floormod(x, tau)
    x = tf.math.divide(x, tau)
    tau += -1
    return x


def time_gate_fast(phi, ron, alpha):
    ron += 1
    cond_1 = tf.cast(tf.less_equal(phi, 0.5 * ron), dtype='float32')
    cond_2 = tf.cast(tf.logical_and(tf.less(0.5 * ron, phi), tf.less(phi, ron)), dtype='float32')
    cond_3 = tf.cast(tf.greater_equal(phi, ron), dtype='float32')

    term_1 = tf.math.multiply(cond_1, 2.0 * phi / ron)
    term_2 = tf.math.multiply(cond_2, 2.0 - 2.0 * phi / ron)
    term_3 = tf.math.multiply(cond_3, alpha * phi)

    ron += -1
    return term_1 + term_2 + term_3


class phased_GRUCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
    """Cell class for the GRU layer.

    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.

    This class processes one step within the whole time sequence input, whereas
    `tf.keras.layer.GRU` processes the whole sequence.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use. Default: hyperbolic tangent
        (`tanh`). If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use for the recurrent step.
        Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
        applied (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state.  Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector. Default: `zeros`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector.
        Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_constraint: Constraint function applied to the bias vector. Default:
        `None`.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
        for the linear transformation of the recurrent state. Default: 0.
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before",
        True = "after" (default and cuDNN compatible).

    Call arguments:
      inputs: A 2D tensor, with shape of `[batch, feature]`.
      states: A 2D tensor with shape of `[batch, units]`, which is the state
        from the previous time step. For timestep 0, the initial state provided
        by user will be feed to cell.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
    """

    def __init__(
            self,
            units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            reset_after=True,
            time_gate=True,
            **kwargs,
    ):
        if units <= 0:
            raise ValueError(
                "Received an invalid value for argument `units`, "
                f"expected a positive integer, got {units}."
            )
        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop(
                "enable_caching_device", True
            )
        else:
            self._enable_caching_device = kwargs.pop(
                "enable_caching_device", False
            )
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))

        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units
        self.time_gate = time_gate

    def build(self, input_shape):
        super().build(input_shape)

        input_dim = input_shape[-1]

        # new '- 1' because time gets treated separately (not part of the regular input)
        if self.time_gate:
            input_dim = input_dim - 1
            # time_shape = input_shape[0]

        default_caching_device = rnn_utils.caching_device(self)
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device,
        )

        if self.use_bias:
            if not self.reset_after:
                if self.time_gate:
                    bias_shape = (4 * self.units + 3,)
                else:
                    bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU
                # biases `(2 * 3 * self.units,)`, so that we can distinguish the
                # classes when loading and converting saved weights.
                if self.time_gate:
                    bias_shape = (2, 4 * self.units + 3)
                else:
                    bias_shape = (2, 3 * self.units)
            self.bias = self.add_weight(
                shape=bias_shape,
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device,
            )
        else:
            self.bias = None



        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = (
            states[0] if tf.nest.is_nested(states) else states
        )  # previous memory

        # new #####################################################
        if self.time_gate:
            inputs = inputs[:, :-1]
            time = inputs[:, -1:]

            #logging.warning('time(:,0) : {}, timeshape: {}'.format(time[:,0], time.shape))

        print(inputs)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = tf.unstack(self.bias)

        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

        x_z = backend.dot(inputs_z, self.kernel[:, : self.units])
        x_r = backend.dot(inputs_r, self.kernel[:, self.units: self.units * 2])
        x_h = backend.dot(inputs_h, self.kernel[:, self.units * 2: self.units * 3])

        if self.use_bias:
            x_z = backend.bias_add(x_z, input_bias[: self.units])
            x_r = backend.bias_add(x_r, input_bias[self.units: self.units * 2])
            x_h = backend.bias_add(x_h, input_bias[self.units * 2: self.units * 3])

        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

        recurrent_z = backend.dot(h_tm1_z, self.recurrent_kernel[:, : self.units])
        recurrent_r = backend.dot(h_tm1_r, self.recurrent_kernel[:, self.units: self.units * 2])

        if self.reset_after and self.use_bias:
            recurrent_z = backend.bias_add(recurrent_z, recurrent_bias[: self.units])
            recurrent_r = backend.bias_add(recurrent_r, recurrent_bias[self.units: self.units * 2])

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        # reset gate applied after/before matrix multiplication
        if self.reset_after:
            recurrent_h = backend.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:self.units * 3])
            if self.use_bias:
                recurrent_h = backend.bias_add(recurrent_h, recurrent_bias[self.units * 2:self.units * 3])
                recurrent_h = r * recurrent_h
        else:
            recurrent_h = backend.dot(r * h_tm1_h, self.recurrent_kernel[:, self.units * 2:self.units * 3])

        hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        # new #####################################################
        # time_gate
        if self.time_gate:
            tau = input_bias[self.units * 3: self.units * 4] + 10
            s = input_bias[self.units * 4: self.units * 4 + 1] + 3000
            ron = input_bias[self.units * 4 + 1: self.units * 4 + 2] + 0.8
            alpha = input_bias[self.units * 4 + 2: self.units * 4 + 3] + 0.5
            phi = phi_fast(time, s, tau)
            k = time_gate_fast(phi, ron, alpha)
            h = tf.math.multiply(k, h) + (1 - k) * h_tm1

        new_state = [h] if tf.nest.is_nested(states) else h
        return h, new_state
