import numpy as np

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops


from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Constant
from keras.layers import Input
from keras.models import Model

class HawkesLayer(Layer):
	def __init__(self, sequences_value, pred_length, delta = 1., sequence_weights=None, proxy_layer=None, sample_stddev=None, **kwargs):
		"""
		can only be the first layer of an architecture
			
		sequences_value[sequence, event, type, feature]

		sequences only contain training events
		"""
		self.sequences_value = np.array(sequences_value,dtype='float32')
		self.sequences_initializer = Constant(self.sequences_value)
		shape = self.sequences_value.shape
		self.nb_sequence = shape[0]
		self.nb_event = shape[1]
		self.nb_type = shape[2]
		self.nb_feature = shape[3]
		self.pred_length = pred_length
		self.delta = delta
		self.proxy_layer = proxy_layer
		self.sample_stddev = sample_stddev

		if self.proxy_layer:
			super(HawkesLayer, self).__init__(**kwargs)
			return

		if sequence_weights:
			assert len(sequence_weights) == self.nb_sequence
			assert len(sequence_weights[0]['spont']) == self.nb_type

			self.spont_initializer = Constant(np.array([x['spont'] for x in sequence_weights]))
			self.Theta_initializer = Constant(np.array([x['theta'] for x in sequence_weights]))
			self.W_initializer = Constant(np.array([x['w'] for x in sequence_weights]))
			self.Alpha_initializer = Constant(np.array([x['alpha'] for x in sequence_weights]))
		else:
			self.spont_initializer = Constant(np.array([[1.3 for j in range(self.nb_type)] for i in range(self.nb_sequence)]))
			self.Theta_initializer = Constant(np.array([[0.05 for j in range(self.nb_type)] for i in range(self.nb_sequence)]))
			self.W_initializer = Constant(np.array([[1. for j in range(self.nb_type)] for i in range(self.nb_sequence)]))
			self.Alpha_initializer = Constant(np.array([[[1. for k in range(self.nb_type)] for j in range(self.nb_type)] for i in range(self.nb_sequence)]))

		super(HawkesLayer, self).__init__(**kwargs)

	def build(self, input_shape):

		assert len(input_shape) == 2
		assert input_shape[1] == 1 # currenly only support one sequence id as input

		self.sequences = self.add_weight(name='pp_sequences',
									shape=(self.nb_sequence, self.nb_event, self.nb_type, self.nb_feature),
									initializer=self.sequences_initializer,
									trainable=False)

		if self.proxy_layer:
			super(HawkesLayer, self).build(input_shape)
			return

		self.spontaneous = self.add_weight(name='pp_spontaneous',
									shape=(self.nb_sequence, self.nb_type),
									initializer=self.spont_initializer,
									trainable=False)

		self.Theta = self.add_weight(name='pp_Theta',
									shape=(self.nb_sequence, self.nb_type),
									initializer=self.Theta_initializer,
									trainable=True)

		self.W = self.add_weight(name='pp_W',
									shape=(self.nb_sequence, self.nb_type),
									initializer=self.W_initializer,
									trainable=True)

		self.Alpha = self.add_weight(name='pp_Alpha',
									shape=(self.nb_sequence, self.nb_type, self.nb_type),
									initializer=self.Alpha_initializer,
									trainable=True)

		super(HawkesLayer, self).build(input_shape)



	def call(self, seq_id):
		if K.dtype(seq_id) != 'int32':
			seq_id = K.cast(seq_id, 'int32')

		# seq_id = K.gather(seq_id, 0)
		# seq_id = seq_id[0,0]
		seq_id = K.gather(K.gather(seq_id,0),0)
		self.train_seq = K.gather(self.sequences, seq_id)[:,:,0] # currently only support the 1st feature
		if self.sample_stddev:
			self.train_seq = self.train_seq + K.random_normal(shape=K.shape(self.train_seq),
										mean=0.,
										stddev=self.sample_stddev)

		if self.proxy_layer:
			spont  = K.gather(self.proxy_layer.spontaneous, seq_id)
			theta = K.gather(self.proxy_layer.Theta, seq_id)
			w = K.gather(self.proxy_layer.W, seq_id)
			alpha = K.gather(self.proxy_layer.Alpha, seq_id)
		else:
			#self.train_seq = K.gather(self.sequences, seq_id)[:,:,0] # currently only support the 1st feature
			spont  = K.gather(self.spontaneous, seq_id)
			theta = K.gather(self.Theta, seq_id)
			w = K.gather(self.W, seq_id)
			alpha = K.gather(self.Alpha, seq_id)


		pred_seq = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.nb_event + self.pred_length, 
			dynamic_size=False, infer_shape=True, clear_after_read=False)

		def copy_unit(t, pred_seq):
			pred_seq = pred_seq.write(t, self.train_seq[t])
			return t+1, pred_seq

		def triggering_unit(int_tao, pred_seq, spont, theta, w, alpha, t, effect):
			tao = K.cast(int_tao, 'float32')
			effect_unit = pred_seq.read(int_tao) * (tf.exp(- w * (t - tao) * self.delta) - tf.exp(- w * (t + 1 - tao) * self.delta))
			# print {
			# 	"effect_unit":effect_unit.get_shape(),
			# 	"pred_seq":pred_seq.read(int_tao).get_shape(),
			# 	"0":(tf.exp(- w * (t - tao) * self.delta) - tf.exp(- w * (t + 1 - tao) * self.delta)).get_shape(),
			# }
			return int_tao + 1, pred_seq, spont, theta, w, alpha, t, effect + effect_unit

		def prediction_unit(int_t, pred_seq, spont, theta, w, alpha):
			t = K.cast(int_t, 'float32')
			term1 = spont / theta * (tf.exp(- theta * t * self.delta) - tf.exp(- theta * (t + 1) * self.delta))
			# term2 = tf.stack([pred_seq.read(tao) * (tf.exp(- w * (t - tao) * self.delta) - tf.exp(- w * (t + 1 - tao) * self.delta)) \
			# 			for tao in range(int_t) ])
			# term2 = tf.reduce_sum(term2,0)
			_0, _1, _2, _3, _4, _5, _6, effect = control_flow_ops.while_loop(
				cond=lambda int_tao, _1, _2, _3, _4, _5, _6, _7: int_tao < int_t,
				body=triggering_unit,
				loop_vars=(tf.constant(0, dtype=tf.int32),pred_seq, spont, theta, w, alpha, t, tf.constant([0.] * self.nb_type,dtype=tf.float32)))

			# print {
			# 	'effect':effect.shape,
			# 	'alpha':alpha.shape,
			# }
			term2 = tf.reduce_sum(tf.matmul(alpha,tf.expand_dims(effect,1)),1) / w
			pred_seq = pred_seq.write(int_t, term1 + term2)
			return int_t+1, pred_seq, spont, theta, w, alpha

		_0, pred_seq = control_flow_ops.while_loop(
			cond=lambda t, pred_seq: t < self.nb_event,
			body=copy_unit,
			loop_vars=(tf.constant(0, dtype=tf.int32),pred_seq))

		_0, pred_seq, _2, _3, _4, _5 = control_flow_ops.while_loop(
			cond=lambda int_t, _1, _2, _3, _4, _5: int_t < self.nb_event + self.pred_length,
			body=prediction_unit,
			loop_vars=(tf.constant(self.nb_event, dtype=tf.int32),pred_seq,spont,theta,w,alpha))

		pred_seq = tf.expand_dims(tf.expand_dims(pred_seq.stack(), 2), 0)  # currently only support the 1st feature and one sample per batch


		return pred_seq
		

	def compute_output_shape(self, input_shape):
		# print '@1 ',input_shape
		return (input_shape[0], self.nb_event + self.pred_length, self.nb_type, self.nb_feature)

class InfiniteDimensionHawkesLayer(HawkesLayer):
	def __init__(self, sequences_value, pred_length, delta = 1., **kwargs):

		super(InfiniteDimensionHawkesLayer, self).__init__(sequences_value, pred_length, delta = 1., **kwargs)

	def call(self, seq_id):
		pass



class Noise(Layer):
	def __init__(self, sequences,pred_length, stddev=0., **kwargs):
		super(Noise, self).__init__(**kwargs)
		self.stddev = stddev

	def call(self, inputs, training=None):
		return inputs + K.random_normal(shape=K.shape(inputs),
											mean=0.,
											stddev=self.stddev)
