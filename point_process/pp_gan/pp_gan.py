#coding:utf-8
import os,sys,csv,json
import numpy
import numpy as np
import multiprocessing

from generator import HawkesGenerator
from discriminator import CNNDiscriminator

cpu = multiprocessing.cpu_count()

class HawkesGAN(object):
	def __init__(self):
		self.gen = HawkesGenerator()
		self.gen_full = HawkesGenerator()
		self.gen_observed = HawkesGenerator()
		self.dis = CNNDiscriminator()

	def full_train(self,full_sequences,observed_sequences,train_sequences,features,publish_years,pids,superparams,
			max_fulltrain_iter=400,train_gan_method='gan',mse_weight=0.,gan_weight=1.,need_pretrain=True,need_noise_dropout=False,
			stddev=5.,sample_stddev=0.,gan_stddev=0.,wgan_stddev=0.,hawkes_output_loss='mse',wgan_clip=1.):
		from keras.layers import Input
		from keras.models import Model
		from keras.optimizers import SGD
		from pp_layer import Noise

		nb_sequence = len(full_sequences)
		observed_length = len(observed_sequences[0])
		train_length = len(train_sequences[0])
		test_length = len(full_sequences[0]) - observed_length
		val_length = observed_length - train_length
		nb_type = len(full_sequences[0][0])
		nb_feature = len(full_sequences[0][0][0])

		# guarantee pretrained generator
		if need_pretrain == True and self.gen.sequence_weights is None:
			raise Exception('generator not pretrained, or weights not loaded')
		# self.gen.create_trainable_model(observed_sequences,test_length)
		self.gen.create_trainable_model(train_sequences,val_length,need_noise_dropout=need_noise_dropout,stddev=stddev,sample_stddev=sample_stddev)
		self.gen_full.create_trainable_model(observed_sequences,test_length,proxy_layer=self.gen.hawkes_layer)
		self.gen_observed.create_trainable_model(train_sequences,val_length,proxy_layer=self.gen.hawkes_layer)

		# build keras models
		# self.gen.model.compile(optimizer='adam', loss='mape', metrics=['accuracy'])
		# self.gen.model.fit(np.arange(nb_sequence),np.array(full_sequences),verbose=1,batch_size=1,epochs=100)
		assert train_gan_method in ['gan','wgan']

		if train_gan_method == 'gan':
			# self.dis.create_trainable_model(observed_length + test_length,nb_type,nb_feature)
			self.dis.create_trainable_model(observed_length,nb_type,nb_feature)
			self.dis.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

			self.dis.model.trainable = False
			for l in self.dis.model.layers: l.trainable = False
			z = Input(batch_shape=(1,1), dtype='int32')
			g_z = self.gen.model(z)
			# full_g_z = self.gen_full.model(z)
			noised_g_z = Noise(train_sequences,val_length)(g_z)
			y = self.dis.model(g_z)
			noised_gen_model = Model(inputs=[z], outputs=[noised_g_z])
			# full_gen_model = Model(inputs=[z], outputs=[full_g_z])
			gan_model = Model(inputs=[z], outputs=[g_z,y])
			gan_model.compile(optimizer='rmsprop', 
				loss={'dis_output':'categorical_crossentropy','hawkes_output':hawkes_output_loss},
				loss_weights={'dis_output':gan_weight,'hawkes_output':mse_weight},
				metrics=['categorical_accuracy'])
		elif train_gan_method == 'wgan':
			from pp_loss import wasserstein_d_loss, wasserstein_g_loss
			self.dis.create_trainable_wasserstein(observed_length,nb_type,nb_feature,wgan_clip=wgan_clip)
			self.dis.model.compile(optimizer='rmsprop', loss=wasserstein_d_loss)

			self.dis.model.trainable = False
			for l in self.dis.model.layers: l.trainable = False
			z = Input(batch_shape=(1,1), dtype='int32')
			g_z = self.gen.model(z)
			# full_g_z = self.gen_full.model(z)
			noised_g_z = Noise(train_sequences,val_length)(g_z)
			y = self.dis.model(g_z)
			noised_gen_model = Model(inputs=[z], outputs=[noised_g_z])
			# full_gen_model = Model(inputs=[z], outputs=[full_g_z])
			gan_model = Model(inputs=[z], outputs=[g_z,y])
			gan_model.compile(optimizer='rmsprop', 
				loss={'dis_output':wasserstein_g_loss,'hawkes_output':hawkes_output_loss},
				loss_weights={'dis_output':gan_weight,'hawkes_output':mse_weight})



		# self.gen.model.summary()
		# self.dis.model.summary()
		# gan_model.summary()
		# sys.stdout.flush()

		# pretrain discrimnator
		Z = np.arange(nb_sequence)
		X = np.array(observed_sequences)
		X_full = np.array(full_sequences)
		b_size = 512
		max_pretrain_iter = 5
		it = 0

		likelihood = self.compute_likelihood()
		mape_acc_test = self.compute_mape_acc(self.gen_full.model,Z,X_full,test_length)
		mape_acc_val = self.compute_mape_acc(self.gen_observed.model,Z,X,val_length)
		weights = self.get_weights_value(self.gen.hawkes_layer)
		print {
			'source':'before pre-training discriminator',
			'mape':mape_acc_test['mape'],
			'acc':mape_acc_test['acc'],
			'acc_vary':mape_acc_test['acc_vary'],
			'mape_val':mape_acc_val['mape'],
			'acc_val':mape_acc_val['acc'],
			'LL':likelihood,
			'alpha':weights['mean_alpha'],
			'spont':weights['mean_spont'],
			'theta':weights['mean_theta'],
			'w':weights['mean_w'],
		}
		sys.stdout.flush()

		while it <= max_pretrain_iter * b_size:
			batch_z = Z[np.arange(it,it+b_size)%nb_sequence]
			batch_g_z = noised_gen_model.predict(batch_z,batch_size=1)
			batch_x = X[np.arange(it,it+b_size)%nb_sequence]
			batch_x_merge = np.array([batch_g_z[i/2] if i % 2 == 0 else batch_x[i/2] for i in range(2 * b_size)])
			batch_y_merge = np.array([[0.,1.] if i % 2 == 0 else [1.,0.] for i in range(2 * b_size)])
			history = self.dis.model.fit(batch_x_merge,batch_y_merge,batch_size=1,epochs=1,verbose=0)

			it += b_size

			# print history.history
			# exit()

			# if it + 1 >= max_pretrain_iter: break


		# full train gan model
		Z = np.arange(nb_sequence)
		X = np.array(observed_sequences)
		X_full = np.array(full_sequences)
		Y = np.array([[1.,0.] for i in range(nb_sequence)])
		b_size_dis = int(32. * nb_sequence / 10000.)
		b_size_gan = int(512. * nb_sequence / 10000.)
		it_dis = 0
		it_gan = 0
		epoch_dis = 1 if train_gan_method == 'gan' else 5

		while it_gan <= b_size_gan * max_fulltrain_iter:
			batch_z = Z[np.arange(it_dis,it_dis+b_size_dis)%nb_sequence]
			batch_g_z = noised_gen_model.predict(batch_z,batch_size=1)
			batch_x = X[np.arange(it_dis,it_dis+b_size_dis)%nb_sequence]
			batch_x_merge = np.array([batch_g_z[i/2] if i % 2 == 0 else batch_x[i/2] for i in range(2 * b_size_dis)])
			batch_y_merge = np.array([[0.,1.] if i % 2 == 0 else [1.,0.] for i in range(2 * b_size_dis)])
			dis_his = self.dis.model.fit(batch_x_merge,batch_y_merge,batch_size=1,epochs=epoch_dis,verbose=0)

			batch_z_gan = Z[np.arange(it_gan,it_gan+b_size_gan)%nb_sequence]
			batch_y_gan = np.array([[1.,0.] for i in range(b_size_gan)])
			# batch_g_z_gan = self.gen.model.predict(batch_z,batch_size=1)
			batch_x_gan = X[np.arange(it_gan,it_gan+b_size_gan)%nb_sequence]
			gan_his = gan_model.fit(batch_z_gan,{'dis_output':batch_y_gan,'hawkes_output':batch_x_gan},
				batch_size=1,epochs=1,verbose=0)

			likelihood = self.compute_likelihood()
			mape_acc_test = self.compute_mape_acc(self.gen_full.model,Z,X_full,test_length)
			mape_acc_val = self.compute_mape_acc(self.gen_observed.model,Z,X,val_length)
			weights = self.get_weights_value(self.gen.hawkes_layer)
			print {
				'source':'full train one batch',
				'batch_dis':it_dis/b_size_dis,
				'batch_gan':it_gan/b_size_gan,
				'loss_dis':dis_his.history['loss'],
				'loss_gan':gan_his.history['loss'],
				'mape':mape_acc_test['mape'],
				'acc':mape_acc_test['acc'],
				'acc_vary':mape_acc_test['acc_vary'],
				'mape_val':mape_acc_val['mape'],
				'acc_val':mape_acc_val['acc'],
				'LL':likelihood,
				'alpha':weights['mean_alpha'],
				'spont':weights['mean_spont'],
				'theta':weights['mean_theta'],
				'w':weights['mean_w'],
			}
			sys.stdout.flush()

			it_dis += b_size_dis
			it_gan += b_size_gan

	def compute_likelihood(self):
		return 0.0

	def compute_mape_acc(self,generator,z,x,test_length):
		g_z = generator.predict(z,batch_size=1)
		assert g_z.shape == x.shape
		count_g_z = np.sum(np.sum(g_z,3),2)
		count_x = np.sum(np.sum(x,3),2)
		for i in range(1,g_z.shape[1]):
			count_g_z[:,i] += count_g_z[:,i-1]
			count_x[:,i] += count_x[:,i-1]
		count_g_z = count_g_z[:,-test_length:]
		count_x = count_x[:,-test_length:]

		mape = np.mean(np.abs(count_g_z - count_x)/(count_x + 0.1),0)
		acc = np.mean(np.abs(count_g_z - count_x)/(count_x + 0.1) < 0.3,0)
		acc_v = {}
		for key in [0.35,0.3,0.25,0.2,0.1]:
			acc_v[str(key)] = np.mean(np.abs(count_g_z - count_x)/(count_x + 0.1) < key,0).tolist()

		return {
			'mape':mape.tolist(),
			'acc':acc.tolist(),
			'acc_vary':acc_v,
		}

	def get_weights_value(self,layer):

		import keras.backend as K
		mean_alpha = K.eval(K.mean(layer.Alpha))
		mean_w = K.eval(K.mean(layer.W))
		theta = K.eval(K.mean(layer.Theta))
		spont = K.eval(K.mean(layer.spontaneous))

		return {
			'mean_alpha':mean_alpha,
			'mean_w':mean_w,
			'mean_theta':theta,
			'mean_spont':spont,
		}

	def save_weights_value(self,layer):

		import keras.backend as K
		Alpha = K.eval(layer.Alpha)
		W = K.eval(layer.W)
		Theta = K.eval(layer.Theta)
		spont = K.eval(layer.spontaneous)

		return {
			'Alpha':Alpha,
			'W':W,
			'Theta':Theta,
			'spont':spont,
		}


	def load(self,f,test_length=10,observed_length=15,train_length=10,nb_type=1):
		# features[paper][feature], sequences[paper][day][feature]
		assert nb_type <= 2
		data = []
		pids = []
		for i,row in enumerate(csv.reader(file(f,'r'))):
			if i % 4 == 2:
				pids.append(str(row[0]))
				row = [float(row[1])]
			elif i % 4 == 0 or i % 4 == 1:
				row = [float(x) for x in row[1:]]
			elif i % 4 == 3:
				_row = [float(x) for x in row[1:]]
				_max = max(_row)
				_min = min(_row)
				row = [(x - _min)/float(_max - _min) for x in _row]
			data.append(row)
		
		I = int(len(data)/4)
		#train_seq = []
		# test_seq = []
		full_sequences = []
		observed_sequences = []
		train_sequences = []
		features = []
		publish_years = []
		for i in range(I):
			publish_year = data[i * 4 + 2][0]
			self_seq = data[i * 4]
			nonself_seq = data[i * 4 + 1]
			feature = data[i * 4 + 3]
			#time_seq = self_seq + nonself_seq
			#time_seq.sort()

			sequence = []
			for year in range(int(test_length + observed_length)) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				if nb_type == 2: sequence.append([[self_count],[nonself_count]])
				if nb_type == 1: sequence.append([[self_count + nonself_count]])
			if nb_type == 2: 
				sequence[0][0][0] += len([x for x in self_seq if -1 < x and x < 0])
				sequence[0][1][0] += len([x for x in nonself_seq if -1 < x and x < 0])
			if nb_type == 1: 
				sequence[0][0][0] += len([x for x in self_seq if -1 < x and x < 0])
				sequence[0][0][0] += len([x for x in nonself_seq if -1 < x and x < 0])
			full_sequences.append(sequence)

			features.append(feature)
			publish_years.append(publish_year)

			sequence = []
			for year in range(observed_length) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				if nb_type == 2: sequence.append([[self_count],[nonself_count]])
				if nb_type == 1: sequence.append([[self_count + nonself_count]])
			if nb_type == 2: 
				sequence[0][0][0] += len([x for x in self_seq if -1 < x and x < 0])
				sequence[0][1][0] += len([x for x in nonself_seq if -1 < x and x < 0])
			if nb_type == 1: 
				sequence[0][0][0] += len([x for x in self_seq if -1 < x and x < 0])
				sequence[0][0][0] += len([x for x in nonself_seq if -1 < x and x < 0])
			observed_sequences.append(sequence)

			sequence = []
			for year in range(train_length) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				if nb_type == 2: sequence.append([[self_count],[nonself_count]])
				if nb_type == 1: sequence.append([[self_count + nonself_count]])
			if nb_type == 2: 
				sequence[0][0][0] += len([x for x in self_seq if -1 < x and x < 0])
				sequence[0][1][0] += len([x for x in nonself_seq if -1 < x and x < 0])
			if nb_type == 1: 
				sequence[0][0][0] += len([x for x in self_seq if -1 < x and x < 0])
				sequence[0][0][0] += len([x for x in nonself_seq if -1 < x and x < 0])
			train_sequences.append(sequence)

		superparams = {}
		# print {
		# 	# 'np.mean(np.array(full_sequences)[:,15:])':np.mean(np.array(full_sequences)[:,15:]),
		# 	'np.mean(np.array(full_sequences)[:,0:15])':np.mean(np.array(full_sequences)[:,0:15]),
		# 	'np.mean(np.array(observed_sequences)[:,0:15])':np.mean(np.array(observed_sequences)[:,0:15]),
		# }
		# exit()
		return full_sequences,observed_sequences,train_sequences,features,publish_years,pids,superparams


# yield
if __name__ == '__main__':
	with open('../log/train_gan.log','w') as f:
		old_stdout = sys.stdout
		sys.stdout = f
		gan = HawkesGAN()
		try:
			gan.gen.sequence_weights = json.load(open('../data/paper3.pretrain.early_stop.sequence_weights.json'))
		except:
			loaded = gan.gen.load('../data/paper3.txt')
			gan.gen.pre_train(*loaded,max_outer_iter=1,alpha_iter=1,w_iter=1)
			with open('../log/paper3.pretrain.early_stop.sequence_weights.json.log','w') as fw:
				json.dump(gan.gen.sequence_weights,fw)
		# exit()
		loaded = gan.load('../data/paper3.txt')
		gan.full_train(*loaded,mse_weight=1.,gan_weight=0.)
		sys.stdout = old_stdout