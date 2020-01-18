#coding:utf-8
import numpy
import numpy as np
import os,sys,time
import json,csv
import cPickle as pickle
import copy
numpy.random.seed(1337)


class HawkesGenerator(object):
	def __init__(self):
		self.nb_type = None
		self.delta = 1.
		self.params = {}
		self.sequence_weights = None
		self.model = None
		self.hawkes_layer = None

	def pre_train(self,sequences,features,publish_years,pids,threshold,cut=15,predict_year=None,
			max_iter=0,max_outer_iter=100,alpha_iter=3,w_iter=30,val_length=5,early_stop=None,keep_weight_sync=True):
		""" 
			cut == observed length == T / delta, is a non-dimensional value
			predict_year is the firt year to begin predict
			cut_point == train_time == T == pred_year - pub_year == cut * delta, where delta is the value of scale (should be 1).
			At least one out of cut and predict_year should be specified.
		"""
		if cut is None:
			T = numpy.array([predict_year - publish_year for publish_year in publish_years],dtype=float)
			train_times = T
		else :
			T = numpy.array([float(cut)] * len(publish_years),dtype=float)
			train_times = T
			predict_year = -1

		for sam in range(len(sequences)):
			sequence = sequences[sam]
			train_sequence = [x for x in sequence if x[0] < train_times[sam]]
			if len(train_sequence) == 0:
				sys.stderr.write(str({
					'error info':'len(train_sequence) == 0',
					'seq_index':sam,
					'pid':pids[sam],
				}) + '\n')
				sys.stderr.flush()
				sequences[sam] = [(0.,0)] + sequence # this should not affect the accuracy of model

		[train_count,num_feature] = [len(features),len(features[0])] 
		nb_type = self.nb_type
		Z = numpy.asmatrix(numpy.ones([num_feature,nb_type]))
		U = numpy.asmatrix(numpy.zeros([num_feature,nb_type]))
		
		beta = numpy.asmatrix(numpy.ones([num_feature,nb_type]))
		Alpha = numpy.ones([train_count,nb_type,nb_type]).tolist()

		W1 = (numpy.zeros([train_count,nb_type]) + 0.05).tolist()
		W2 = numpy.ones([train_count,nb_type]).tolist()


		init_time = time.time()
		init_clock = time.clock()

		likelihood = self.compute_likelihood(beta,train_count,features,W1,W2,Alpha,sequences,train_times)
		self.update_params(pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2,cut=cut)
		mape_acc = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,cut=cut)
		mape_acc_val = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,
			duration=val_length,pred_year=predict_year-val_length,cut=cut-val_length)

		iterations = 1
		print {
			'source':'initial',
			'outer_iter':0,
			'inner_iter':0,
			'time':time.time() - init_time,
			'clock':time.clock() - init_clock,
			'LL':likelihood,
			'w1':numpy.mean(W1),
			'w2':numpy.mean(W2),
			'mean_alpha':numpy.mean(Alpha),
			'mean_beta':numpy.mean(beta),
			'mape':mape_acc['mape'],
			'acc':mape_acc['acc'],
			'acc_vary':mape_acc['acc_vary'],
			'mape_val':mape_acc_val['mape'],
			'acc_val':mape_acc_val['acc'],
		}
		sys.stdout.flush()

		for outer_times in range(max_outer_iter):

			# print 'step 1 : update Alpha,beta ...'

			for times in range(alpha_iter):
				D = numpy.asmatrix(numpy.zeros([num_feature,nb_type]))
				E = numpy.asmatrix(numpy.zeros([num_feature,nb_type]))
				for sam in range(train_count): 
					s = sequences[sam]
					s = [x for x in s if x[0] < train_times[sam]]
					n = len(s)
					fea = numpy.mat(features[sam])
					sw1 = W1[sam]
					sw2 = W2[sam]
					salpha = Alpha[sam]

					# Expectation-step
					psi = [numpy.multiply(fea,beta[:,s[0][1]].T) / float(fea * beta[:,s[0][1]])] # all psi in AAAI paper
					# phi = [0.] # all phi in the AAAI paper
					Phi = numpy.asmatrix(numpy.zeros([nb_type,nb_type])) # sum of all phi
					old_sum2 = 0.
					for i in range(1,n):
						w1 = sw1[s[i][1]]
						w2 = sw2[s[i][1]]
						alpha = salpha[s[i][1]][s[i-1][1]]
						betam = beta[:,s[i][1]]
						mu = float(fea * betam) * numpy.exp(-w1*s[i][0])
						sum1 = mu
						sum2 = (old_sum2 + alpha) * numpy.exp(-w2*(s[i][0]-s[i-1][0]))
						old_sum2 = sum2
						lambda_ = sum1 + sum2
						psi += [numpy.multiply(fea,betam.T) * numpy.exp(-w1*s[i][0]) / float(lambda_)]
						Phi[s[i][1],s[i-1][1]] += 1. - mu/float(lambda_)

					# Minimization-step
					for m in range(self.nb_type):
						w1 = sw1[m]
						D[:,m] += numpy.sum([x for (j,x) in enumerate(psi) if s[j][1] == m],axis=0).T
						E[:,m] += fea.T * (1. - numpy.exp(- w1 * T[sam])) / float(w1)

					for m in range(self.nb_type):
						w2 = sw2[m]
						for d in range(self.nb_type):
							alpha1 = Phi[m,d]
							sm = numpy.mat([x[0] for x in s if x[1] == d])
							alpha2 = numpy.sum(1. - numpy.exp(-w2*(T[sam] - sm))) / float(w2)
							salpha[m][d] = alpha1/float(alpha2) if alpha2 != 0. else 1.
							if alpha2 == 0.0:
								sys.stderr.write(str({
									'error info':'alpha2 == 0',
									'seq_index':sam,
									'pid':pids[sam],
									'm':m,
									'd':d,
									'alpha2':alpha2,
									'alpha1':alpha1,
									'salpha':salpha,
								}) + '\n')

					Alpha[sam] = salpha

				penalty = 5.
				reg_beta = 20.
				E_ = E + penalty * (U - Z)
				beta = (numpy.sqrt(numpy.multiply(E_,E_) + 4 * penalty * D) - E_) / float(2 * penalty)

				Z = self.shrinkage(beta + U, reg_beta/float(penalty)) 
				U += beta - Z

				likelihood = self.compute_likelihood(beta,train_count,features,W1,W2,Alpha,sequences,train_times)
				self.update_params(pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2,cut=cut)
				mape_acc = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,cut=cut)
				mape_acc_val = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,
					duration=val_length,pred_year=predict_year-val_length,cut=cut-val_length)

				print {
					'source':'update Alpha beta',
					'outer_iter':outer_times,
					'inner_iter':times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':likelihood,
					'w1':numpy.mean(W1),
					'w2':numpy.mean(W2),
					'mean_alpha':numpy.mean(Alpha),
					'mean_beta':numpy.mean(beta),
					'mape':mape_acc['mape'],
					'acc':mape_acc['acc'],
					'acc_vary':mape_acc['acc_vary'],
					'mape_val':mape_acc_val['mape'],
					'acc_val':mape_acc_val['acc'],
				}
				sys.stdout.flush()
				iterations += 1
				if early_stop is not None and iterations >= early_stop: break

			# print 'step 2 : update w by gradient descent ...'

			if early_stop is not None and iterations >= early_stop: break
			step_size = 1e-2
			for times in range(w_iter):
				step_size /= 1 + 10 * step_size
				for sam in range(train_count):
					s = sequences[sam]
					s = [x for x in s if x[0] < train_times[sam]]
					s_full = numpy.mat([x[0] for x in s])
					n = len(s)
					fea = numpy.mat(features[sam])
					sw1 = W1[sam]
					sw2 = W2[sam]
					salpha = Alpha[sam]
					pw1 = 0.
					pw2 = 0.
					old_sum2 = 0.
					for i in range(1,n):
						w1 = sw1[s[i][1]]
						w2 = sw2[s[i][1]]
						alpha = salpha[s[i][1]][s[i-1][1]]
						betam = beta[:,s[i][1]]
						s_past = numpy.mat([x[0] for x in s[0:i]])

						mu = float(fea * betam) * numpy.exp(-w1*s[i][0])
						sum1 = mu
						sum2 = (old_sum2 + alpha) * numpy.exp(-w2*(s[i][0]-s[i-1][0]))
						old_sum2 = sum2
						lambda_ = sum1 + sum2
						pw2t = alpha * numpy.sum(numpy.multiply(
							numpy.exp(-w2*(s[i][0]-s_past)),-(s[i][0]-s_past)))
						pw1 += float(fea * betam) * numpy.exp(-w1*s[i][0]) * (-s[i][0]) / float(lambda_)
						pw2 += float(pw2t) / float(lambda_)
					pw1 -= float(fea * betam) * (
						numpy.exp(-w1*T[sam]) * T[sam] * w1 - (1. - numpy.exp(-w1*T[sam]))) / float(w1**2)

					second_order = numpy.multiply(numpy.exp(-w2*(T[sam]-s_full)),(T[sam]-s_full)*w2)
					first_order = 1. - numpy.exp(-w2*(T[sam]-s_full))
					pw2 -= alpha * numpy.sum((second_order - first_order) / float(w2**2))
					step1 = pw1 if numpy.abs(pw1) < 1. else numpy.sign(pw1)
					step2 = pw2 if numpy.abs(pw2) < 1. else numpy.sign(pw2)
					W1[sam] = (numpy.array(sw1) + step_size*step1).tolist()
					W2[sam] = (numpy.array(sw2) + step_size*step2).tolist()


				likelihood = self.compute_likelihood(beta,train_count,features,W1,W2,Alpha,sequences,train_times)
				self.update_params(pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2,cut=cut)
				mape_acc = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,cut=cut)
				mape_acc_val = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,
					duration=val_length,pred_year=predict_year-val_length,cut=cut-val_length)

				print {
					'source':'update w1 w2',
					'outer_iter':outer_times,
					'inner_iter':times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':likelihood,
					'w1':numpy.mean(W1),
					'w2':numpy.mean(W2),
					'mean_alpha':numpy.mean(Alpha),
					'mean_beta':numpy.mean(beta),
					'mape':mape_acc['mape'],
					'acc':mape_acc['acc'],
					'acc_vary':mape_acc['acc_vary'],
					'mape_val':mape_acc_val['mape'],
					'acc_val':mape_acc_val['acc'],
				}
				sys.stdout.flush()
				iterations += 1
				if early_stop is not None and iterations >= early_stop: break

			# print 'step 3 : check terminate condition ...'

			if early_stop is not None and iterations >= early_stop: break

		self.update_sequence_weights(pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2)
		return self.params

	def update_params(self,pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2,cut=None):

		patent = {}
		for item in range(len(Alpha)):
			a_patent = {}
			a_patent['alpha'] = Alpha[item]
			a_patent['w1'] = W1[item]
			a_patent['w2'] = W2[item]
			a_patent['fea'] = features[item]
			a_patent['cite'] = sequences[item]
			a_patent['year'] = publish_years[item]
			patent[pids[item]] = a_patent

		params = {}
		params['predict_year'] = predict_year
		params['cut_point'] = cut
		params['train_count'] = len(patent)
		params['beta'] = beta.tolist()
		params['patent'] = patent
		params['feature_name'] = []

		self.params = params

	def update_sequence_weights(self,pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2):

		result = []
		for i in range(len(pids)):
			seq = {}
			fea = numpy.mat(features[i])
			beta = numpy.mat(beta)
			seq['seq_id'] = i
			seq['paper_id'] = pids[i]
			seq['theta'] = W1[i]
			seq['w'] = W2[i]
			seq['alpha'] = Alpha[i]
			seq['fea'] = features[i]
			seq['beta'] = beta.tolist()
			seq['spont'] = (fea*beta).tolist()[0]
			result.append(seq)

		self.sequence_weights = result


	def compute_likelihood(self,beta,train_count,features,W1,W2,Alpha,sequences,train_times):
		likelihood = 0.
		for item in range(train_count):
			fea = numpy.mat(features[item])
			sw1 = W1[item]
			sw2 = W2[item]
			salpha = Alpha[item]
			s = sequences[item]
			s = [x for x in s if x[0] < train_times[item]]
			obj = self.calculate_objective((fea*beta).tolist()[0],sw1,salpha,sw2,s,train_times[item])
			likelihood -= obj
		likelihood /= train_count
		return likelihood


	def shrinkage(self,vector,kappa):
		mat1 = vector - kappa
		mat2 = -vector - kappa

		mat1 = numpy.clip(mat1,0.,numpy.inf)
		mat2 = numpy.clip(mat2,0.,numpy.inf)
		return mat1 - mat2

	def calculate_objective(self,spont,sw1,salpha,sw2,events,train_time):
		T = train_time
		N = len(events)
		s = events
		spontaneous = spont[s[0][1]]
		w1 = sw1[s[0][1]]
		obj = numpy.log(spontaneous*numpy.exp(-w1*s[0][0]))

		old_sum2 = 0.
		for i in range(1,N):
			spontaneous = spont[s[i][1]]
			w1 = sw1[s[i][1]]
			w2 = sw2[s[i][1]]
			alpha = salpha[s[i][1]][s[i-1][1]]
			mu = spontaneous*numpy.exp(-w1*s[i][0])
			sum1 = mu
			sum2 = (old_sum2 + alpha)*numpy.exp(-w2*(s[i][0]-s[i-1][0]))
			old_sum2 = sum2
			obj += numpy.log(sum1+sum2)

		for m in range(self.nb_type):
			w2 = sw2[m]
			activate_sum = 0.
			for i in range(N):
				alpha = salpha[m][s[i][1]]
				activate = numpy.exp(-w2*(T-s[i][0]))
				activate_sum += (1. - activate) * alpha / float(w2)
			obj -= activate_sum

		for m in range(self.nb_type):
			spontaneous = spont[m]
			w1 = sw1[m]
			obj -= (spontaneous/w1) * (1 - numpy.exp(-w1*T))

		return obj


	def compute_mape_acc(self,model,sequences,features,publish_years,pids,threshold,duration=10,pred_year=None,cut=None):
		patents = self.params['patent']
		g_z = []
		x = []
		for i, key in enumerate(patents):
			if cut is None:
				pred_year = pred_year or self.params['predict_year']
			else:
				pred_year = cut + int(float((patents[key]['year'])))
			real = self.real_one(key,cut_point=pred_year - int(float((patents[key]['year']))) + duration)
			pred = self.predict_one(key,duration=duration,pred_year=pred_year)
			x.append(real)
			g_z.append(pred)

		x = numpy.array(x)
		g_z = numpy.array(g_z)
		assert g_z.shape == x.shape
		print g_z
		count_g_z = np.sum(g_z,2)
		count_x = np.sum(x,2)
		for i in range(1,g_z.shape[1]):
			count_g_z[:,i] += count_g_z[:,i-1]
			count_x[:,i] += count_x[:,i-1]
		count_g_z = count_g_z[:,-duration:]
		count_x = count_x[:,-duration:]

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


	def predict_one(self,_id,duration,pred_year):
		try:
			patent = self.params['patent'][str(_id)]
		except KeyError,e:
			return None
		sw1 = patent['w1']
		salpha = patent['alpha']
		sw2 = patent['w2']
		fea = numpy.mat(patent['fea'])
		ti = patent['cite']
		beta = numpy.mat(self.params['beta'])

		cut_point = pred_year - int(float((patent['year'])))
		tr = [x for x in ti if x[0] < cut_point]

		return self.predict_year_by_year(tr,cut_point,duration,
			(fea*beta).tolist()[0],sw1,salpha,sw2)

	def real_one(self,_id,cut_point):
		try:
			patent = self.params['patent'][str(_id)]
		except KeyError,e:
			return None

		cites = patent['cite']
		N = len(cites)
		real_seq = numpy.zeros([cut_point,self.nb_type]).tolist()

		# copy unit
		left = 0
		for t in range(cut_point):
			while left < N and cites[left][0] < t + 1:
				real_seq[t][cites[left][1]] += 1.
				left += 1

		return real_seq


	def predict_year_by_year(self,tr,cut_point,duration,spont,sw1,salpha,sw2):
		N = len(tr)
		pred_seq = numpy.zeros([cut_point,self.nb_type]).tolist()

		# copy unit
		left = 0
		for t in range(cut_point):
			while left < N and tr[left][0] < t + 1:
				pred_seq[t][tr[left][1]] += 1.
				left += 1

		# prediction_unit
		spont = numpy.mat(spont)
		theta = numpy.mat(sw1)
		w = numpy.mat(sw2)
		alpha = numpy.mat(salpha)
		for t in range(cut_point,cut_point+duration):
			term1 = numpy.multiply(spont/theta,
				(numpy.exp(-theta*t) - numpy.exp(-theta*(t + 1.))))
			# triggering_unit
			effect = numpy.mat([0.] * self.nb_type)
			for tao in range(t):
				counts = numpy.mat(pred_seq[tao])
				effect_unit = numpy.multiply(counts,
					(numpy.exp(- w * (t - tao)) - numpy.exp(- w * (t + 1 - tao))))
				effect += effect_unit
			term2 = (alpha * (effect.T)).T / w
			pred_seq.append((term1 + term2).tolist()[0])

		return pred_seq

	def compute_early_stopping(curve):
		early_stop = -1
		moving_average = curve[0]
		k = 20.
		th = 0.001
		for n in range(1,len(curve)):
			av = curve[n] * (1./k) + moving_average * (k - 1.) / k
			if moving_average - av <= th:
				early_stop = n
				break
			moving_average = av
		self.early_stop = early_stop
		return early_stop


	def create_trainable_model(self,sequences, pred_length, proxy_layer=None, need_noise_dropout=False, stddev=5.,sample_stddev=None):
		from keras.layers import Input, GaussianNoise
		from keras.models import Model

		from pp_layer import HawkesLayer

		if self.sequence_weights is None:
			sys.stderr.write(str({
				'error info':'unpretrained generator',
			}) + '\n')
			sys.stderr.flush()

		x = Input(batch_shape=(1,1), dtype='int32')
		hawkes_layer = HawkesLayer(sequences,pred_length,sequence_weights=self.sequence_weights,proxy_layer=proxy_layer,sample_stddev=sample_stddev)
		y = hawkes_layer(x)
		if need_noise_dropout == True:
			y = GaussianNoise(stddev)(y)

		model = Model(inputs=[x], outputs=[y], name='hawkes_output')

		self.model = model
		self.hawkes_layer = hawkes_layer
		return model

	def load(self,f,nb_type=1):
		"""
			data format : types | start time | feature
			the value of time scale should be 1. 
			if the distance between adjacent scale mark is too long, it should be splitted.
		"""
		self.nb_type = nb_type
		if nb_type > 1:
			data = []
			pids = []
			span = nb_type + 2
			for i,row in enumerate(csv.reader(file(f,'r'))):
				if i % span == span - 2: # start time
					pids.append(str(row[0]))
					row = [float(row[1])]
				elif i % span < span - 2: # event types
					row = [float(x) for x in row[1:]]
				elif i % span == span - 1: # profile feature
					_row = [float(x) for x in row[1:]]
					_max = max(_row)
					_min = min(_row)
					row = [(x - _min)/float(_max - _min) for x in _row]
				data.append(row)
			
			I = int(len(data)/span)
			train_seq = []
			test_seq = []
			sequences = []
			features = []
			publish_years = []
			for i in range(I):
				publish_year = data[i * span + span - 2][0]
				feature = data[i * span + span - 1]
				# time_seq = self_seq + nonself_seq
				# time_seq.sort()
				sequence = data[(i*span):(i*span + span - 2)]

				for j in range(span - 2):
					sequence[j] = [float(int(x)) for x in sequence[j] if x > -1.]

				time_seq = []
				for k,seq in enumerate(sequence):
					time_seq += [(x,k) for x in seq]
				time_seq.sort(key=lambda x:x[0])

				sequences.append(time_seq)
				features.append(feature)
				publish_years.append(publish_year)

			threshold = 0.01
			return sequences,features,publish_years,pids,threshold
		else:
			span = 0
			key_prev = ''
			for i,row in enumerate(csv.reader(file(f,'r'))):
				key = str(row[0])
				if key == key_prev or key_prev == '': 
					span += 1
					key_prev = key
				else:
					break

			data = []
			pids = []
			for i,row in enumerate(csv.reader(file(f,'r'))):
				if i % span == span - 2: # start time
					pids.append(str(row[0]))
					row = [float(row[1])]
				elif i % span < span - 2: # event types
					row = [float(x) for x in row[1:]]
				elif i % span == span - 1: # profile feature
					_row = [float(x) for x in row[1:]]
					_max = max(_row)
					_min = min(_row)
					row = [(x - _min)/float(_max - _min) for x in _row]
				data.append(row)
			
			I = int(len(data)/span)
			train_seq = []
			test_seq = []
			sequences = []
			features = []
			publish_years = []
			for i in range(I):
				publish_year = data[i * span + span - 2][0]
				feature = data[i * span + span - 1]
				# time_seq = self_seq + nonself_seq
				# time_seq.sort()
				sequence = data[(i*span):(i*span + span - 2)]

				for j in range(span - 2):
					sequence[j] = [float(int(x)) for x in sequence[j] if x > -1.]

				time_seq = []
				for seq in sequence:
					time_seq += [(x,0) for x in seq]
				time_seq.sort(key=lambda x:x[0])

				sequences.append(time_seq)
				features.append(feature)
				publish_years.append(publish_year)

			threshold = 0.01
			return sequences,features,publish_years,pids,threshold



if __name__ == '__main__':
	with open('../log/train.log','w') as f:
		sys.stdout = f
		predictor = HawkesGenerator()
		# loaded = predictor.load('../data/paper3.txt')
		loaded = predictor.load('../data/atmerror2.txt',nb_type=2)
		# print loaded[0][0]
		# print loaded[0][1]
		# print loaded[0][2]
		# exit()
		model = predictor.pre_train(*loaded,alpha_iter=1,w_iter=1)
		
		pass