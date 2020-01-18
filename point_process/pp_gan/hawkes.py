#coding:utf-8
import numpy
import os,sys,time
import json,csv
numpy.random.seed(1337)

class MHawkes(object):
	def __init__(self):
		pass

	def train(self, train_seq, test_seq, superparams, initparams,max_iter=0,max_outer_iter=100):

		model = {}
		model['superparams'] = superparams
		I = len(train_seq)
		M = 2
		sigma = superparams['sigma']
		model['mu'] = numpy.random.random(M)
		model['AA'] = numpy.random.random(M**2)
		model['theta'] = initparams['theta']
		model['w'] = initparams['w']
		model['M'] = M
		model['I'] = I

		reg_alpha = numpy.zeros(model['AA'].shape)

		init_time = time.time()
		init_clock = time.clock()
		iteration = 1
		outer_times = 1

		L_converge = False
		while not L_converge:
			Q_converge = False
			while not Q_converge:
				LL = 0

				Bmu = numpy.zeros(model['mu'].shape)
				Cmu = 0.0

				Gamma = numpy.random.random((M**2,I))
				for i in range(I):
					T = train_seq[i]['times'][-1] + 0.01
					times = numpy.array(train_seq[i]['times'],dtype=float)
					dims = numpy.array(train_seq[i]['dims'],dtype=int)
					N = len(times)
					Cmu += T

					B = sigma * reg_alpha
					B = B.reshape((M,M))
					C = numpy.zeros(B.shape)
					AA = model['AA'].reshape(M,M)
					w = model['w']

					LL += T * numpy.sum(model['mu'])

					for j in range(0,N):
						t_j = times[j]
						m_j = dims[j]
						int_g = self.G(w,T - times[j])
						B[:,m_j] += int_g
						LL += numpy.sum(int_g * AA[:,m_j])
						_lambda = model['mu'][m_j]
						if j == 0:
							psi = 1.0
						else:
							psi = _lambda
							t_k = times[0:j]
							m_k = dims[0:j]
							idx = [_idx for _idx in range(len(t_k)) if t_j - t_k[_idx] <= superparams['impact_period']]
							if len(idx) == 0:
								psi /= _lambda
							else:
								_g = self.g(w,t_j - t_k[idx])
								_a = AA[m_j, m_k[idx]]
								phi = _g * _a
								_lambda += numpy.sum(phi)
								psi /= _lambda
								phi /= _lambda

								for k in range(len(idx)):
									C[m_j,m_k[idx[k]]] += phi[k]
							LL -= numpy.log(_lambda)
						Bmu[m_j] += psi
					AA = (- B + numpy.sqrt(B ** 2 + 8 * sigma * C )) / (4 * sigma) * numpy.sign(C)
					Gamma[:,i] = AA.reshape(M**2)

				mu = Bmu / Cmu
				model['mu'] = mu

				# check convergence
				AA = numpy.mean(Gamma,1)
				error = numpy.sum(numpy.abs(model['AA'] - AA)) / numpy.sum(model['AA'])
				# print json.dumps({'iter':iteration,'time':time.time() - init_time,'clock':time.clock() - init_clock,'LL':LL,'error':error})
				print json.dumps({
					'iter':iteration,
					'outer_iter':outer_times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':LL,
					'error':error,
					'mean_alpha':AA.tolist(),
					# 'mean_alpha_self':numpy.mean(Gamma),
					# 'mean_alpha_nonself':numpy.mean(Gamma),
				})
				iteration += 1

				model['AA'] = AA
				if iteration > max_iter or error < superparams['thres']:
					Q_converge = True
					break
				else:
					Q_converge = False


			if outer_times > max_outer_iter:# or abs(old_tlikelihood-tlikelihood)< threshold/10.0:
				L_converge = True
			else:
				L_converge = False
			outer_times += 1
		return model

	def G(self,w,t):
		return (1 - numpy.exp(- w * t)) / w

	def g(self,w,t):
		return numpy.exp(- w * t)

	def predict(self, model, train_seq, test_seq, superparams, initparams):
		M = model['M']
		pred_seqs = []
		pred_seqs_self = []
		pred_seqs_nonself = []

		real_seqs = []
		real_seqs_self = []
		real_seqs_nonself = []
		mapes = []

		mapes_self = []
		mapes_nonself = []
		I = len(train_seq)

		for i in range(I):
			T = train_seq[i]['times'][-1] + 0.01
			times = numpy.array(train_seq[i]['times'],dtype=float)
			dims = numpy.array(train_seq[i]['dims'],dtype=int)
			features = numpy.array(train_seq[i]['features'],dtype=float)
			N = len(times)
			N_self = len([x for x in dims if x == 0])
			N_nonself = len([x for x in dims if x == 1])
			if N != N_self + N_nonself:
				print 'N != N_self + N_nonself'
				exit()

			AA = model['AA'].reshape(M,M)
			w = model['w']

			duration = model['superparams']['duration']
			mape = []
			mape_self = []
			mape_nonself = []

			pred_seq = []
			pred_seq_self = []
			pred_seq_nonself = []

			real_seq = []
			real_seq_self = []
			real_seq_nonself = []

			for year in range(duration+1):
				LL = 0
				LL_self = 0
				LL_nonself = 0
				for j in range(N):
					m_j = dims[j]
					int_g = self.G(w,T + year - times[j]) - self.G(w,T - times[j])
					LL += numpy.sum(int_g * AA[:,m_j])
					LL_self += int_g * AA[0,m_j]
					LL_nonself += int_g * AA[1,m_j]

				LL += year * numpy.sum(model['mu'])
				LL_self += year * numpy.sum(model['mu'])
				LL_nonself += year * numpy.sum(model['mu'])


				pred = N + LL
				pred_self = N_self + LL_self
				pred_nonself = N_nonself + LL_nonself
				real = N + len([x for x in test_seq[i]['times'] if x + model['superparams']['cut_point'] < T + year])
				real_self = N_self + len([x for _x,x in enumerate(test_seq[i]['times']) if x + model['superparams']['cut_point'] < T + year and test_seq[i]['dims'][_x] == 0])
				real_nonself = N_nonself + len([x for _x,x in enumerate(test_seq[i]['times']) if x + model['superparams']['cut_point'] < T + year and test_seq[i]['dims'][_x] == 1])
				mape.append(abs(pred - real) / float(real + 0.1))
				mape_self.append(abs(pred_self - real_self) / float(real_self + 0.1))
				mape_nonself.append(abs(pred_nonself - real_nonself) / float(real_nonself + 0.1))

				pred_seq.append(pred)
				pred_seq_self.append(pred_self)
				pred_seq_nonself.append(pred_nonself)

				real_seq.append(real)
				real_seq_self.append(real_self)
				real_seq_nonself.append(real_nonself)

			mapes.append(mape)
			mapes_self.append(mape_self)
			mapes_nonself.append(mape_nonself)

			pred_seqs.append(pred_seq)
			pred_seqs_self.append(pred_seq_self)
			pred_seqs_nonself.append(pred_seq_nonself)

			real_seqs.append(real_seq)
			real_seqs_self.append(real_seq_self)
			real_seqs_nonself.append(real_seq_nonself)

		av_mape = numpy.mean(numpy.array(mapes),0)
		av_acc = numpy.mean(numpy.array(mapes) < model['superparams']['epsilon'],0)

		av_mape_self = numpy.mean(numpy.array(mapes_self),0)
		av_acc_self = numpy.mean(numpy.array(mapes_self) < model['superparams']['epsilon'],0)

		av_mape_nonself = numpy.mean(numpy.array(mapes_nonself),0)
		av_acc_nonself = numpy.mean(numpy.array(mapes_nonself) < model['superparams']['epsilon'],0)

		result = {
			'av_mape':av_mape.tolist(),
			'av_acc':av_acc.tolist(),
			'av_mape_self':av_mape_self.tolist(),
			'av_acc_self':av_acc_self.tolist(),
			'av_mape_nonself':av_mape_nonself.tolist(),
			'av_acc_nonself':av_acc_nonself.tolist(),
			# 'mapes':mapes,
			# 'mapes_self':mapes_self,
			# 'mapes_nonself':mapes_nonself,
			# 'pred_seqs':pred_seqs,
			# 'pred_seqs_self':pred_seqs_self,
			# 'pred_seqs_nonself':pred_seqs_nonself,
			# 'real_seqs':real_seqs,
			# 'real_seqs_self':real_seqs_self,
			# 'real_seqs_nonself':real_seqs_nonself,
			}
		
		for key in result:
			result[key].pop(0)
		
		return result

	def predict_one(self,patent_id):
		pass

	def load(self,f,cut=15):
		data = []
		for i,row in enumerate(csv.reader(file(f,'r'))):
			if i % 4 == 2:
				row = [float(row[1])]
			elif i % 4 == 0 or i % 4 == 1:
				row = [float(x) for x in row[1:]]
			elif i % 4 == 3:
				_row = [float(x) for x in row[1:]]
				_max = max(_row)
				_min = min(_row)
				row = [(x - _min)/float(_max - _min) for x in _row]
			data.append(row)

		T = cut - 1
		lines = 4
		I = int(len(data)/lines)
		train_seq = []
		test_seq = []
		for i in range(I):
			publish_year = data[i * lines + 2]
			self_seq = data[i * lines]
			nonself_seq = data[i * lines + 1]
			feature = data[i * lines + 3]
			varying = data[(i * lines + 4):(i * lines + lines)]

			time_seq = self_seq + nonself_seq
			dim_seq = ([0] * len(self_seq)) + ([1] * len(nonself_seq))
			S = zip(time_seq,dim_seq)
			S = sorted(S,key=lambda x:x[0])
			Y = [x[0] for x in S]
			dim_seq = [x[1] for x in S]
			cut_point = T - 1
			time_train = [y for y in Y if y <= cut_point]
			dim_train = [e for i,e in enumerate(dim_seq) if Y[i] <= cut_point]
			if len(time_train) < 5:
				continue
			_dict = {}
			_dict['times'] = time_train
			_dict['dims'] = dim_train
			_dict['features'] = feature
			_dict['varying'] = varying
			_dict['publish_year'] = publish_year
			train_seq.append(_dict)

			_dict = {}
			_dict['times'] = [y - cut_point for y in Y if y > cut_point]
			_dict['dims'] = [e for i,e in enumerate(dim_seq) if Y[i] > cut_point]
			_dict['features'] = feature
			_dict['varying'] = varying
			_dict['publish_year'] = publish_year
			test_seq.append(_dict)


		superparams = {}
		superparams['M'] = 2
		superparams['outer'] = 20
		superparams['inner'] = 10
		superparams['K'] = len(train_seq[0]['features'])
		superparams['impact_period'] = 5
		superparams['sigma'] = 1
		superparams['thres'] = 1e-3
		superparams['cut_point'] = cut_point
		superparams['duration'] = 10
		superparams['epsilon'] = 0.3

		initparams = {}
		initparams['theta'] = 0.2
		initparams['w'] = 1.0
		initparams['b'] = 0.5


		return train_seq,test_seq,superparams,initparams




class MTPP(object):
	def __init__(self):
		pass

	def train(self, train_seq, test_seq, superparams, initparams,max_iter=0,max_outer_iter=100):

		model = {}
		model['superparams'] = superparams
		I = len(train_seq)
		K = len(train_seq[0]['features'])
		M = 2
		sigma = superparams['sigma']
		model['beta'] = numpy.random.random((M,K))
		model['Gamma'] = numpy.random.random((M**2,I))
		model['theta'] = initparams['theta']
		model['w'] = initparams['w']
		model['b'] = initparams['b']
		model['M'] = M
		model['K'] = K
		model['I'] = I

		reg_alpha = numpy.zeros(model['Gamma'].shape)
		reg_beta = numpy.zeros(model['beta'].shape)

		init_time = time.time()
		init_clock = time.clock()
		iteration = 1
		outer_times = 1

		L_converge = False
		while not L_converge:
			Q_converge = False
			while not Q_converge:
				LL = 0.0
				Gamma = numpy.array(model['Gamma'])
				beta = numpy.array(model['beta'])
				intrinsic1 = numpy.zeros((M,K))
				intrinsic2 = numpy.zeros((M,K))

				for i in range(I):
					T = train_seq[i]['times'][-1] + 0.01
					times = numpy.array(train_seq[i]['times'],dtype=float)
					dims = numpy.array(train_seq[i]['dims'],dtype=int)
					features = numpy.array(train_seq[i]['features'],dtype=float)
					v = numpy.array(train_seq[i]['varying'],dtype=float)
					N = len(times)
			
					B = sigma * reg_alpha[:,i]
					B = B.reshape((M,M))
					C = numpy.zeros(B.shape)
					D = numpy.zeros(model['beta'].shape)
					init_intensity = numpy.dot(model['beta'], features)
					init_intensities = model['beta'] * numpy.tile(features,(M,1))
					gamma = model['Gamma'][:,i].reshape(M,M)
					theta = model['theta']
					w = model['w']
					b = model['b']
			
					for j in range(0,N):
						t_j = times[j]
						m_j = dims[j]
						int_g = self.G(w,T - times[j],b)
						B[:,m_j] += int_g
						LL += numpy.sum(int_g * gamma[:,m_j])
						_lambda = init_intensity[m_j] * self.g(theta,t_j,b)
						if j == 0:
							psi = init_intensities[m_j,:] * self.g(theta,t_j,b) / _lambda
						else:
							psi = init_intensities[m_j,:] * self.g(theta,t_j,b)
							t_k = times[0:j]
							m_k = dims[0:j]
							# idx = [_idx for _idx in range(len(t_k)) if t_j - t_k[_idx] <= superparams['impact_period']]
							if len(m_k) == 0:
								psi /= _lambda
							else:
								_g = self.g(w,t_j - t_k,b)
								_a = gamma[m_j, m_k]
								phi = _g * _a
								_lambda += numpy.sum(phi)
								psi /= _lambda
								phi /= _lambda
								
								for k in range(len(m_k)):
									C[m_j,m_k[k]] += phi[k]
							LL -= numpy.log(_lambda)
						D[m_j,:] += psi
					LL += self.G(theta,T,b) * numpy.sum(init_intensity)
					
					intrinsic1 += D
					intrinsic2 += numpy.tile(features * self.G(theta,T,b), (M,1) )
					gamma = (- B + numpy.sqrt(B ** 2 + 8 * sigma * C )) / (4 * sigma) * numpy.sign(C)
					Gamma[:,i] = gamma.reshape(M**2)
				# update beta
				B = intrinsic2 + sigma * reg_beta
				beta = (- B + numpy.sqrt(B ** 2 + 4 * sigma * intrinsic1)) / (2 * sigma) * numpy.sign(intrinsic1)
				
				# check convergence
				error = numpy.sum(numpy.abs(model['Gamma'] - Gamma)) / numpy.sum(model['Gamma'])
				# print json.dumps({'iter':iteration,'time':time.time() - init_time,'clock':time.clock() - init_clock,'LL':LL,'error':error})
				print json.dumps({
					'iter':iteration,
					'outer_iter':outer_times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':LL,
					'w':w,
					'theta':theta,
					'b':b,
					'error':error,
					'mean_alpha':numpy.mean(Gamma),
				})
				iteration += 1

				model['Gamma'] = Gamma
				model['beta'] = beta
				if iteration > max_iter or error < superparams['thres']:
					Q_converge = True
					break
				else:
					Q_converge = False

			# update theta & w & b
			LL_old = LL
			# step = 10**-4
			# for i in range(I):
			# 	T = train_seq[i]['times'][-1] + 0.01
			# 	times = numpy.array(train_seq[i]['times'],dtype=float)
			# 	dims = numpy.array(train_seq[i]['dims'],dtype=int)
			# 	features = numpy.array(train_seq[i]['features'],dtype=float)
			# 	N = len(times)
			#
			# 	init_intensity = numpy.dot(model['beta'], features)
			# 	init_intensities = model['beta'] * numpy.tile(features,(M,1))
			# 	gamma = model['Gamma'][:,i].reshape(M,M)
			# 	theta = model['theta']
			# 	w = float(model['w'])
			# 	b = float(model['b'])
			#
			# 	Lw = 0.0
			# 	for j in range(0,N):
			# 		t_j = times[j]
			# 		m_j = dims[j]
			# 		grad1 = (self.f(w,b) + self.f(w,T - t_j - b)) * w - (2.0 - numpy.exp(- w * b) - numpy.exp(- w *(T - t_j - b)))
			# 		grad1 /= w ** 2
			# 		grad1 *= numpy.sum(gamma[:,m_j])
			#
			# 		# int_g = self.G(w,T - times[j],b)
			# 		# B[:,m_j] += int_g
			# 		# LL += numpy.sum(int_g * gamma[:,m_j])
			# 		_lambda = init_intensity[m_j] * self.g(theta,t_j,b)
			# 		grad2 = 0.0
			# 		if j == 0:
			# 			pass
			# 		else:
			# 			# psi = init_intensities[m_j,:] * self.g(theta,t_j,b)
			# 			t_k = times[0:j]
			# 			m_k = dims[0:j]
			# 			# idx = [_idx for _idx in range(len(t_k)) if t_j - t_k[_idx] <= superparams['impact_period']]
			# 			if len(m_k) == 0:
			# 				pass
			# 			else:
			# 				_g = self.g(w,t_j - t_k,b)
			# 				_a = gamma[m_j, m_k]
			# 				_f = self.f(w,numpy.abs(t_j - t_k - b))
			# 				_lambda += numpy.sum(_g * _a)
			# 				grad2 += numpy.sum(_f * _a)
			# 		grad2 /= _lambda
			# 		Lw += grad1 + grad2
			#
			# 	model['w'] -= step * numpy.sign(Lw)
			if outer_times > max_outer_iter:# or abs(old_tlikelihood-tlikelihood)< threshold/10.0:
				L_converge = True
			else:
				L_converge = False
			outer_times += 1
			# if iteration > max_iter or error < superparams['thres']:
			# 	L_converge = True
			# else:
			# 	L_converge = False
		return model

	def G(self,w,t,b):
		return (1 - numpy.exp(- w * b) + 1 - numpy.exp(- w * (t - b))) / w

	def g(self,w,t,b):
		return numpy.exp(- w * numpy.abs(t - b))

	def f(self,w,x):
		return x * numpy.exp(- w * x)

	def predict(self, model, train_seq, test_seq, superparams, initparams):
		M = model['M']
		pred_seqs = []
		pred_seqs_self = []
		pred_seqs_nonself = []

		real_seqs = []
		real_seqs_self = []
		real_seqs_nonself = []		
		mapes = []

		mapes_self = []
		mapes_nonself = []
		I = len(train_seq)

		for i in range(I):
			T = train_seq[i]['times'][-1] + 0.01
			times = numpy.array(train_seq[i]['times'],dtype=float)
			dims = numpy.array(train_seq[i]['dims'],dtype=int)
			features = numpy.array(train_seq[i]['features'],dtype=float)
			N = len(times)
			N_self = len([x for x in dims if x == 0])
			N_nonself = len([x for x in dims if x == 1])
			if N != N_self + N_nonself:
				print 'N != N_self + N_nonself'
				exit()

			Musum = numpy.dot(model['beta'], features)
			gamma = model['Gamma'][:,i].reshape(M,M)
			theta = model['theta']
			w = model['w']
			b = model['b']
			
			duration = model['superparams']['duration']
			mape = []
			mape_self = []
			mape_nonself = []

			pred_seq = []
			pred_seq_self = []
			pred_seq_nonself = []

			real_seq = []
			real_seq_self = []
			real_seq_nonself = []
			
			for year in range(duration+1):
				LL = 0
				LL_self = 0
				LL_nonself = 0
				for j in range(N):
					m_j = dims[j]
					int_g = self.G(w,T + year - times[j],b) - self.G(w,T - times[j],b)
					LL += numpy.sum(int_g * gamma[:,m_j])
					LL_self += int_g * gamma[0,m_j]
					LL_nonself += int_g * gamma[1,m_j]
				LL += (self.G(theta,T + year,b) - self.G(theta,T,b)) * numpy.sum(Musum)
				LL_self += (self.G(theta,T + year,b) - self.G(theta,T,b)) * numpy.sum(Musum)
				LL_nonself += (self.G(theta,T + year,b) - self.G(theta,T,b)) * numpy.sum(Musum)

				pred = N + LL
				pred_self = N_self + LL_self
				pred_nonself = N_nonself + LL_nonself
				real = N + len([x for x in test_seq[i]['times'] if x + model['superparams']['cut_point'] < T + year])
				real_self = N_self + len([x for _x,x in enumerate(test_seq[i]['times']) if x + model['superparams']['cut_point'] < T + year and test_seq[i]['dims'][_x] == 0])
				real_nonself = N_nonself + len([x for _x,x in enumerate(test_seq[i]['times']) if x + model['superparams']['cut_point'] < T + year and test_seq[i]['dims'][_x] == 1])
				mape.append(abs(pred - real) / float(real + 0.01))
				mape_self.append(abs(pred_self - real_self) / float(real_self + 0.01))
				mape_nonself.append(abs(pred_nonself - real_nonself) / float(real_nonself + 0.01))

				pred_seq.append(pred)
				pred_seq_self.append(pred_self)
				pred_seq_nonself.append(pred_nonself)

				real_seq.append(real)
				real_seq_self.append(real_self)
				real_seq_nonself.append(real_nonself)

			mapes.append(mape)
			mapes_self.append(mape_self)
			mapes_nonself.append(mape_nonself)

			pred_seqs.append(pred_seq)
			pred_seqs_self.append(pred_seq_self)
			pred_seqs_nonself.append(pred_seq_nonself)

			real_seqs.append(real_seq)
			real_seqs_self.append(real_seq_self)
			real_seqs_nonself.append(real_seq_nonself)

		av_mape = numpy.mean(numpy.array(mapes),0)
		av_acc = numpy.mean(numpy.array(mapes) < model['superparams']['epsilon'],0)

		av_mape_self = numpy.mean(numpy.array(mapes_self),0)
		av_acc_self = numpy.mean(numpy.array(mapes_self) < model['superparams']['epsilon'],0)

		av_mape_nonself = numpy.mean(numpy.array(mapes_nonself),0)
		av_acc_nonself = numpy.mean(numpy.array(mapes_nonself) < model['superparams']['epsilon'],0)

		return {
			'av_mape':av_mape.tolist(),
			'av_acc':av_acc.tolist(),
			'av_mape_self':av_mape_self.tolist(),
			'av_acc_self':av_acc_self.tolist(),
			'av_mape_nonself':av_mape_nonself.tolist(),
			'av_acc_nonself':av_acc_nonself.tolist(),
			# 'mapes':mapes,
			# 'mapes_self':mapes_self,
			# 'mapes_nonself':mapes_nonself,
			# 'pred_seqs':pred_seqs,
			# 'pred_seqs_self':pred_seqs_self,
			# 'pred_seqs_nonself':pred_seqs_nonself,
			# 'real_seqs':real_seqs,
			# 'real_seqs_self':real_seqs_self,
			# 'real_seqs_nonself':real_seqs_nonself,
			}

	def predict_one(self,patent_id):
		pass

	def load(self,f,cut=15):
		data = []
		for i,row in enumerate(csv.reader(file(f,'r'))):
			if i % 4 == 2:
				row = [float(row[1])]
			elif i % 4 == 0 or i % 4 == 1:
				row = [float(x) for x in row[1:]]
			elif i % 4 == 3:
				_row = [float(x) for x in row[1:]]
				_max = max(_row)
				_min = min(_row)
				row = [(x - _min)/float(_max - _min) for x in _row]
			data.append(row)

		T = cut
		lines = 4
		I = int(len(data)/lines)
		train_seq = []
		test_seq = []
		for i in range(I):
			publish_year = data[i * lines + 2]
			self_seq = data[i * lines]
			nonself_seq = data[i * lines + 1]
			feature = data[i * lines + 3]
			varying = data[(i * lines + 4):(i * lines + lines)]

			time_seq = self_seq + nonself_seq
			dim_seq = ([0] * len(self_seq)) + ([1] * len(nonself_seq))
			S = zip(time_seq,dim_seq)
			S = sorted(S,key=lambda x:x[0])
			Y = [x[0] for x in S]
			dim_seq = [x[1] for x in S]
			cut_point = T
			time_train = [y for y in Y if y <= cut_point]
			dim_train = [e for i,e in enumerate(dim_seq) if Y[i] <= cut_point]
			if len(time_train) < 5:
				continue
			_dict = {}
			_dict['times'] = time_train
			_dict['dims'] = dim_train
			_dict['features'] = feature
			_dict['varying'] = varying
			_dict['publish_year'] = publish_year
			train_seq.append(_dict)

			_dict = {}
			_dict['times'] = [y - cut_point for y in Y if y > cut_point]
			_dict['dims'] = [e for i,e in enumerate(dim_seq) if Y[i] > cut_point]
			_dict['features'] = feature
			_dict['varying'] = varying
			_dict['publish_year'] = publish_year
			test_seq.append(_dict)


		superparams = {}
		superparams['M'] = 2
		superparams['outer'] = 20
		superparams['inner'] = 10
		superparams['K'] = len(train_seq[0]['features'])
		superparams['impact_period'] = 5
		superparams['sigma'] = 1
		superparams['thres'] = 1e-3
		superparams['cut_point'] = cut_point
		superparams['duration'] = 10
		superparams['epsilon'] = 0.3

		initparams = {}
		initparams['theta'] = 0.2
		initparams['w'] = 1.0
		initparams['b'] = 0.5


		return train_seq,test_seq,superparams,initparams



class Single(object):
	def __init__(self):
		self.params = {}

	def train(self,sequences,features,publish_years,pids,threshold,cut=None,predict_year=2000,max_iter=0,max_outer_iter=100):



		if cut is None:
			T = numpy.array([predict_year - publish_year + 1 for publish_year in publish_years],dtype=float)
			train_times = T
		else :
			T = numpy.array([cut+0.001] * len(publish_years))
			train_times = T

		[train_count,num_feature] = [len(features),len(features[0])] 
		rho = 1
		lam = 2
		Z = numpy.mat([1.0]*num_feature)
		U = numpy.mat([0.0]*num_feature)
		
		beta = numpy.mat([1.0]*num_feature)  
		# W1 = [0.05]*train_count 
		sw1 = 0.05
		alpha = [1.0]*train_count
		# W2 = [1.0]*train_count 
		sw2 = 1.0
		old_tlikelihood = 0.0

		init_time = time.time()
		init_clock = time.clock()
		times = 1
		outer_times = 1
		while 1:
			# update alpha,beta
			old_likelihood = 0
			while 1:
				v1 = numpy.mat([0.0]*num_feature) 
				v2 = 0
				for sam in range(train_count): 
					s = sequences[sam]
					s = [x for x in s if x <= train_times[sam]]
					n = len(s)
					fea = numpy.mat(features[sam])
					# sw1 = numpy.mat(W1[sam])
					# sw2 = numpy.mat(W2[sam])
					salpha = alpha[sam]
					old_obj = 0 

					while 1:
						# E-step
						p1 = numpy.multiply(beta,fea) / (beta * fea.T) 
						p2 = 0
						old_sum2 = 0
						for i in range(1,n): 
							mu = beta * fea.T * numpy.exp(- sw1 * s[i])
							sum1 = mu[0,0]
							sum2 = (old_sum2 + salpha) * numpy.exp(- sw2 * (s[i] - s[i-1]))
							old_sum2 = sum2
							summ = sum1 + sum2
							p1 += numpy.multiply(beta, fea) * numpy.exp(- sw1 * s[i]) / float(summ)
							p2 += 1 - mu/float(summ)
						# M-step
						alpha1 = p2
						alpha2 = (n - numpy.sum(numpy.exp(- sw2 * (T[sam] - numpy.array(s)))))/float(sw2)
						if alpha2 == 0.0:
							print 'error: alpha2 == 0.0'
							print 'alpha1 = ',alpha1
							print 'alpha2 = ',alpha2
							print 'sam = ',sam
							print 'sequences[sam]',sequences[sam]
							print 'T[sam]',T[sam]
							print 's',s
							print '(T[sam] - numpy.array(s)) = ',(T[sam] - numpy.array(s))
							exit()
						salpha = alpha1/float(alpha2)
						if salpha > 1e-2: 
							pass
						else:
							salpha=1e-2
						obj = self.calculate_objective(beta*fea.T,sw1,salpha,sw2,s,train_times[sam])
						if abs(old_obj-obj) < threshold * float(train_count):
							break
						old_obj = obj 
						if obj == -numpy.inf:
							print 'obj = ',obj
							print 'salpha = ',salpha
							print 'sw1 = ',sw1
							print 'sw2 = ',sw2
							print 's = ',s
							print 'beta*fea.T = ',beta*fea.T
							exit()
						# break

					v1 += p1 
					v2 += fea * (1 - numpy.exp(- sw1 * T[sam])) / float(sw1) 
					alpha[sam] = salpha 

				# update beta, beta = v1./v2;
				for find in range(num_feature): 
					B = v2[0,find] + rho * (U[0,find] - Z[0,find]) 
					beta[0,find] = (numpy.sqrt(B**2 + 4*rho*v1[0,find]) - B) /float(2*rho)
				
				# z-update without relaxation
				Z = self.shrinkage(beta+U, lam/float(rho)) 
				U = U + beta - Z

				# compute likilyhood
				likelihood = 0;
				for item in range(train_count):
					fea = numpy.mat(features[item])
					# sw1 = W1[item]
					# sw2 = W2[item]
					salpha = alpha[item]
					s = sequences[item]
					s = [x for x in s if x <= train_times[item]]
					obj = self.calculate_objective(beta*fea.T,sw1,salpha,sw2,s,train_times[item])
					likelihood = likelihood - obj

				print {
					'iter':times,
					'outer_iter':outer_times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':likelihood[0,0],
					'w1':sw1,
					'w2':sw2,
					'mean_alpha':numpy.mean(alpha),
				}
				times += 1

				if times > max_iter:# or abs(old_likelihood-likelihood) < threshold:
					break
				# old_likelihood = likelihood

			# update w by gradient descent

			for sam in range(train_count):
				s = sequences[sam]
				s = numpy.mat([x for x in s if x <= train_times[sam]])
				n = s.shape[1]
				fea = numpy.mat(features[sam])
				# sw1 = numpy.mat(W1[sam])
				# sw2 = numpy.mat(W2[sam])
				salpha = alpha[sam]
				# old_obj = 0
				# count = 0
				step_size = 1e-3
				# while 1:
				pw1 = -s[0,0]
				pw2 = numpy.mat(0.0)
				old_sum2 = 0
				for i in range(1,n):
					mu = beta * fea.T * numpy.exp(- sw1 * s[0,i])
					sum1 = mu
					sum2 = (old_sum2 + salpha) * numpy.exp(- sw2 *(s[0,i] - s[0,i-1]))
					old_sum2 = sum2
					summ = sum1 + sum2;
					pw2t = salpha * numpy.sum(numpy.multiply(numpy.exp(- sw2 * \
						(s[0,i] - s[0,0:i])),-(s[0,i]-s[0,0:i])))
					pw1 = pw1 + beta * fea.T * numpy.exp(- sw1 * s[0,i]) * (-s[0,i]) / float(summ)
					pw2 = pw2 + pw2t / float(summ)
				pw1 = pw1 - beta * fea.T * (numpy.exp(-sw1*T[sam]) * T[sam] * sw1 - \
					(1 - numpy.exp(-sw1*T[sam]))) /float(sw1**2)
				upper = numpy.multiply(numpy.exp(-sw2*(T[sam]-s)),((T[sam]-s)*sw2))
				lower = (1-numpy.exp(-sw2*(T[sam]-s)))

				pw2 = pw2 - salpha*numpy.sum(( upper - lower )/float(sw2**2))
				sw1 += step_size*numpy.sign(pw1)[0,0]
				sw2 += step_size*numpy.sign(pw2)[0,0]
				if sw1 > 1e-5 and sw1 < 1e0:
					pass
				else:
					sw1 = 5e-2
				if sw2 > 1e-3 and sw2 < 1e1:
					pass
				else:
					sw2 = 1
				# obj = self.calculate_objective(beta*fea.T,sw1,salpha,sw2,s.tolist()[0],train_times[sam])
				# if abs((old_obj - obj)/float(train_count)) < threshold or count > 2e1 :
				# 	break
				# old_obj = obj
				# count = count +1
			# W1[sam] = sw1
			# W2[sam] = sw2

			# compute likilyhood
			# tlikelihood = 0
			# for item in range(train_count):
			# 	fea = numpy.mat(features[item])
			# 	# sw1 = W1[item]
			# 	# sw2 = W2[item]
			# 	salpha = alpha[item]
			# 	s = sequences[item]
			# 	s = [x for x in s if x <= train_times[item]]
			# 	obj = self.calculate_objective(beta*fea.T,sw1,salpha,sw2,s,train_times[item])
			# 	tlikelihood = tlikelihood - obj

			if outer_times > max_outer_iter:# or abs(old_tlikelihood-tlikelihood)< threshold/10.0:
				break
			outer_times += 1
			# old_tlikelihood = tlikelihood

		patent = {}
		for item in range(len(alpha)):
			a_patent = {}
			a_patent['alpha'] = alpha[item]
			a_patent['fea'] = features[item]
			a_patent['cite'] = sequences[item]
			a_patent['year'] = publish_years[item]
			patent[pids[item]] = a_patent

		params = {}
		params['predict_year'] = predict_year
		params['train_count'] = len(patent)
		params['beta'] = beta.tolist()[0]
		params['w1'] = sw1
		params['w2'] = sw2
		params['patent'] = patent
		params['feature_name'] = ["assignee_type",
								"n_inventor",
								"n_claim",
								"n_backward",
								"ratio_cite",
								"generality",
								"originality",
								"forward_lag",
								"backward_lag"]

		self.params = params
		return params

	def shrinkage(self,vector,kappa):
		mat1 = vector-kappa
		mat2 = -vector-kappa
		for i in range(mat1.shape[0]):
			for j in range(mat1.shape[1]):
				if mat1[i,j] < 0 :
					mat1[i,j] = 0
				if mat2[i,j] < 0 :
					mat2[i,j] = 0
		return mat1 - mat2

	def calculate_objective(self,spontaneous,w1,alpha,w2,events,train_times):
		T=train_times
		N=len(events)
		s=events
		old_sum2 = 0
		obj = numpy.log(spontaneous*numpy.exp(-w1*s[0]))
		for i in range(1,N):
			mu = spontaneous*numpy.exp(-w1*s[i])
			sum1 = mu
			sum2 = (old_sum2 + alpha)*numpy.exp(-w2*(s[i]-s[i-1]))
			old_sum2 = sum2
			obj=obj+numpy.log(sum1+sum2)
		activate = numpy.exp(-w2*(T-numpy.mat(s)))
		activate_sum = numpy.sum((1-activate))*alpha/float(w2)
		obj= obj - activate_sum 
		obj = obj - (spontaneous/w1) * (1 - numpy.exp(-w1*T))
		return obj

	def predict(self,model,sequences,features,publish_years,pids,threshold):
		patents = self.params['patent']
		diffs = []
		ll = 20
		for i, key in enumerate(patents):
			real = self.real_one(key)
			pred = self.predict_one(key,10,self.params['predict_year'])
			diff = []
			ir = 0 # index_real
			for p in pred:
				while int(real[ir][0]) < int(p[0]):
					ir += 1
					if ir >= len(real):
						ir = len(real) - 1
						break
				if ir != len(real) - 1 and int(real[ir][0]) != int(p[0]):
					ir -= 1
				diff.append((p[0],(p[1] - real[ir][1])/float(real[ir][1]) + 0.1))
			diffs.append(diff)
			if ll > len(diff) : ll = len(diff)
		mape = [0.0] * ll
		acc = [0.0] * ll
		for diff in diffs:
			for i in range(ll):
				mape[i] += abs(diff[i][1])
				if abs(diff[i][1]) < 0.3 :
					acc[i] += 1
		for i in range(ll):
			mape[i] /= float(len(diffs))
			acc[i] /= float(len(diffs))
		
		return {'mape':mape,'acc':acc}

	def predict_one(self,_id,duration,pred_year):
		try:
			patent = self.params['patent'][str(_id)]
		except KeyError,e:
			return None
		w1 = self.params['w1']
		alpha = patent['alpha']
		w2 = self.params['w2']
		fea = numpy.mat(patent['fea'])
		ti = patent['cite']
		beta = numpy.mat(self.params['beta'])

		cut_point = pred_year - int(float((patent['year'])))
		tr = numpy.mat([x for x in ti if x <= cut_point])
		pred = self.predict_year_by_year(tr,cut_point,duration,
			beta*numpy.mat(fea).T,w1,alpha,w2)

		_dict = {}
		for i in range(len(pred)):
			year = pred_year + i + 1
			_dict[year] = pred[i]
		_list = sorted(_dict.items(),key=lambda x:x[0])
		return _list

	def real_one(self,_id):
		try:
			patent = self.params['patent'][str(_id)]
		except KeyError,e:
			return None
		cites = patent['cite']
		_dict = {}
		for i in range(len(cites)):
			year = int(float(cites[i])) + int(float(patent['year']))
			_dict[year] = i + 1
		_list = sorted(_dict.items(),key=lambda x:x[0])
		return _list

	def predict_year_by_year(self,tr,cut_point,duration,spontaneous,w1,alpha,w2):
		N = tr.shape[1] 
		pred = []
		for t in range(cut_point+1,cut_point+duration+1):
			delta_ct = spontaneous/w1*(numpy.exp(-w1*(t-1))-numpy.exp(-w1*t)) + \
				alpha/w2*(numpy.sum(numpy.exp(-w2*((t-1)-tr)))-numpy.sum(numpy.exp(-w2*(t-tr))))
			delta_ct = delta_ct[0,0]
			if len(pred) == 0:
				ct = N + delta_ct
			else :
				ct = pred[-1] + delta_ct
			tr = tr.tolist()[0]
			tr.extend([t for i in range(int(delta_ct))])
			tr = numpy.mat(tr)
			pred.append(ct)
		return pred	

	def load(self,f):
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
		train_seq = []
		test_seq = []
		sequences = []
		features = []
		publish_years = []
		for i in range(I):
			publish_year = data[i * 4 + 2][0]
			self_seq = data[i * 4]
			nonself_seq = data[i * 4 + 1]
			feature = data[i * 4 + 3]
			time_seq = self_seq + nonself_seq
			time_seq.sort()

			sequences.append(time_seq)
			features.append(feature)
			publish_years.append(publish_year)

		threshold = 0.01
		return sequences,features,publish_years,pids,threshold

if __name__ == '__main__':
	predictor = MHawkes()
	loaded = predictor.load('../data/paper3.txt')
	# model = predictor.train(*loaded)
	result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
	# print loaded
	# print result['av_mape']
	# print result['av_acc']
	print result

	predictor = Single()
	loaded = predictor.load('../data/paper3.txt')
	result = predictor.predict(predictor.train(*loaded,max_iter=2),*loaded)
	print result

	predictor = MTPP()
	loaded = predictor.load('../data/paper3.txt')
	# model = predictor.train(*loaded)
	result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
	print result
