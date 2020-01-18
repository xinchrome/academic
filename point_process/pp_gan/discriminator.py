#coding:utf-8
import os,sys,csv,numpy
numpy.random.seed(1337)

class RNNDiscriminator(object):
	def __init__(self):
		self.params = {}
		self.l1 = 1.
		self.l2 = 1.

	def train(self,sequences,labels,features,publish_years,pids,superparams,cut=None,predict_year=2000,max_iter=0,max_outer_iter=200):
		from keras.layers import Input, Dense, Masking, LSTM, Activation, Dropout, merge
		from keras.models import Model
		from keras.regularizers import l1,l2

		f = Input(shape=(1,len(features[0])),dtype='float')
		# features[paper][feature], sequences[paper][day][feature]
		k = Dense(len(sequences[0][0]),activation='relu',W_regularizer=l1(self.l1))(f)
		# k = merge([k,k],mode='concat',concat_axis=1)
		k1 = Dropout(0.5)(k)

		k2 = Input(shape=(len(sequences[0]),len(sequences[0][0])),dtype='float')

		g1 = merge([k1,k2],mode='concat',concat_axis=1)

		m1 = Masking(mask_value= -1.)(g1)

		n1 = LSTM(128,W_regularizer=l2(self.l2),dropout_W=0.5,dropout_U=0.5)(m1)
		n1 = Dense(2,activation='softmax')(n1)

		model = Model(inputs=[f,k2], outputs=[n1])
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		model.fit(
			[numpy.array([[f] for f in features]),numpy.array(sequences)], [numpy.array(labels)],
			epochs=500, batch_size=1,
			verbose=1,validation_split=0.2)
		self.params['model'] = model
		# embeddings = model.layers[1].W.get_value()
		return model

	def create_trainable_model(self,sequences):
		pass

	def load(self,f,cut=15,normalize_sequence=False):
		# features[paper][feature], sequences[paper][day][feature]
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
		sequences = []
		labels = []
		features = []
		publish_years = []
		for i in range(I):
			publish_year = data[i * 4 + 2][0]
			self_seq = data[i * 4]
			nonself_seq = data[i * 4 + 1]
			feature = data[i * 4 + 3]
			#time_seq = self_seq + nonself_seq
			#time_seq.sort()

			# sequences.append(time_seq)
			sequence = []
			for year in range(int(10 + cut)) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				sequence.append([self_count,nonself_count])
			sequences.append(sequence)
			labels.append([1.,0.])
			features.append(feature)
			publish_years.append(publish_year)

			sequence = []
			for year in range(cut) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				sequence.append([self_count,nonself_count])
			for year in range(cut,int(10 + cut)) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1]) + 0.1 * (year - cut) # * 1.5 #
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1]) + 0.1 * (year - cut) # * 1.5 #
				sequence.append([self_count,nonself_count])
			sequences.append(sequence)
			labels.append([0.,1.])
			features.append(feature)
			publish_years.append(publish_year)

			# sequence = []
			# for year in range(cut) :
			# 	self_count = len([x for x in self_seq if year <= x and x < year + 1])
			# 	nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
			# 	sequence.append([self_count,nonself_count])
			# for year in range(cut,int(10 + cut)) :
			# 	self_count = len([x for x in self_seq if year <= x and x < year + 1]) * 0.5
			# 	nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1]) * 0.5
			# 	sequence.append([self_count,nonself_count])
			# sequences.append(sequence)
			# labels.append([0.,1.])
			# features.append(feature)
			# publish_years.append(publish_year)

		if normalize_sequence == True :
			for i in xrange(len(sequences)) :
				_max = max(sequences[i],key=lambda x: x[0] + x[1])
				_max = float(_max[0] + _max[1])
				for j in xrange(len(sequences[i])):
					sequences[i][j][0] /= _max
					sequences[i][j][1] /= _max

		superparams = {}
		return sequences,labels,features,publish_years,pids,superparams


class CNNDiscriminator(object):
	def __init__(self):
		self.params = {}
		self.l1 = 1.
		self.l2 = 1.

	def train(self,sequences,labels,features,publish_years,pids,superparams,cut=None,predict_year=2000,max_iter=0,max_outer_iter=100):
		from keras.layers import Input, Dense, Flatten, Convolution2D, Activation, Dropout, merge
		from keras.models import Model
		from keras.regularizers import l1,l2

		# f = Input(shape=(1,len(features[0]),1),dtype='float')
		# # features[paper][feature], sequences[paper][day][feature]
		# k = Dense(len(sequences[0][0]),activation='relu',W_regularizer=l1(self.l1))(f)
		# # k = merge([k,k],mode='concat',concat_axis=1)
		# k1 = Dropout(0.5)(k)

		print len(sequences[0])
		k2 = Input(shape=(len(sequences[0]),len(sequences[0][0]),1),dtype='float')

		# g1 = merge([k1,k2],mode='concat',concat_axis=1)
		g1 = k2

		print g1

		# n1 = LSTM(1,activation='sigmoid',W_regularizer=l2(self.l2),dropout_W=0.5,dropout_U=0.5)(g1)
		g1 = Convolution2D(128,kernel_size=[len(sequences[0])-10+1,1],strides=(2,1),activation='relu')(g1)
		g1 = Dropout(0.5)(g1)
		g1 = Convolution2D(128,kernel_size=[3,len(sequences[0][0])],activation='relu')(g1)
		g1 = Dropout(0.5)(g1)
		g1 = Flatten()(g1)
		n1 = Dense(2,activation='softmax')(g1)

		model = Model(inputs=[k2], outputs=[n1])
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		model.fit(
			[numpy.array(sequences)], [numpy.array(labels)],
			epochs=500, batch_size=10,
			verbose=1,validation_split=0.2)
		self.params['model'] = model
		# embeddings = model.layers[1].W.get_value()
		return model

	def create_trainable_model(self,nb_event,nb_type,nb_feature):
		from keras.layers import Input, Dense, Flatten, Convolution2D, Activation, Dropout, merge
		from keras.models import Model
		from keras.regularizers import l1,l2

		x = Input(batch_shape=(1, nb_event, nb_type, nb_feature), dtype='float')
		y = Convolution2D(128, kernel_size=[nb_event-10+1, 1], strides=(2,1), activation='relu')(x)
		y = Dropout(0.5)(y)
		y = Convolution2D(128, kernel_size=[3, nb_type], activation='relu')(y)
		y = Dropout(0.5)(y)
		y = Flatten()(y)
		y = Dense(2,activation='softmax')(y)

		model = Model(inputs=[x], outputs=[y], name='dis_output')
		self.model = model
		return model

	def create_trainable_wasserstein(self,nb_event,nb_type,nb_feature,wgan_clip=1.):
		from keras.layers import Input, Dense, Flatten, Convolution2D, Activation, Dropout, merge
		from keras.models import Model
		from keras.constraints import MinMaxNorm

		constraint = MinMaxNorm(min_value= - wgan_clip,max_value=wgan_clip)

		x = Input(batch_shape=(1, nb_event, nb_type, nb_feature), dtype='float')
		y = Convolution2D(128, kernel_size=[nb_event-10+1, 1], strides=(2,1), activation='relu',
			kernel_constraint=constraint,bias_constraint=constraint)(x)
		y = Dropout(0.5)(y)
		y = Convolution2D(128, kernel_size=[3, nb_type], activation='relu',
			kernel_constraint=constraint,bias_constraint=constraint)(y)
		y = Dropout(0.5)(y)
		y = Flatten()(y)
		y = Dense(2,activation=None,
			kernel_constraint=constraint,bias_constraint=constraint)(y)

		model = Model(inputs=[x], outputs=[y], name='dis_output')
		self.model = model
		return model


	def load(self,f,cut=15):
		# features[paper][feature], sequences[paper][day][feature]
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
		sequences = []
		labels = []
		features = []
		publish_years = []
		for i in range(I):
			publish_year = data[i * 4 + 2][0]
			self_seq = data[i * 4]
			nonself_seq = data[i * 4 + 1]
			feature = data[i * 4 + 3]
			#time_seq = self_seq + nonself_seq
			#time_seq.sort()

			# sequences.append(time_seq)
			sequence = []
			for year in range(int(10 + cut)) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				sequence.append([[self_count],[nonself_count]])
			sequences.append(sequence)
			labels.append([1.,0.])
			features.append(feature)
			publish_years.append(publish_year)

			sequence = []
			for year in range(cut) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				sequence.append([[self_count],[nonself_count]])
			for year in range(cut,int(10 + cut)) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1]) + 0.01 * (year - cut) # * 1.5 #
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1]) + 0.01 * (year - cut) # * 1.5 #
				sequence.append([[self_count],[nonself_count]])
			sequences.append(sequence)
			labels.append([0.,1.])
			features.append(feature)
			publish_years.append(publish_year)

			# sequence = []
			# for year in range(cut) :
			# 	self_count = len([x for x in self_seq if year <= x and x < year + 1])
			# 	nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
			# 	sequence.append([[self_count],[nonself_count]])
			# for year in range(cut,int(10 + cut)) :
			# 	self_count = len([x for x in self_seq if year <= x and x < year + 1]) * 0.5
			# 	nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1]) * 0.5
			# 	sequence.append([[self_count],[nonself_count]])
			# sequences.append(sequence)
			# labels.append([0.,1.])
			# features.append(feature)
			# publish_years.append(publish_year)

		superparams = {}
		return sequences,labels,features,publish_years,pids,superparams



if __name__ == '__main__':
	with open('../log/train_cnn.log','w') as f:
		sys.stdout = f
		# predictor = RNNGenerator()
		# loaded = predictor.load('../data/paper2.txt')
		# predictor.train(*loaded,cut=10,max_outer_iter=0)
		if len(sys.argv) == 2 :
			disc = RNNDiscriminator()
			loaded = disc.load('../data/paper3.txt')
			disc.train(*loaded,cut=10,max_outer_iter=0)
			exit()
		disc = CNNDiscriminator()
		loaded = disc.load('../data/paper3.txt')
		disc.train(*loaded,cut=10,max_outer_iter=0)







