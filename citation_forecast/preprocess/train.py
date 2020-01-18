#coding:utf-8
from __future__ import print_function

import csv
import json
import os
import sys

import preprocess.train_fast as train_fast
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/' + '..')

from engine.util.env_config import env_config

params_file = 'params-test.json'

# aln_fea : intrinsic quality of a paper
# aln_seq : citation years
# aln_vau : (paper_id, venue, author_id, venue_type, *author_list)
# param.json: 
# {
# 	'trainT':trainT,
# 	'theta':theta,
# 	'paper':{
# 		id1:{
# 			'parameter':[w1,alpha,w2],
# 			'trainF':[1,2,3,4,5,6,7,8,9],
# 			'trainS':[2,3,3,4,5,]
# 		}
# 		id2:{
# 			'parameter':[w1,alpha,w2],
# 			'trainF':[1,2,3,4,5,6,7,8,9],
# 			'trainS':[2,3,3,4,5,]
# 		}
# 	}
# 	'author':{},
# 	'institute':{}
# }


def train(train_count=env_config.train_count):
	seq_csv = open(env_config.aln_seq, 'rb')
	mat = [x for x in csv.reader(seq_csv)]#csvread('aln_seq.csv');
	citing_events = []#{};
	for i in range(len(mat)):#iter=1:size(mat,1)
		temp = mat[i]#mat(iter,:);
		temp = [float(x) for x in temp if float(x) != 0.0]#filter(lambda x:(float(x) != 0.0), temp)#nonzeros(temp);
		citing_events.append(temp)#{iter} = temp;
	#end
	#N = len(mat)#size(mat,1);
	#del mat#clear mat



	print (citing_events[0])
	print (citing_events[1])
	return

	

	fea_csv = open(env_config.aln_fea, 'rb')

	fea = []
	for row in csv.reader(fea_csv):
		row = [float(x) for x in row[0:-1]]
		max_ = max(row)
		min_ = min(row)
		new_row = [(x - min_)/float(max_ - min_) for x in row]
		fea.append(new_row)
	vau_csv = open(env_config.aln_vau, 'rb')
	vau = [row for row in csv.reader(vau_csv)]

	# N = len(fea)#size(fea,1);% 
	# print ('N = ',N)
	N = train_count
	
	#%%select data
	publish_year = []
	trainS = []#{};
	trainF = []#[];
	trainV = []
	item = -1
	#count = 1#1;
	dict_ = {}
	while len(trainV) < N: #for item in range(N):#item=1:N
		item += 1
		if item >= len(citing_events):
			#raise Exception("@1 train_count is too large")
			break

		if not dict_.has_key(vau[item][0]):
			dict_[vau[item][0]] = 1
		else:
			raise Exception("@2 duplicate paper id")

		temp_year = citing_events[item]#{item};
		temp = [x - temp_year[0] + 1 for x in temp_year][1:]#temp-temp(1)+1; 从1开始，对年份编号
		#temp = temp(2:end);
		# xn = [x for x in temp if x <= 6]#temp(temp<=6);
		# if len(xn) < 5:#length(xn)<5 or len(temp) < trainT or 
		# 	continue
		#end

		publish_year.append(temp_year[0])
		trainS.append(temp)#{count} = temp;
		#count=count+1;
		trainF.append(fea[item])# = [trainF ;fea(item,:)];
		trainV.append(vau[item])

	theta,parameter = train_fast.train_fast(trainS, trainF, publish_year, converge=env_config.converge)


	paper = {}
	parameter = parameter.tolist()
	for item in range(len(parameter)):
		id_param = {}
		id_param['parameter'] = parameter[item]
		id_param['trainF'] = trainF[item]
		id_param['trainS'] = trainS[item]
		id_param['year'] = publish_year[item]
		paper[trainV[item][0]] = id_param

	# author = {}
	# institute = {}

	param = {}
	#param['trainT'] = 15
	param['total_papers'] = len(paper)
	param['theta'] = theta.tolist()[0]
	param['paper'] = paper
	# param['author'] = author
	# param['institute'] = institute


	with open(params_file,"w") as fw:
		json.dump(param,fw)


if __name__ == '__main__':
	train()
	pass

