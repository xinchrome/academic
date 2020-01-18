#coding:utf-8
from __future__ import print_function

import os
import sys
import json
import numpy as np
# from numpy import mat,exp
# import numpy.sum

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/' + '..')
from engine.util.env_config import env_config

with open(env_config.params_file) as fr:
	params_json = json.load(fr)


def predict_ct(paper_id,predy=10):

	#with open(common.params_file) as fr:
	param = params_json
	#trainS,trainF,theta,parameter = param['trainS'],param['trainF'],param['theta'],param['parameter']
	paper_param = param['paper'][paper_id]
	w1 = paper_param['parameter'][0]
	alpha = paper_param['parameter'][1]
	w2 = paper_param['parameter'][2]
	#CT = thinning2.thinning3(trainS,trainF,theta,parameter,trainT,predy)
	fea = np.mat(paper_param['trainF'])
	ti = paper_param['trainS']
	theta = np.mat(param['theta'])

	trainT = 2015 - int(paper_param['year']) #int(tr[0,-1])
	tr = np.mat([x for x in ti if x <= trainT])
	N = tr.shape[1] #length(tr);
	#print '@1 N, trainT = ',N,',',trainT
	pred = []
	for t in range(trainT+1,trainT+predy+1): #t=T+1:T+predy
		# DO NOT USE SIMULAITON
		# ct = N+theta*numpy.mat(fea).T/w1*(numpy.exp(-w1*trainT)-numpy.exp(-w1*t)) + \
		# 		alpha/w2*(numpy.sum(numpy.exp(-w2*(trainT-tr)))-numpy.sum(numpy.exp(-w2*(t-tr))))
		# #N+theta*trainF(item,:)'/w1*(exp(-w1*T)-exp(-w1*t))+
		# #alpha(item)/w2*(sum(exp(-w2*(T-tr)))-sum(exp(-w2*(t-tr))));
		# pred.append(ct[0,0]) # = [temp,ct];

		# USE SIMULATION
		delta_ct = theta*np.mat(fea).T/w1*(np.exp(-w1*(t-1))-np.exp(-w1*t)) + \
			alpha/w2*(np.sum(np.exp(-w2*((t-1)-tr)))-np.sum(np.exp(-w2*(t-tr))))
		delta_ct = delta_ct[0,0]
		if len(pred) == 0:
			ct = N + delta_ct
		else :
			ct = pred[-1] + delta_ct
		tr = tr.tolist()[0]
		tr.extend([t for i in range(int(delta_ct))])
		tr = np.mat(tr)
		pred.append(ct)
	real = []
	for t in range(1,trainT+1):
		ct = len([x for x in ti if x < t])
		real.append(ct)
	return pred,real

if __name__ == '__main__':
	#test()
	print(predict_ct('084484EB'))
	pass