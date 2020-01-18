#coding:utf-8

import numpy

def cal_obj(v,w1,alpha,w2,events,trainT): #function [obj]=cal_obj(v,w1,alpha,w2,events,trainT)
	T=trainT
	N=len(events)
	s=events
	old_sum2 = 0
	obj = numpy.log(v*numpy.exp(-w1*s[0])) #log(v*exp(-w1*s(1)));
	for i in range(1,N): #i=2:N
		mu = v*numpy.exp(-w1*s[i]) #v*exp(-w1*s(i));
		sum1 = mu
		sum2 = (old_sum2 + alpha)*numpy.exp(-w2*(s[i]-s[i-1])) #(old_sum2+ alpha)*exp(-w2*(s(i)-s(i-1)));
		old_sum2 = sum2
		obj=obj+numpy.log(sum1+sum2)
	#end
	____1 = numpy.exp(-w2*(T-numpy.mat(s)))
	____2 = numpy.sum((1-    ____1))*alpha/float(w2)
	obj= obj - ____2 #obj - sum((1-exp(-w2*(T-s))))*alpha/w2;
	obj = obj - (v/w1) * (1 - numpy.exp(-w1*T)) #obj - v/w1*(1-exp(-w1*T));
	return obj
#end
