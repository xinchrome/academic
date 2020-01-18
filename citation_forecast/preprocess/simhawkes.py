#coding:utf-8
from __future__ import print_function
import numpy

#function [ct] = simhawkes(trseq,T1,T2,v,w1,alpha,w2)
def simhawkes(trseq,T1,T2,v,w1,alpha,w2):
	t = T1
	while t<T2:
		lam = v*numpy.exp(-w1*t) + numpy.sum(alpha*numpy.exp(-w2*(t-trseq)))
		#v*exp(-w1*t) + sum(alpha*exp(-w2*(t-trseq)));
		u = numpy.random.random() # rand();
		t = t+(-numpy.log(numpy.random.random())/float(lam)) #t+(-log(rand)/lam);
		lam2 = v*numpy.exp(-w1*t) + numpy.sum(alpha*numpy.exp(-w2*(t-trseq))) 
		#v*exp(-w1*t) + sum(alpha*exp(-w2*(t-trseq)));
		if t<T2 and u*lam<lam2:
			trseq = numpy.concatenate((trseq,numpy.mat(t)),axis=1) #[trseq;t];
		#end
		if trseq.shape[1] > 1e3: #length(trseq)>1e3
			break
		#end
	#end
	return trseq
	#ct = trseq
	
	#end

if __name__ == '__main__':
	print (simhawkes(numpy.mat(range(10)),10,20,numpy.mat(10),10,10,10))