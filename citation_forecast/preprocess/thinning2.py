#coding:utf-8
from __future__ import print_function

import numpy

from matplotlib import pyplot

import preprocess.simhawkes as simhawkes

def histc(a,edges):
	edges.sort()
	a.sort()
	j = 0
	counts = [0]*len(edges)
	for i in range(len(edges)-1):
		while j < len(a):
			if a[j] >= edges[i] and a[j] < edges[i+1]:
				counts[i] += 1
				j += 1
			elif a[j] < edges[i]:
				j += 1
				continue
			elif a[j] >= edges[i]:
				break
	while j < len(a):
		if a[j] == edges[i+1]:
			counts[i+1] += 1
		j += 1
	return counts

# if __name__ == '__main__':
# 	print (range(0,11))
# 	print (range(1,10,4))
# 	print (histc(range(0,11),range(1,10,4)))
# 	exit()


#function [avMAPE,avACC,MAPE,True_,Pre] = thinning2(trainS,trainF,theta,parameter,trainT,predy,epilson,pred_mode)
def thinning2(trainS,trainF,theta,parameter,trainT,predy,epilson,pred_mode):
	[tsize,fsize] = [len(trainF),len(trainF[0])] #size(trainF);
	Pre = []
	True_ = []
	MAPE = []
	err=[]
	parameter = numpy.mat(parameter)
	W1 = parameter[:,0].T #parameter(:,1);
	alpha = parameter[:,1].T #parameter(:,2);
	W2 = parameter[:,2].T #parameter(:,3);
	T = trainT
	theta = numpy.mat(theta)

	for item in range(tsize): #item=1:tsize
		pre = [] #[0.0]*predy #zeros(1,predy);
		true_ = [] #[0.0]*predy #zeros(1,predy);
		mape = [] #[0.0]*predy
		errt = [] #[0.0]*predy
		w1 = W1[0,item] #(item);
		w2 = W2[0,item] #(item);
		ti = trainS[item] #{item};
		tr = numpy.mat([x for x in ti if x <= trainT]) #ti(ti<=trainT);
		N = tr.shape[1] #length(tr);
		for t in range(T+1,T+predy+1): #t=T+1:T+predy
			if pred_mode == 1:
				print ('@1')
				trseq = simhawkes.simhawkes(tr,T,T+predy,theta * numpy.mat(trainF[item]).T,w1,alpha[0,item],w2)
				#simhawkes(tr,T,T+predy,theta*trainF(item,:)',w1,alpha(item),w2);
				ct = numpy.mat(len([x for x in trseq[0].tolist()[0] if x < t])) #length(find(sim<t));
			else:
				ct = N+theta*numpy.mat(trainF[item]).T/w1*(numpy.exp(-w1*T)-numpy.exp(-w1*t)) + \
					alpha[0,item]/w2*(numpy.sum(numpy.exp(-w2*(T-tr)))-numpy.sum(numpy.exp(-w2*(t-tr))))
				#N+theta*trainF(item,:)'/w1*(exp(-w1*T)-exp(-w1*t))+
				#alpha(item)/w2*(sum(exp(-w2*(T-tr)))-sum(exp(-w2*(t-tr))));
			#end
			rd = len([x for x in ti if x <= t]) #length(ti(ti<=t));
			#%[ct-N,rd-N]
			pre.append(ct[0,0])##(t-T) = ct;
			true_.append(rd)#(t-T) = rd;
			mape.append(abs((ct[0,0]-rd)/float(rd)))#(t-T) = abs((ct-rd)/rd);
			errt.append(rd-ct[0,0])#(t-T)=rd-ct;
		#end
		Pre.append(pre) # #= [Pre ; pre];
		True_.append(true_) # = [True_ ; true_];
		err.append(errt) #=[err;errt];
		MAPE.append(mape) # = [MAPE ; mape];
		print ('item = ',item)
	#end
	Pre = numpy.mat(Pre)#
	True_ = numpy.mat(True_)#.append(true_) # = [True_ ; true_];
	err = numpy.mat(err)#.append(errt) #=[err;errt];
	MAPE = numpy.mat(MAPE)#.append(mape) # = [MAPE ; mape];
		
	edges = range(10,1000) #= 10:1000;
	TN =  histc(True_[:,6].T.tolist()[0],edges) #histc(True_(:,7),edges);
	PN = histc(Pre[:,6].T.tolist()[0],edges) #histc(Pre(:,7),edges);
	#%loglog(x,y,'-s') plot line
	#figure;
	pyplot.scatter(edges,TN,s=10,c='b',label='real') #h1=scatter(edges(1:end-1),TN(1:end-1),'r');
	# print (TN)
	# pyplot.show()
	#hold on
	pyplot.scatter(edges,PN,s=10,c='r',label='predicted') #h2=scatter(edges(1:end-1),PN(1:end-1),'b');
	#legend([h1,h2],'real','predicted')
	legend = pyplot.legend(loc='upper right')
	# set(gca, 'XScale', 'log')
	# set(gca, 'YScale', 'log')
	pyplot.xlabel('citations') # xlabel('#citations')
	pyplot.ylabel('papers') # ylabel('#papers')
	pyplot.title('PHPupdatew') # title('PHPupdatew')
	pyplot.semilogx()
	pyplot.show()




#function [CT] = thinning2(trainS,trainF,theta,parameter,trainT,predy
def thinning3(trainS,trainF,theta,parameter,trainT,predy):
	#% you can take ct as the predicted citation times
	[tsize,fsize] = [len(trainF),len(trainF[0])] #size(trainF);
	parameter = numpy.mat(parameter)
	W1 = parameter[:,0].T #parameter(:,1);
	alpha = parameter[:,1].T #parameter(:,2);
	W2 = parameter[:,2].T #parameter(:,3);
	T = trainT
	theta = numpy.mat(theta)

	CT = []
	for item in range(tsize): #item=1:tsize
		w1 = W1[0,item] #(item);
		w2 = W2[0,item] #(item);
		ti = trainS[item] #{item};
		tr = numpy.mat([x for x in ti if x <= trainT]) #ti(ti<=trainT);
		N = tr.shape[1] #length(tr);

		temp = []
		for t in range(T+1,T+predy+1): #t=T+1:T+predy
			ct = N+theta*numpy.mat(trainF[item]).T/w1*(numpy.exp(-w1*T)-numpy.exp(-w1*t)) + \
					alpha[0,item]/w2*(numpy.sum(numpy.exp(-w2*(T-tr)))-numpy.sum(numpy.exp(-w2*(t-tr))))
			#N+theta*trainF(item,:)'/w1*(exp(-w1*T)-exp(-w1*t))+
			#alpha(item)/w2*(sum(exp(-w2*(T-tr)))-sum(exp(-w2*(t-tr))));
			temp.append(ct[0,0]) # = [temp,ct];
		#end
		CT.append(temp) #{item}=temp;

	return CT

#end


