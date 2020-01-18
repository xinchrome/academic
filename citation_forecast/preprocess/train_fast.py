#coding:utf-8
from __future__ import print_function
import numpy
import sys
import time

from preprocess.cal_obj import cal_obj

# function z = shrinkage(a, kappa)
#     z = max(0, a-kappa) - max(0, -a-kappa);
# end

import signal

threshold = 0

def sighandler(signum,stackframe):
	global threshold
	threshold = 2 * threshold
	print ('current threshold = ',threshold)

signal.signal(signal.SIGINT,sighandler)
signal.signal(signal.SIGTERM,sighandler)


def shrinkage(a,kappa):
	mat1 = a-kappa
	mat2 = -a-kappa
	for i in range(mat1.shape[0]):
		for j in range(mat1.shape[1]):
			if mat1[i,j] < 0 :
				mat1[i,j] = 0
			if mat2[i,j] < 0 :
				mat2[i,j] = 0
	return mat1 - mat2

def train_fast(trainS,trainF,publish_year=None,trainT=1000,converge=1e-1):
	global threshold
	threshold = converge
	parameter = 0

	[tsize,fsize] = [len(trainF),len(trainF[0])] #size(trainF);

	if not publish_year:
		T = numpy.array([trainT]*tsize,dtype=float)
		trainT = T
	else:
		T = numpy.array([2015 - year + 1 for year in publish_year],dtype=float)
		trainT = T
	rho = 1
	lam = 2
	Z = numpy.mat([1.0]*fsize) #ones(1,fsize);
	U = numpy.mat([0.0]*fsize) #zeros(1,fsize);
	
	theta = numpy.mat([1.0]*fsize) #ones(1,fsize); 
	W1 = [0.05]*tsize #ones(1,tsize)*0.05;
	alpha = [1.0]*tsize #ones(1,tsize);
	W2 = [1.0]*tsize #ones(1,tsize);
	old_tlikelihood = 0.0
	times = 0
	while 1:
		#%% update alpha,beta
		old_likelihood = 0
		while 1:
			v1 = numpy.mat([0.0]*fsize) #zeros(1,fsize);
			v2 = 0
			for sam in range(tsize): #sam=1:tsize
				s = trainS[sam] #=trainS{sam};
				s = [x for x in s if x <= trainT[sam]] #=s(s<=trainT);
				n = len(s) #s.shape[1]
				fea = numpy.mat(trainF[sam]) #trainF(sam,:);
				#n = len(s) #=length(s);
				sw1 = numpy.mat(W1[sam])#=W1(sam);
				sw2 = numpy.mat(W2[sam])#=W2(sam);
				salpha = alpha[sam]#=alpha(sam);
				old_obj = 0 #=0;
				#%EM start
				while 1:
					#%E-step
					p1 = numpy.multiply(theta,fea) / (theta * fea.T) #theta.*fea/(theta*fea');
					p2 = 0
					old_sum2 = 0
					#%pre-compute all exp(i,j) for w2 remain constant within inner loop,here use recursively computing
					for i in range(1,n): #i=2:n
						mu = theta * fea.T * numpy.exp(- sw1[0,0] * s[i]) #theta*fea'*exp(-sw1[0,0]*s(i));
						sum1 = mu[0,0]
						#%sum2 = old_sum2*exp(-sw2[0,0]*(s(i)-s(i-1))) + salpha*exp(-sw2[0,0]*(s(i)-s(i-1)));
						sum2 = (old_sum2 + salpha) * numpy.exp(- sw2[0,0] * (s[i] - s[i-1])) 
						#(old_sum2+salpha)*exp(-sw2[0,0]*(s(i)-s(i-1)));
						old_sum2 = sum2
						summ = sum1 + sum2
						p1 =  p1 + numpy.multiply(theta,fea) * numpy.exp(- sw1[0,0] * s[i]) / float(summ)
						#p1 + theta.*fea*exp(-sw1[0,0]*s(i))/summ;
						#%1-dimensional is simple
						p2 = p2 + 1 - mu/float(summ)
					#end
					#%M-step
					alpha1 = p2
					alpha2 = (n - numpy.sum(numpy.exp(- sw2[0,0] * (T[sam] - numpy.array(s)))))/float(sw2[0,0]) 
					#(n - numpy.sum(exp(-sw2[0,0]*(T-s))))/sw2[0,0];
					salpha = alpha1/float(alpha2)
					if salpha > 1e-2: #salpha<1e-2 || isnan(salpha)
						pass
					else:
						salpha=1e-2
					#end
					#%check whether stop
					#obj = cal_obj(theta*fea',sw1[0,0],salpha,sw2[0,0],s,trainT);
					obj = cal_obj(theta*fea.T,sw1[0,0],salpha,sw2[0,0],s,trainT[sam])
					if abs((old_obj-obj)/float(tsize)) < threshold: #1e-2 :
						break
					#end
					old_obj = obj #* 0.1 + old_obj * 0.9

					#print ('theta*fea.T = ',theta*fea.T)

					if obj == -numpy.inf:
						print ('obj = ',obj)
						print ('salpha = ',salpha)
						print ('sw1[0,0] = ',sw1[0,0])
						print ('sw2[0,0] = ',sw2[0,0])
						print ('s = ',s)
						print ('theta*fea.T = ',theta*fea.T)
						exit()


				#print ('@3 E-M step, convergence = ', obj)

					#break
				#end%while
				#%salpha
				v1 = v1 + p1 #v1+p1;
				v2 = v2 + fea * (1 - numpy.exp(- sw1[0,0] * T[sam])) / float(sw1[0,0]) #v2+fea*(1-exp(-sw1*T))/sw1;
				#%sv = v1./v2;
				alpha[sam] = salpha  #(sam)=salpha;
			#end%for

			#%update theta
			#%theta = v1./v2;
			for find in range(fsize): #find = 1:fsize
				B = v2[0,find] + rho * (U[0,find] - Z[0,find]) #v2(find)+rho*(U(find)-Z(find));
				theta[0,find] = (numpy.sqrt(B**2 + 4*rho*v1[0,find]) - B) /float(2*rho)
				#print ('theta[0, %d] = '%find,theta[0,find],(numpy.sqrt(B**2 + 4*rho*v1[0,find]) - B) /float(2*rho))
				#(find) = (sqrt(B^2+4*rho*v1(find))-B)/(2*rho);
			#end
			
			#% z-update without relaxation
			Z = shrinkage(theta+U, lam/float(rho)) #shrinkage(theta+U,lam/rho); %@todo
			U = U + theta - Z

			#%compute likilyhood
			likelihood = 0;
			for item in range(tsize): #item =1:tsize
				fea = numpy.mat(trainF[item]) #(item,:);
				sw1 = W1[item] #(item);
				sw2 = W2[item] #(item);
				salpha = alpha[item]#(item);
				s = trainS[item]#{item};
				s = [x for x in s if x <= trainT[item]] #s(s<=trainT);
				obj = cal_obj(theta*fea.T,sw1,salpha,sw2,s,trainT[item]) #(theta*fea',sw1,salpha,sw2,s,trainT);
				likelihood = likelihood - obj
			#end
			print ('@4 likelihood=', likelihood)
			if abs((old_likelihood-likelihood)/float(tsize)) < threshold: #1e-1:
				break
			#end
			old_likelihood = likelihood# * 0.1 + old_likelihood * 0.9



			#break
		#end%while
		#print ('@1 updated alpha and beta, alpha = ',alpha)

		#%% update w by gradient descent
		#%@todo, check gradients
		for sam in range(tsize): #sam = 1:tsize
			#s=trainS{sam};
			#s=s(s<=trainT);
			# s = trainS[sam]#{item};
			# s = [x for x in s if x <= trainT] #s(s<=trainT);			
			# fea = trainF[sam] #(item,:);fea = trainF(sam,:);
			# n=len(s)
			# sw1=W1(sam);
			# sw2=W2(sam);
			# salpha=alpha(sam);
			# step_size = 1e-3;
			# old_obj = 0;
			# count = 0;

			s = trainS[sam] #=trainS{sam};
			s = numpy.mat([x for x in s if x <= trainT[sam]]) #=s(s<=trainT);
			n = s.shape[1] #s.shape[1]
			fea = numpy.mat(trainF[sam]) #trainF(sam,:);
			#n = len(s) #=length(s);
			sw1 = numpy.mat(W1[sam])#=W1(sam);
			sw2 = numpy.mat(W2[sam])#=W2(sam);
			salpha = alpha[sam]#=alpha(sam);
			old_obj = 0 #=0;
			count = 0
			step_size = 1e-3
			while 1:
				pw1 = -s[0,0] #-s(1);
				pw2 = numpy.mat(0.0)
				old_sum2 = 0
				for i in range(1,n): #i=2:n
					mu = theta * fea.T * numpy.exp(- sw1[0,0] * s[0,i]) #theta*fea'*exp(-sw1[0,0]*s(i));
					sum1 = mu
					sum2 = (old_sum2 + salpha) * numpy.exp(- sw2[0,0] *(s[0,i] - s[0,i-1])) #sw2[0,0]*(s(i)-s(i-1)));
					old_sum2 = sum2
					summ = sum1 + sum2;
					#%@todo accelerate, compute complexity
					pw2t = salpha * numpy.sum(numpy.multiply(numpy.exp(- sw2[0,0] * \
						(s[0,i] - s[0,0:i])),-(s[0,i]-s[0,0:i])))
					#salpha*numpy.sum(exp(-sw2[0,0]*(s(i)-s(1:i-1))).*(-   (  (s(i)-s(1:i-1))    )  ));
					pw1 = pw1 + theta * fea.T * numpy.exp(- sw1[0,0] * s[0,i]) * (-s[0,i]) / float(summ) 
					#=pw1+theta*fea'*exp(-sw1[0,0]*s(i))*(-s(i))/summ;
					pw2 = pw2 + pw2t / float(summ) #=pw2+pw2t/summ;
				#end
				pw1 = pw1 - theta * fea.T * (numpy.exp(-sw1[0,0]*T[sam]) * T[sam] * sw1[0,0] - \
					(1 - numpy.exp(-sw1[0,0]*T[sam]))) /float(sw1[0,0]**2)
				#pw1 - theta*fea'*(exp(-sw1[0,0]*T)*(T*sw1[0,0])-(1-exp(-sw1[0,0]*T)))/(sw1[0,0]^2);
				# print ('sw2[0,0] = ', sw2[0,0])
				# print ('T-s = ', T-s)
				# print ('numpy.exp(-sw2[0,0]*(T-s)) = ',numpy.exp(-sw2[0,0]*(T-s)))
				# print ('((T-s)*sw2[0,0]) = ',((T-s)*sw2[0,0]))
				____1 = numpy.multiply(numpy.exp(-sw2[0,0]*(T[sam]-s)),((T[sam]-s)*sw2[0,0]))
				____2 = (1-numpy.exp(-sw2[0,0]*(T[sam]-s)))

				pw2 = pw2 - salpha*numpy.sum(( ____1 - ____2 )/float(sw2[0,0]**2))
				#pw2 - salpha*numpy.sum((exp(-sw2[0,0]*(T-s)).*((T-s)*sw2[0,0])-(1-exp(-sw2[0,0]*(T-s))))/(sw2[0,0]^2));
				sw1[0,0] = sw1[0,0] + step_size*numpy.sign(pw1)
				sw2[0,0] = sw2[0,0] + step_size*numpy.sign(pw2)
				if sw1[0,0] > 1e-5 and sw1[0,0] < 1e0:  #sw1[0,0]<1e-5 ||sw1[0,0]>1e0 ||isnan(sw1[0,0])
					pass
				else:
					sw1[0,0] = 5e-2
				#end
				if sw2[0,0] > 1e-3 and sw2[0,0] < 1e1: #sw2[0,0]<1e-3 ||sw2[0,0]>1e1 ||isnan(sw2[0,0])
					pass
				else:
					sw2[0,0] = 1
				#end
				obj = cal_obj(theta*fea.T,sw1[0,0],salpha,sw2[0,0],s.tolist()[0],trainT[sam])
				#(theta*fea',sw1,salpha,sw2,s,trainT);
				if abs((old_obj - obj)/float(tsize)) < threshold or count > 2e1 :#abs(old_obj-obj)<1e-1 || count>2e1
					break
				#end
				old_obj = obj #*0.1 + old_obj * 0.9
				count = count +1



				#break
			#end%end while
			W1[sam] = sw1#(sam)=sw1;
			W2[sam] = sw2#(sam)=sw2;
		#end
		#print ('@2 updated w = ',W1,W2)
		
		#%% check whether to stop
		#%compute likilyhood
		tlikelihood = 0
		for item in range(tsize): #item =1:tsize
			fea = numpy.mat(trainF[item]) #trainF(item,:);
			sw1 = W1[item]#=W1(item);
			sw2 = W2[item]#=W2(item);
			salpha = alpha[item]#=alpha(item);
			s = trainS[item] #=trainS{item};
			s = [x for x in s if x <= trainT[item]] #=s(s<=trainT);
			obj = cal_obj(theta*fea.T,sw1,salpha,sw2,s,trainT[item])#(theta*fea',sw1,salpha,sw2,s,trainT);
			tlikelihood = tlikelihood - obj
		#end
		print ('@3 likelihood convergence=',abs((old_tlikelihood-tlikelihood)/float(tsize)),'threshold=',threshold,' times=',times)
		print ('\033[1;32;40m' + time.ctime(),'\033[0m |')
		if abs((old_tlikelihood-tlikelihood)/float(tsize))< threshold/10.0: #1e-1:
			break
		#end
		times += 1
		old_tlikelihood = tlikelihood #* 0.1 + old_tlikelihood * 0.9


		#break
		
	#end%while
	W1 = [x[0,0] for x in W1]
	alpha = [numpy.mat(x)[0,0] for x in alpha]
	W2 = [x[0,0] for x in W2]
	#print (len(alpha),alpha)
	parameter = numpy.concatenate((numpy.mat(W1),numpy.mat(alpha),numpy.mat(W2)),axis=0).T #[W1',alpha',W2'];

	
	return theta, parameter

