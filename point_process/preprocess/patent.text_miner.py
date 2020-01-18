#coding:utf-8

import sklearn
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

import os,sys
import json
import cPickle as pickle
import sqlite3
import numpy
import re
import database
import math

from preprocess_config import patent_config


def select_some_htmls():
	_dir = patent_config.crawed_uspto_text_dir + '/'
	with open(patent_config.pid_csv) as f:
		pids = []
		for ln in f:
			pid = str(ln.split(',')[0]).strip()
			pids.append(pid)

	cnt = 0
	not_contained = []
	for pid in pids:
		if os.path.isfile(_dir + str(pid) + '.html') :
			os.rename(_dir + str(pid) + '.html',_dir + '../text2/' + str(pid) + '.html')
			# 	htmls.append(f.read())
			cnt += 1
		else:
			not_contained.append(str(pid))
	print cnt,len(not_contained)




def vectorize(docs,method='tfidf'):
	if method == 'tfidf':
		vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.7)
	elif method == 'count':
		vectorizer = CountVectorizer(max_df = 0.7)
	elif method == 'hash':
		vectorizer = HashingVectorizer(non_negative = True, n_features = 10000)
	else:
		raise Exception('Unkonwn vectorizer ')

	X = vectorizer.fit_transform(docs)
	return X

def svd(X,n=100):
	svd = TruncatedSVD(n_components=n, random_state=42)
	X = svd.fit_transform(X)
	return X


def lda(docs,n=100):
	tokenizer = RegexpTokenizer(r'\w+')
	en_stop = get_stop_words('en')
	p_stemmer = PorterStemmer()
	doc_set = docs

	texts = []
	for i in doc_set:
	    raw = i.lower()
	    tokens = tokenizer.tokenize(raw)
	    stopped_tokens = [i for i in tokens if not i in en_stop]
	    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	    texts.append(stemmed_tokens)
	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(texts)
	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in texts]
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=n, id2word = dictionary)
	# ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=n, id2word = dictionary)
	print corpus[0]
	print ldamodel.get_document_topics(corpus[0])
	X = numpy.zeros((len(corpus),n))
	for i in range(len(corpus)):
		topics = ldamodel.get_document_topics(corpus[i])
		for t in topics:
			X[i,t[0]] = t[1]
	return X

def similarity():
	conn = database.get_connect()
	cur = conn.cursor()
	cur.execute('select * from text')
	pids = []
	abstracts = []
	claims = []
	descriptions = []
	for r in cur:
		pids.append(r[0])
		abstracts.append(r[1])
		claims.append(r[2])
		descriptions.append(r[3])
	print len(abstracts)

	prefix = ['abstracts','claims','descriptions']

	numpy.save(patent_config.similarity + '_pids',numpy.array(pids))
	del pids

	for i, docs in enumerate([abstracts,claims,descriptions]):
		X_tfidf = vectorize(docs,'tfidf')
		# dist = pairwise_distances(X_tfidf)
		# numpy.save(patent_config.similarity + '_' + prefix[i] + '_tfidf',dist,False)
		# del dist
		# X_count = vectorize(docs,'count')
		# dist = pairwise_distances(X_count)
		# numpy.save(patent_config.similarity + '_' + prefix[i] + '_count',dist,False)
		X_svd_tfidf = svd(X_tfidf,100)
		del X_tfidf
		dist = pairwise_distances(X_svd_tfidf)
		del X_svd_tfidf
		numpy.save(patent_config.similarity + '_' + prefix[i] + '_svd_tfidf',dist,False)
		del dist
		# X_svd_count = svd(X_count,100)
		# dist = pairwise_distances(X_svd_count)
		# numpy.save(patent_config.similarity + '_' + prefix[i] + '_svd_count',dist,False)
		X_lda = lda(docs,100)
		dist = pairwise_distances(X_lda)
		del X_lda
		numpy.save(patent_config.similarity + '_' + prefix[i] + '_lda',dist,False)
		del dist

def distance_distribution():
	prefix = ['abstracts','claims','descriptions']
	colors = ['red','blue','green']
	D_tfidf = []
	D_count = []
	D_svd_tfidf = []
	D_svd_count = []
	D_lda = []
	for i in range(len(prefix)):
		# D_tfidf.append(numpy.load(patent_config.similarity + '_' + prefix[i] + '_tfidf.npy'))
		# D_count.append(numpy.load(patent_config.similarity + '_' + prefix[i] + '_count.npy'))
		D_svd_tfidf.append(numpy.load(patent_config.similarity + '_' + prefix[i] + '_svd_tfidf.npy'))
		# D_svd_count.append(numpy.load(patent_config.similarity + '_' + prefix[i] + '_svd_count.npy'))
		D_lda.append(numpy.load(patent_config.similarity + '_' + prefix[i] + '_lda.npy'))

	max_dist = 0.0
	for dist_matrices in [D_svd_tfidf,D_lda]:
		for matrix in dist_matrices:
			_max = numpy.max(matrix)
			if _max > max_dist:
				max_dist = _max
	max_dist = max_dist * 1.01
	curves = []
	units = 50
	for dist_matrices in [D_svd_tfidf,D_lda]:
		for matrix in dist_matrices:
			curve = [0] * units
			for i in range(len(matrix)-1):
				for j in range(i+1,len(matrix)):
					# k = 0
					# while max_dist * ((k+1)/float(units)) < matrix[i,j] and k < units:
					# 	k += 1
					# foor maps a point into an interval
					k = int(math.floor(matrix[i,j] / float(max_dist) * units))
					if k < units:
						curve[k] += 1
					else:
						curve[units-1] += 1
			curves.append(curve)
	print len(curves)
	with open(patent_config.sim_distribution,'w') as f:
		json.dump({'curves':curves,'max_dist':max_dist},f)

def topic_words_and_svd_weights():
	pass

def statistics():
	# pkl = pickle.load(open(patent_config.similarity_pkl))
	# X_lda = pkl['claims'+'lda']
	# print X_lda[100,200]
	# numpy.save(patent_config.similarity + 'lda',numpy.ones((5000,5000)),False)
	# ar = numpy.load(patent_config.similarity + 'lda.npy')
	# print ar[100,100]
	# curves = [1]
	# with open(patent_config.sim_distribution,'w') as f:
	# 	json.dump(curves,f)
	pass
	

if __name__ == '__main__':
	#clear_database()
	# create_database()
	#insert_citation()
	#insert_fake_data()
	# db2csv()
	# insert_text()
	# similarity()
	# statistics()
	# distance_distribution()
	pass


