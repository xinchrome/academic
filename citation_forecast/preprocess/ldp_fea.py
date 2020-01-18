#coding:utf-8
from __future__ import print_function
'''
Created on

This version add paper that written by authors identified by venues
and exclude paper with citation more than 1000
@author: Administrator
'''
import os, datetime
import pickle
import networkx as nx
import numpy as np
from collections import defaultdict
#from draw.density_estimate import density_est as det

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/' + '..')

from engine.util.env_config import env_config

#from fuzzywuzzy import fuzz,process

##############################################
#                strategy
# venue -> paper -> author -> more paper(idset)
#
#
#
##############################################


#common.dataset_dir =/ ("/DATA/data/mag/MAG/")
cspath = env_config.preprocessed_dataset_dir + '/'

def CSvenuePapers():
	# select papers from CS venues
	#output CS papers (set), paper venue dict (dict()), and venue paper dict (defaultdict(set))
	computer_venue = set()
	with open(cspath+"Venues.txt") as f:
		for line in f:
			computer_venue.add(line.strip().split("\t")[0].strip())
		print (len(computer_venue))
	CSvenuePaper = set()
	CSvenuePaper_Venue = dict()
	CSvenue_paper = defaultdict(set)
	with open(env_config.dataset_dir+ "/Papers.txt") as f:
		for line in f:
			nl = line.strip().split("\t")
			itsvenue = nl[8] if nl[8] else nl[9]
			if not itsvenue:
				continue
			if itsvenue in computer_venue:
				CSvenuePaper.add(nl[0])
				CSvenuePaper_Venue[nl[0]]=itsvenue
				CSvenue_paper[itsvenue].add(nl[0])
	print (len(CSvenuePaper)) # 2,961,379
	pickle.dump(CSvenuePaper, open(cspath+"CSvenuePaper","wb")) 
	pickle.dump(CSvenuePaper_Venue, open(cspath+"CSvenuePaper_Venue","wb")) 
	pickle.dump(CSvenue_paper, open(cspath+"CSvenue_paper","wb"))
	print ('find papers finish')

def PaperAuthorAffiliations():# 
	# output: author paper dict (defaultdict(set)), paper author dict (defaultdict(list))
	#         affiliation paper dict (defaultdict(set)), paper affiliation dict (defaultdict(list))
	#         author affiliation dict (defaultdict(set))
	#         author co-author dict (defaultdict(set)) 
	ids = pickle.load(open(cspath+"CSvenuePaper","rb"))
	authorCSpaper = defaultdict(set)
	paperAuthor = defaultdict(list)
	affCSpaper = defaultdict(set)
	paperaAff = defaultdict(list)
	authorAff = defaultdict(set)
	with open(env_config.dataset_dir+ "/PaperAuthorAffiliations.txt") as f:
		for line in f:
			nl = line.strip().split("\t")
			if nl[0] in ids:
				authorCSpaper[nl[1]].add(nl[0])
				paperAuthor[nl[0]].append(nl[1])
				affCSpaper[nl[2]].add(nl[0])
				paperaAff[nl[0]].append(nl[2])
				authorAff[nl[1]].add(nl[2])
	pickle.dump(authorCSpaper, open(cspath+"authorCSpaper","wb"))#CS authors
	pickle.dump(paperAuthor, open(cspath+"paperAuthor","wb"))#CS authors
	pickle.dump(affCSpaper, open(cspath+"affCSpaper","wb"))#CS authors
	pickle.dump(paperaAff, open(cspath+"paperaAff","wb"))#CS authors
	pickle.dump(authorAff, open(cspath+"authorAff","wb"))#CS authors

	#coauthor 
	coauthor = defaultdict(set)
	for au,pap in authorCSpaper.iteritems():
		for p in pap:
			coauthor[au].update(paperAuthor[p])
	pickle.dump(coauthor, open(cspath+"coauthor","wb"))
	print ('paper author aff finish')
	
	
def papercitation():
	#output: paper paper citations dict (defaultdict(set))
	#        paper paper reference dict (defaultdict(set))
	ids = pickle.load(open(cspath+"CSvenuePaper","rb")) # all papers in CS venues
	paper_citation = defaultdict(set)
	paperreference = defaultdict(set)
	with open(env_config.dataset_dir+ "/PaperReferences.txt") as f:
		for line in f:
			citing,cited = line.strip().split("\t")
			if cited in ids:
				paper_citation[cited].add(citing)
			if citing in ids:
				paperreference[citing].add(cited)
	with open(cspath+"Citations","wb") as f:
		pickle.dump(paper_citation, f)
	with open(cspath+"paperreference","wb") as f:
		pickle.dump(paperreference, f)  
	print ("Citation finished")

def PaperKeywords():
	#output: paper keywords dict (defaultdict(set))
	#        keywords papers dict (defaultdict(set))
	CSpaper = pickle.load(open(cspath+"CSvenuePaper","rb")) # all papers in CS venues
	paperFOD = defaultdict(set)
	keyhot = defaultdict(list)

	with open(env_config.dataset_dir+ "/PaperKeywords.txt") as f:
		for line in f:
			nl = line.strip().split("\t")
			if nl[0] in CSpaper:
				paperFOD[nl[0]].add(nl[2])
				keyhot[nl[2]].append(nl[0])
	pickle.dump(paperFOD, open(cspath+"paperFOD","wb"))
	
	fieldhotness = defaultdict(set)
	for key,val in keyhot.iteritems():
		fieldhotness[key]=set(val)
	pickle.dump(fieldhotness, open(cspath+"fieldhot","wb"))
	print ("PaperKeywords finished")

def Papers(paperset,savefile):
	# output: paperinfo (defaultdict(dict))
	CSpaper = paperset
	paperinfo = defaultdict(dict)
	with open(env_config.dataset_dir+ "/Papers.txt") as f:
		for line in f:
			nl = line.strip().split("\t")
			if nl[0] in CSpaper:
				paperinfo[nl[0]]['title']=nl[1]
				try:
					if not nl[4]:
						print ('no time info avaliable')
						papertime = datetime.datetime(2080,1,1)
					elif len(nl[4].split('/'))==1:
						papertime = datetime.datetime.strptime(nl[4],'%Y')
					elif len(nl[4].split('/'))==2:
						papertime = datetime.datetime.strptime(nl[4],'%Y/%m')
					elif len(nl[4].split('/'))==3:
						papertime = datetime.datetime.strptime(nl[4],'%Y/%m/%d')
					else:
						print ('paper time format error',nl[4])
						papertime = datetime.datetime(2080,1,1)
				except Exception as e:
					print (repr(e))
					papertime = datetime.datetime(2080,1,1)
				paperinfo[nl[0]]['date']=papertime
				paperinfo[nl[0]]['venue']=nl[6]
				paperinfo[nl[0]]['journlID']=nl[8]
				paperinfo[nl[0]]['conferID']=nl[9]
				paperinfo[nl[0]]['rank']=nl[10]
	pickle.dump(paperinfo, open(cspath+savefile,"wb"))
	print ('paper info finish')

def authorNet_feature():
	# output: compute the author centrialy for each author
	#         author centrality dict
	authorCo = pickle.load(open(cspath+"coauthor","rb")) #
	nodeSet = set()
	edgeSet = set()
	for key,val in authorCo.iteritems():
		nodeSet.add(key)
		edgeSet.update([(key,item) for item in val if item!=key])
	pickle.dump(nodeSet,open(cspath+"co_nodeSet","wb"))
	pickle.dump(edgeSet,open(cspath+"co_edgeSet","wb"))
	g = nx.Graph()
	g.add_nodes_from(nodeSet)
	g.add_edges_from(edgeSet)
	interested_node = None

	clo_cen = defaultdict(int)
	for node in g.nodes():
		clo_cen[node]=1
		
	# Closeness centrality
	#clo_cen = nx.betweenness_centrality(g, k=int(len(g.nodes())/5))
	#centrality is time-consuming, denote this in real atmosphere
	pickle.dump(clo_cen,open(cspath+"author_cen","wb"))
	print ('authorNet_feature finish')
	
def venueNet_feature():
	# output: compute the author centrialy for each venue
	#         venue centrality dict
	
	CSpaper = pickle.load(open(cspath+"CSvenuePaper","rb")) # all papers in CS venues
	CSvenue_paper = pickle.load(open(cspath+"CSvenue_paper","rb")) #data type, dict, key, value: list
	Citations = pickle.load(open(cspath+"Citations","rb"))
	CSPV = pickle.load( open(cspath+"CSvenuePaper_Venue","rb")) #data type, dict, key, value: list
	nodeSet = set()
	edgeSet = set()
	for key,val in CSvenue_paper.iteritems():
		nodeSet.add(key)
		temp = defaultdict(int)
		for p in val:
			for citing in Citations[p]:
				if citing in CSpaper:
					temp[(CSPV[citing],key)] +=1
		edges = [(key[0],key[1],val) for key,val in temp.iteritems()]
		edgeSet.update(edges)
	g = nx.DiGraph()
	g.add_nodes_from(nodeSet)
	g.add_weighted_edges_from(edgeSet)

	pr = defaultdict(int)
	for node in g.nodes():
		pr[node]=1
		
	#DG.add_weighted_edges_from([(1,2,0.5), (3,1,0.75)])
	#pr = nx.pagerank(g) 
	#page rank is time-consuming, replace this in real atmosphere
	pickle.dump(pr,open(cspath+"venue_cen","wb"))
	print ('venueNet_feature finish')


def paperFeature():
	CSpaper = pickle.load(open(cspath+"CSvenuePaper","rb")) # all papers in CS venues
	authorCSpaper = pickle.load(open(cspath+"authorCSpaper","rb"))#CS authors
	paperAuthor = pickle.load(open(cspath+"paperAuthor","rb"))
	authorAff = pickle.load(open(cspath+"authorAff","rb"))
	paperaAff = pickle.load(open(cspath+"paperaAff","rb"))
	affCSpaper = pickle.load(open(cspath+"affCSpaper","rb"))
	Citations = pickle.load(open(cspath+"Citations","rb"))
	paperFOD = pickle.load( open(cspath+"paperFOD","rb")) 
	coauthor = pickle.load( open(cspath+"coauthor","rb")) 
	author_cen = pickle.load( open(cspath+"author_cen","rb")) 
	venue_cen = pickle.load( open(cspath+"venue_cen","rb")) 
	fieldhot = pickle.load( open(cspath+"fieldhot","rb")) 
	paperreference = pickle.load( open(cspath+"paperreference","rb")) 
	CSvenue_paper = pickle.load(open(cspath+"CSvenue_paper","rb")) #data type
	CSPV = pickle.load( open(cspath+"CSvenuePaper_Venue","rb")) #data type,
	
	samplePaper = pickle.load( open(cspath+"papersample","rb")) 
	
	#main data
	paperfea = defaultdict(dict)
	
	#author
	authorHindex = dict()
	authorrank = dict()
	authordiv = dict()
	authornoca = dict()
	authorprod = dict()
	authororg = dict()
	authorcen = dict()
	for key,value in authorCSpaper.iteritems():
		citations = [len(Citations[item]) for item in value]
		citations = sorted(citations,reverse = True)
		hindex = 0
		for ind,val in enumerate(citations):
			if val<(ind+1):
				hindex = ind+1
				break
		hindex = len(citations) if hindex==0 else hindex
		authorHindex[key] = hindex
		authorrank[key] = np.mean(citations) if citations else 0
		temp = set()
		for item in value:
			temp.update(paperFOD[item])
		authordiv[key] = len(temp)
		authornoca[key] = len(coauthor[key])
		authorprod[key] = len(value)
		authororg[key] = len(authorAff[key])
		authorcen[key] = author_cen[key] # check whether dictional right
	#inistute rank
	affrank = dict()
	for key,val in affCSpaper.iteritems():
		citations = [len(Citations[item]) for item in val]
		affrank[key] = np.mean(citations) if citations else 0
	#venue  
	venuerank = dict()
	venuepub = dict()
	venuediv = dict()
	venueaut = dict()
	venuecen = dict()
	for key,value in CSvenue_paper.iteritems():
		citations = [len(Citations[item]) for item in value]
		venuerank[key] = np.mean(citations) if citations else 0
		temp = set()
		for item in value:
			temp.update(paperFOD[item])
		venuediv[key] = len(temp)
		venuepub[key] = len(value)
		temp = set()
		for item in value:
			temp.update(paperAuthor[item])
		venueaut[key] = len(temp)
		venuecen[key] = venue_cen[key] # check whether dictional right
	
	for pap in samplePaper:
		# author-wise
		#compute h-index,
		its = [authorHindex[item] for item in paperAuthor[pap]]           
		paperfea[pap]['hindex'] = np.mean(its) if its else 0
		#compute rank
		its = [authorrank[item] for item in paperAuthor[pap]]           
		paperfea[pap]['authorrank'] = np.mean(its) if its else 0
		#compute div
		its = [authordiv[item] for item in paperAuthor[pap]]           
		paperfea[pap]['authordiv'] = np.mean(its) if its else 0
		#compute noca
		its = [authornoca[item] for item in paperAuthor[pap]]           
		paperfea[pap]['noca'] = np.mean(its) if its else 0
		#compute product
		its = [authorprod[item] for item in paperAuthor[pap]]           
		paperfea[pap]['product'] = np.mean(its) if its else 0
		#compute inistute num
		its = [authororg[item] for item in paperAuthor[pap]]           
		paperfea[pap]['inistnum'] = np.mean(its) if its else 0
		#compute team size
		paperfea[pap]['teamsize'] = len(paperAuthor[pap])
		#compute author central
		its = [authorcen[item] for item in paperAuthor[pap]]           
		paperfea[pap]['authorcen'] = np.mean(its) if its else 0
		#institue rank
		its = [affrank[item] for item in paperaAff[pap]]
		paperfea[pap]['insitrank'] = np.mean(its) if its else 0
		
		#paper-wise
		#first pre-attachment
		paperfea[pap]['firstpa'] = len(Citations[pap])
		#second pre-attachment
		its = [len(Citations[item]) for item in Citations[pap] if item in CSpaper]
		paperfea[pap]['secondpa'] = np.mean(its) if its else 0
		#field hot
		its = [len(fieldhot[item]) for item in paperFOD[pap]]
		paperfea[pap]['fieldhot'] = np.mean(its) if its else 0
		#first ref
		paperfea[pap]['firstref'] = len(paperreference[pap])
		#sec ref
		its = [len(paperreference[item]) for item in paperreference[pap] if item in CSpaper]
		paperfea[pap]['secref'] = np.mean(its) if its else 0
		#ref div
		temp = set()
		for item in paperreference[pap]:
			if item in CSpaper:
				temp.update(paperFOD[item])
		paperfea[pap]['refdiv'] = len(temp)
		#key div
		paperfea[pap]['keydiv'] = len(paperFOD[item])
		#top div
		temp = set()
		for item in Citations[pap]:
			if item in CSpaper:
				temp.update(paperFOD[item])
		paperfea[pap]['topdiv'] = len(temp)
		
		#venue-wise
		its = CSPV[pap]
		paperfea[pap]['venuediv'] = venuediv[its]
		paperfea[pap]['venuerank'] = venuerank[its]
		paperfea[pap]['venuecen'] = venuecen[its]
		paperfea[pap]['venuepub'] = venuepub[its]
		paperfea[pap]['venueaut'] = venueaut[its]
	
	pickle.dump(paperfea, open(cspath+"paperfea","wb"))
	print ("paperfea finish")
	
	for key,val in paperfea.iteritems():
		print (key,val)
	

def selectpaper(low=1969,high=1989):
	CSpaper = pickle.load(open(cspath+"CSvenuePaper","rb")) # all papers in CS venues
	paperinfo = pickle.load(open(cspath+"paperinfo","rb"))
	paper_citations = pickle.load(open(cspath+"Citations","rb"))
	papersample = []
	for item in CSpaper:
		if low<paperinfo[item]['date'].year<high and len(paper_citations[item])>5:
			papersample.append(item)
	papersample = set(papersample)
	pickle.dump(papersample, open(cspath+"papersample","wb"))
	
	paperset = set()
	for item in papersample:
		paperset.update(paper_citations[item])
	Papers(paperset,'paperinfo_plus')
	paperinfo_plus = pickle.load(open(cspath+"paperinfo_plus","rb"))
	with open(cspath+"sequence.csv","wb") as f:
		for paper in papersample:
			pubyear = paperinfo[paper]['date']
			cited = [(paperinfo_plus[item]['date']-pubyear).days/365.0 for item in paper_citations[paper]]
			cited.sort()
			f.write(paper+",")#用paper做词典的key
			f.write(str(pubyear.year))
			f.writelines([",%s" % item for item in cited])
			f.write("\n")
	print ("select paper finish")

def align():
	paperfea = pickle.load(open(cspath+"paperfea","rb"))
	featurelists = []
	papers = []
	for key,val in paperfea.iteritems():
		#print (key,val)
		papers.append(key)
		featurelists.append([feature_item[1] for feature_item in val.iteritems() ])
	print (len(papers),len(featurelists),len(featurelists[0]))
	print (featurelists[0])
	print (papers[0])

	with open(cspath+"paperfeature.txt","wb") as paper_features:
		for i in range(len(papers)):
			paper = papers[i]
			featurelist = featurelists[i]
			paper_features.write(paper+",")#用paper做词典的key
			paper_features.writelines(["%s," %item for item in featurelist])
			paper_features.write("\n")

	seq = defaultdict(list)
	with open(cspath+"sequence.csv") as f:
		for line in f:
			nl = line.strip().split(",")
			seq[nl[0]] = nl[1:]
	fea = defaultdict(list)
	with open(cspath+"paperfeature.txt") as f:
		for line in f:
			nl = line.strip().split(",")
			fea[nl[0]] = nl[1:]

	# venueauthor = defaultdict(list)
	# with open(cspath+"paper_author_venue.txt") as f:
	#     for line in f:
	#         nl = line.strip().split(",")
	#         #venueauthor[nl[0]] = nl[1:]
	#         venueauthor[nl[0]] = nl
	#return
  
	fseq = open(cspath+"aln_seq.csv","wb")
	ffea = open(cspath+"aln_fea.csv","wb")
	fvau = open(cspath+"aln_vau.csv","wb")
	for key,val in seq.iteritems():
		fseq.writelines(["%s," % item for item in val[0:-1]])
		fseq.write(val[-1]+"\n")
		ffea.writelines(["%s," % item for item in fea[key][0:-1]])
		ffea.write(fea[key][-1]+"\n")
		# fvau.writelines(["%s," % item for item in venueauthor[key][0:-1]])
		# fvau.write(venueauthor[key][-1]+"\n")
		fvau.write("%s\n" % key)

	fseq.close()
	ffea.close()
	fvau.close()

def statistics():
	seq = defaultdict(list)
	with open(cspath+"sequence.csv") as f:
		for line in f:
			nl = line.strip().split(",")
			seq[nl[0]] = nl[1:]
			print (nl)
			#return

if __name__=="__main__":
	# CSvenuePapers()
	# CSpaper = pickle.load(open(cspath+"CSvenuePaper","rb")) # all papers in CS venues
	# PaperAuthorAffiliations()
	# papercitation()
	# PaperKeywords()
	# Papers(CSpaper,'paperinfo')
	# authorNet_feature()
	# venueNet_feature()
	# selectpaper(2000,2016)
	# paperFeature()
	# align()
	# statistics()
	pass
