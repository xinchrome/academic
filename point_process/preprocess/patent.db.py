#coding:utf-8

import os,sys
import json,re,csv
import sqlite3
import numpy
import time,math
from pymongo import MongoClient

from preprocess_config import patent_config

# database definition

tables = [
	{
		'name':'patent',
		'fields':[
			["pid","varchar(255)"],
			["assignee_type","varchar(255)"],
			["n_inventor","varchar(255)"],
			["n_claim","varchar(255)"],
			["n_backward","varchar(255)"],
			["ratio_cite","varchar(255)"],
			["generality","varchar(255)"],
			["originality","varchar(255)"],
			["forward_lag","varchar(255)"],
			["backward_lag","varchar(255)"],
			["grant_year","varchar(255)"],
			["forward_list","text"],
		],
	},
	{
		'name':'text',
		'fields':[
			["pid","varchar(255)"],
			["abstract","text"],
			["claims","text"],
			["description","text"],
			["a_vector","text"],
			["c_vector","text"],
			["d_vector","text"],
			["svd","text"],
		],
	},
	{
		'name':'vocabulary',
		'fields':[
			["word","varchar(255)"],
			["location","varchar(255)"],
		],
	},
	{
		'name':'life',
		'fields':[
			["pid","varchar(255)"],
			["expire","text"],
			["grant","varchar(255)"],
		]
	},
]

# clear database
def clear_database():
	conn = get_connect()
	cur = conn.cursor()
	for table in tables:
		sql = 'drop table if exists ' + table['name']
		cur.execute(sql)
		conn.commit()
	conn.close()	

#create database
def create_database():
	conn = get_connect()
	cur = conn.cursor()
	for table in tables:
		sql = "create table if not exists "  + table['name'] + " ("
		comma = 0
		for field in table['fields']:
			if comma == 0:
				comma = 1
			else:
				sql += ","
			sql += " ".join(field)
		sql += ")"
		cur.execute(sql)
		conn.commit()
	conn.close()

def get_connect():
	dbfile = patent_config.db_file
	conn = sqlite3.connect(dbfile)
	return conn

def get_mongo():
	mongo = MongoClient(**patent_config.mongodb)
	return mongo

# insert data from dictionary without commit
def insert_row_dict(conn,tablename,row_dict):
	cur = conn.cursor()
	items = row_dict.items()
	sql = 'insert into '+ tablename + ' ('
	comma = 0
	for item in items:
		if comma == 0:
			comma = 1
		else:
			sql += ','
		sql += str(item[0])
	sql += ') values('
	comma = 0
	for item in items:
		if comma == 0:
			comma = 1
		else:
			sql += ','
		#sql += "'" + str(item[1]) + "'"
		if patent_config.db_type == 'mysql':
			sql += '%s'
		else:
			sql += '?'
	sql += ')'
	cur.execute("%s"%sql,[str(item[1]) for item in items])

location = {
	"pid":0,
	"assignee_type":7,
	"n_inventor":-1,
	"n_claim":9,
	"n_backward":12,
	"ratio_cite":14,
	"generality":15,
	"originality":16,
	"forward_lag":17,
	"backward_lag":18,
	"grant_year":1,
	"forward_list":-1,
	"assignee":6,
	'self_lower':21,
	'self_upper':22,
}

def insert_citation():
	cite = patent_config.original_text_dir + '/cite75_99.txt'
	pat = patent_config.original_text_dir + '/apat63_99.txt'
	inventor = patent_config.original_text_dir + '/ainventor.txt'
	pat_iter = open(pat)
	cite_iter = open(cite)
	inventor_iter = open(inventor)
	pat_iter.next()
	cite_iter.next()
	inventor_iter.next()

	patents = {}
	for line in pat_iter:
		fields = [str(x) for x in line.split(',')]
		if int(fields[location['grant_year']]) < 1975 : # first restriction: year >= 1975
			continue
		_id = int(fields[location['pid']])
		info = {}
		for key in location:
			if location[key] > 0:
				info[key] = fields[location[key]]
		info['n_inventor'] = 0
		patents[_id] = info
	print _id,patents[_id]

	cited_by = {}
	for line in cite_iter:
		citing,cited = (int(line.split(',')[0]),int(line.split(',')[1]))
		if patents.has_key(cited) and patents.has_key(citing):
			if cited_by.has_key(cited) :
				cited_by[cited].append(int(patents[citing]['grant_year']))
			else:
				cited_by[cited] = [int(patents[citing]['grant_year'])]

	patents_reduced = {}
	for key in cited_by: # second restriction: at least been cited once
		if patents[key]['assignee_type'] not in ['2','4','6']:# third restriction: U.S. patents only
			continue
		else:
			patents_reduced[key] = patents[key]
			patents_reduced[key]['forward_list'] = cited_by[key]
	print len(patents_reduced)
	for line in inventor_iter:
		_id = int(line.split(',')[0])
		if patents_reduced.has_key(_id):
			patents_reduced[_id]['n_inventor'] += 1

	conn = get_connect()
	for item in patents_reduced.items():
		patent = item[1]
		patent['pid'] = str(item[0])
		patent['n_inventor'] = str(patent['n_inventor'])
		patent['forward_list'] = str(patent['forward_list'])[1:-1]
		insert_row_dict(conn,'patent',patent)
	conn.commit()
	conn.close()

def db2csv():
	conn = get_connect()
	cur = conn.cursor()

	pids = []
	cites = []
	features = []
	cur.execute('select * from patent')
	#description = [x[0] for x in cur.description]
	for i,r in enumerate(cur):
		# if i % 2 != 0:
		# 	continue
		#patent = dict(zip(description,[str(x) for x in r]))
		year = [int(r[-2])]
		year.extend([int(x) for x in r[-1].split(',')])
		if len([t for t in year if t < 1990]) <= 5:
			continue
		pids.append(int(r[0]))
		cites.append(year)
		features.append([str(float(x)) if len(x) > 0 else '0.0' for x in r[1:-2]])

	print len(features),len(cites),len(pids)

	feature = open(patent_config.feature_csv,'w')
	cite = open(patent_config.citation_csv,'w')
	pid = open(patent_config.pid_csv,'w')
	for line in features:
		feature.writelines(["%s," % item for item in line])
		feature.write("\n")
	for line in cites:
		cite.writelines(["%d," % item for item in line])
		cite.write("\n")
	pid.writelines(["%d,\n" % item for item in pids])
	feature.close()
	cite.close()
	pid.close()
	conn.close()

def insert_patents():
	pat = patent_config.original_text_dir + '/apat63_99.txt'
	pat_iter = open(pat)
	pat_iter.next()
	patents = []
	for line in pat_iter:
		fields = [str(x) for x in line.split(',')]
		if int(fields[location['grant_year']]) < 1975 : # first restriction: year >= 1975
			continue
		_id = int(fields[location['pid']])
		info = {}
		for key in location:
			if location[key] > 0:
				info[key] = fields[location[key]]
		info['_id'] = _id
		patents.append(info)
	print _id,info['_id']	

	mongo = get_mongo()
	coll = mongo.patent.patents
	coll.remove({})
	coll.insert_many(patents)	
	mongo.close()

def insert_similarity_citation():
	cite = patent_config.original_text_dir + '/cite75_99.txt'
	cite_iter = open(cite)
	cite_iter.next()

	with open(patent_config.params_json[0:-5] + '1995.json') as f:
		param = json.load(f)
		_patents_sample = param['patent']
		patents_sample = {}
		for key in _patents_sample:
			patents_sample[key] = _patents_sample[key]
			if len(patents_sample) >= 300:
				break
	print len(patents_sample)
	print patents_sample.has_key(4198461)
	print patents_sample.has_key('4198461')

	mongo = get_mongo()
	coll = mongo.patent.patents


	cited_by = {}
	cite = {}
	for line in cite_iter:
		citing,cited = (int(line.split(',')[0]),int(line.split(',')[1]))
		if patents_sample.has_key(str(cited)):
			doc = coll.find_one({'_id':citing})
			if doc is None:
				continue
			if cited_by.has_key(cited) :
				cited_by[cited].append( [int(doc['grant_year']),citing] )
			else:
				cited_by[cited] = [[int(doc['grant_year']),citing]]
		if patents_sample.has_key(str(citing)):
			doc = coll.find_one({'_id':cited})
			if doc is None:
				continue
			if cite.has_key(citing) :
				cite[citing].append( [int(doc['grant_year']),cited] )
			else:
				cite[citing] = [[int(doc['grant_year']),cited]]

	print len(cited_by),len(cite)

	with open(patent_config.multihawkes_json,'w') as f:
		json.dump({'cited_by':cited_by,'cite':cite},f)

	mongo.close()

def parse(html):
	reg = re.compile('<center>.*?abstract.*?</center>[\s\S]*?<hr>',re.I)
	m = reg.search(html)
	if m: 
		abstract = m.group()
	else:
		abstract = ""
	reg = re.compile('<center>.*?claims.*?</center>[\s\S]*?<hr>[\s\S]*?<hr>',re.I)
	m = reg.search(html)
	if m: 
		claims = m.group()
	else:
		claims = ""
	reg = re.compile('<center>.*?description.*?</center>[\s\S]*?<hr>[\s\S]*?<hr>',re.I)
	m = reg.search(html)
	if m: 
		description = m.group()
	else:
		description = ""

	return abstract,claims,description

def insert_text_info():
	_dir = patent_config.crawed_uspto_text_dir + '/'

	with open(patent_config.multihawkes_json) as f:
		citation = json.load(f)
	cited_by = citation['cited_by']
	cite = citation['cite']

	pids = {}
	for c in [cite,cited_by]:
		for _id in c:
			pids[_id] = 1
			for year,__id in c[_id]:
				pids[__id] = 1
	print len(pids)

	not_contained = []
	texts = []
	for pid in pids:
		if os.path.isfile(_dir + str(pid) + '.html') :
			with open(_dir + str(pid) + '.html') as f:
				html = f.read()
				abstract,claims,desc = parse(html)
				if len(claims) > 0 and len(abstract) > 0 and len(desc) > 0 :
					texts.append([str(pid),abstract,claims,desc])
					# break
				else:
					print 'fail to parse - %s.html'%str(pid)
					texts.append([str(pid),abstract,claims,desc])
		else:
			not_contained.append(str(pid))
	print len(texts)
	print len(not_contained)
	# print texts[0][1]
	# return

	conn = get_connect()
	cur = conn.cursor()
	cur.execute('delete from text')
	for text in texts:
		row_dict = {
			'pid':str(text[0]),
			'abstract':text[1],
			'claims':text[2],
			'description':text[3],
			'a_vector':"",
			'c_vector':"",
			'd_vector':"",
			'svd':"",
			}
		insert_row_dict(conn,'text',row_dict)
	conn.commit()
	conn.close()


def generate_similarity_json():
	prefix = ['claims']
	D_svd_tfidf = []
	D_lda = []
	for i in range(len(prefix)):
		D_svd_tfidf.append(numpy.load(patent_config.similarity + '_' + prefix[i] + '_svd_tfidf.npy'))
		D_lda.append(numpy.load(patent_config.similarity + '_' + prefix[i] + '_lda.npy'))
	pids = numpy.load(patent_config.similarity + '_pids.npy')

	pid_idx = dict([(str(pid),i) for i,pid in enumerate(pids)])
	print pid_idx[pids[100]]

	print len(D_lda[0]),len(pid_idx)

	max_dist = 0.0
	for dist_matrices in [D_svd_tfidf,D_lda]:
		for matrix in dist_matrices:
			_max = numpy.max(matrix)
			if _max > max_dist:
				max_dist = _max
	max_dist = max_dist * 1.01
	print max_dis

	with open(patent_config.multihawkes_json) as f:
		citation = json.load(f)
	cited_by = citation['cited_by']
	cite = citation['cite']

	forward_lda = {}
	forward_svd = {}
	backward_lda = {}
	backward_svd = {}

	for _id in cite:
		backward_lda[str(_id)] = []
		backward_svd[str(_id)] = []
		publish = int(patents_sample[str(_id)]['year'])
		for year,__id in cite[_id]:
			model = D_lda[0]
			backward_lda[str(_id)].append((year - publish, model[pid_idx[str(__id)],pid_idx[str(_id)]]))
			model = D_svd_tfidf[0]
			backward_svd[str(_id)].append((year - publish, model[pid_idx[str(__id)],pid_idx[str(_id)]]))
	print backward_svd[_id]

	for _id in cited_by:
		forward_lda[str(_id)] = []
		forward_svd[str(_id)] = []
		publish = int(patents_sample[str(_id)]['year'])
		for year,__id in cited_by[_id]:
			model = D_lda[0]
			forward_lda[str(_id)].append((year - publish, model[pid_idx[str(__id)],pid_idx[str(_id)]]))
			model = D_svd_tfidf[0]
			forward_svd[str(_id)].append((year - publish, model[pid_idx[str(__id)],pid_idx[str(_id)]]))
	print forward_svd[_id]
	print len(forward_lda),len(forward_svd),len(backward_lda),len(backward_svd)

	features = {}
	for _id in cited_by:
		features[str(_id)] = patents_sample[str(_id)]['fea']

	print len(features)

	with open(patent_config.similarity_correlation,'w') as f:
		json.dump({
			'features':features,
			'forward_lda':forward_lda,
			'forward_svd':forward_svd,
			'backward_lda':backward_lda,
			'backward_svd':backward_svd,
			},f)

def generate_similarity_sequence():
	with open(patent_config.params_json[0:-5] + '1995.json') as f:
		param = json.load(f)
		patents = param['patent']

	with open(patent_config.similarity_correlation,'r') as f:
		data = json.load(f)
		forward_lda = data['forward_lda']
		forward_svd = data['forward_svd']
		features = data['features']

	txts = {}
	for model in ['forward_svd','forward_lda']:
		for dist_split in [0.0,0.3,0.5,0.8,1.0,1.3,1.5]:
			txt = []
			for key in forward_lda:
				similar = [key]
				nonsimilar = [key]
				feature = [key] + features[key]
				for e in forward_lda[key]:
					if e[1] < dist_split:
						similar.append(e[0])
					else:
						nonsimilar.append(e[0])
				txt.append(similar)
				txt.append(nonsimilar)
				txt.append(feature)
				txt.append([key,'a','b','c'])
			txts[model + '_' + str(dist_split)] = txt
		print len(txt)
	for key in txts:
		txt = txts[key]
		with open(patent_config.similarity_sequence + '_' + str(key) + '.txt','w') as f:
			for line in txt:
				f.writelines(["%s," % item for item in line[0:-1]])
				f.write("%s" % line[-1])
				f.write("\n")

def get_self_citation_share():
	cite = patent_config.original_text_dir + '/cite75_99.txt'
	pat = patent_config.original_text_dir + '/apat63_99.txt'
	pat_iter = open(pat)
	cite_iter = open(cite)
	pat_iter.next()
	cite_iter.next()

	patents = {}
	self_lower = 0.0
	count_lower = 0
	self_upper = 0.0
	count_upper = 0
	for line in pat_iter:
		fields = [str(x) for x in line.split(',')]
		if int(fields[location['grant_year']]) < 1975 or  int(fields[location['grant_year']]) > 1990:
			continue
		else:
			try:
				if len(fields[location['self_lower']]) > 0 and float(fields[location['self_lower']]) > 0.0:
					self_lower += float(fields[location['self_lower']])	
					count_lower += 1
				if len(fields[location['self_upper']]) > 0 and float(fields[location['self_upper']]) > 0.0:
					self_upper += float(fields[location['self_upper']])	
					count_upper += 1
			except:
				pass
	self_lower /= count_lower
	self_upper /= count_upper
	print 'self_lower =',self_lower,'self_upper =',self_upper

def generate_multi_csv():
	cite = patent_config.original_text_dir + '/cite75_99.txt'
	pat = patent_config.original_text_dir + '/apat63_99.txt'
	pat_iter = open(pat)
	cite_iter = open(cite)
	pat_iter.next()
	cite_iter.next()


	patents = {}
	for line in pat_iter:
		fields = [str(x) for x in line.split(',')]
		if int(fields[location['grant_year']]) < 1970 :
			continue
		else:
			grant_year = int(fields[location['grant_year']])
		if len((fields[location['assignee']])) <= 0:
			continue
		else:
			assignee = int(fields[location['assignee']])
		pid = int(fields[location['pid']])
		
		patents[pid] = {'grant_year':grant_year,'pid':pid,'assignee':assignee}
	print pid,patents[pid]
	print len(patents)


	with open(patent_config.params_json[0:-5] + '1995.json') as f:
		param = json.load(f)
		patents_sample = param['patent']
	print len(patents_sample)
	print patents_sample.has_key(4198461)
	print patents_sample.has_key('4198461')


	self_cited_by = {}
	non_self_cited_by = {}
	for line in cite_iter:
		citing,cited = (int(line.split(',')[0]),int(line.split(',')[1]))
		if patents.has_key(cited) and patents.has_key(citing) and patents_sample.has_key(str(cited)):
			if not self_cited_by.has_key(cited) :
				self_cited_by[cited] = []
			if not non_self_cited_by.has_key(cited) :
				non_self_cited_by[cited] = []
			if patents[citing]['assignee'] == patents[cited]['assignee']:
				self_cited_by[cited].append(int(patents[citing]['grant_year']))
			else:
				non_self_cited_by[cited].append(int(patents[citing]['grant_year']))			
	print len(self_cited_by),len(non_self_cited_by)

	pids = []
	features = {}
	self_count = 0
	non_self_count = 0
	for key in self_cited_by:
		if len(self_cited_by[key]) > 0 and len(non_self_cited_by[key]) > 0:
			pids.append(key)
			features[key] = patents_sample[str(key)]['fea']
			if str(key) == "4232177":
				print "has 4232177"
			if str(key) == "4203444":
				print "has 4203444"
			self_count += len(self_cited_by[key])
			non_self_count += len(non_self_cited_by[key])
	print len(pids)
	print 'self_count =',self_count,' non_self_count =',non_self_count

	txt = []
	for key in pids:
		self = [key] + [x - int(patents[key]['grant_year']) + 1 for x in self_cited_by[key]]
		nonself = [key] + [x - int(patents[key]['grant_year']) + 1 for x in non_self_cited_by[key]]
		feature = [key] + features[key]
		txt.append(self)
		txt.append(nonself)
		txt.append([key,patents[key]['grant_year']])
		txt.append(feature)
		

	with open(patent_config.train_sequence,'w') as f:
		for line in txt:
			f.writelines(["%s," % item for item in line[0:-1]])
			f.write("%s" % line[-1])
			f.write("\n")

def generate_multi_csv():
	cite = patent_config.original_text_dir + '/cite75_99.txt'
	pat = patent_config.original_text_dir + '/apat63_99.txt'
	pat_iter = open(pat)
	cite_iter = open(cite)
	pat_iter.next()
	cite_iter.next()


	patents = {}
	for line in pat_iter:
		fields = [str(x) for x in line.split(',')]
		if int(fields[location['grant_year']]) < 1970 :
			continue
		else:
			grant_year = int(fields[location['grant_year']])
		if len((fields[location['assignee']])) <= 0:
			continue
		else:
			assignee = int(fields[location['assignee']])
		pid = int(fields[location['pid']])
		
		patents[pid] = {'grant_year':grant_year,'pid':pid,'assignee':assignee}
	print pid,patents[pid]
	print len(patents)


	with open(patent_config.params_json[0:-5] + '1995.json') as f:
		param = json.load(f)
		patents_sample = param['patent']
	print len(patents_sample)
	print patents_sample.has_key(4198461)
	print patents_sample.has_key('4198461')


	self_cited_by = {}
	non_self_cited_by = {}
	for line in cite_iter:
		citing,cited = (int(line.split(',')[0]),int(line.split(',')[1]))
		if patents.has_key(cited) and patents.has_key(citing) and patents_sample.has_key(str(cited)):
			if not self_cited_by.has_key(cited) :
				self_cited_by[cited] = []
			if not non_self_cited_by.has_key(cited) :
				non_self_cited_by[cited] = []
			if patents[citing]['assignee'] == patents[cited]['assignee']:
				self_cited_by[cited].append(int(patents[citing]['grant_year']))
			else:
				non_self_cited_by[cited].append(int(patents[citing]['grant_year']))			
	print len(self_cited_by),len(non_self_cited_by)

	pids = []
	features = {}
	self_count = 0
	non_self_count = 0
	for key in self_cited_by:
		if len(self_cited_by[key]) > 0 and len(non_self_cited_by[key]) > 0:
			pids.append(key)
			features[key] = patents_sample[str(key)]['fea']
			if str(key) == "4232177":
				print "has 4232177"
			if str(key) == "4203444":
				print "has 4203444"
			self_count += len(self_cited_by[key])
			non_self_count += len(non_self_cited_by[key])
	print len(pids)
	print 'self_count =',self_count,' non_self_count =',non_self_count

	txt = []
	for key in pids:
		self = [key] + [x - int(patents[key]['grant_year']) + 1 for x in self_cited_by[key]]
		nonself = [key] + [x - int(patents[key]['grant_year']) + 1 for x in non_self_cited_by[key]]
		feature = [key] + features[key]
		txt.append(self)
		txt.append(nonself)
		txt.append([key,patents[key]['grant_year']])
		txt.append(feature)
		

	with open(patent_config.train_sequence,'w') as f:
		for line in txt:
			f.writelines(["%s," % item for item in line[0:-1]])
			f.write("%s" % line[-1])
			f.write("\n")

def read(count):
	citation_csv = file(patent_config.citation_csv,'r')
	citing_events = []
	for row in csv.reader(citation_csv):
		row = [float(x) for x in row[0:-1]]
		citing_events.append(row)

	feature_csv = file(patent_config.feature_csv,'r')
	fea = []
	for row in csv.reader(feature_csv):
		row = [float(x) for x in row[0:-1]]
		_max = max(row)
		_min = min(row)
		new_row = [(x - _min)/float(_max - _min) for x in row]
		fea.append(new_row)

	pid_csv = file(patent_config.pid_csv,'r')
	pids = [row for row in csv.reader(pid_csv)]
	
	publish_years = []
	selected_cites = []
	selected_feas = []
	selected_pids = []
	item = -1
	_dict = {}
	while len(selected_pids) < count:
		item += 1
		if item >= len(citing_events):
			break
		if not _dict.has_key(pids[item][0]):
			_dict[pids[item][0]] = 1
		else:
			raise Exception("@2 duplicate id")
		years = citing_events[item]
		if len(years) < 50: # long 20
			continue
		cite_years = [x - years[0] + 1 for x in years[1:-1]]
		publish_years.append(years[0])
		selected_cites.append(cite_years)
		selected_feas.append(fea[item])
		selected_pids.append(pids[item])

	return selected_pids,selected_cites,selected_feas,publish_years

def generate_long_multi_csv():
	cite = patent_config.original_text_dir + '/cite75_99.txt'
	pat = patent_config.original_text_dir + '/apat63_99.txt'
	pat_iter = open(pat)
	cite_iter = open(cite)
	pat_iter.next()
	cite_iter.next()


	patents = {}
	for line in pat_iter:
		fields = [str(x) for x in line.split(',')]
		if int(fields[location['grant_year']]) < 1970 :
			continue
		else:
			grant_year = int(fields[location['grant_year']])
		if len((fields[location['assignee']])) <= 0:
			continue
		else:
			assignee = int(fields[location['assignee']])
		pid = int(fields[location['pid']])
		
		patents[pid] = {'grant_year':grant_year,'pid':pid,'assignee':assignee}
	print pid,patents[pid]
	print len(patents)


	# with open(patent_config.params_json[0:-5] + '1995.json') as f:
	# 	param = json.load(f)
	# 	patents_sample = param['patent']
	# print len(patents_sample)
	# print patents_sample.has_key(4198461)
	# print patents_sample.has_key('4198461')

	selected_pids,selected_cites,selected_feas,publish_years = read(50000)
	del selected_cites
	del publish_years
	patents_sample = {}
	for i,pid in enumerate(selected_pids):
		patents_sample[int(pid[0])] = {}
		patents_sample[int(pid[0])]['fea'] = selected_feas[i]

	self_cited_by = {}
	non_self_cited_by = {}
	for line in cite_iter:
		citing,cited = (int(line.split(',')[0]),int(line.split(',')[1]))
		if patents.has_key(cited) and patents.has_key(citing) and patents_sample.has_key(int(cited)):
			if not self_cited_by.has_key(cited) :
				self_cited_by[cited] = []
			if not non_self_cited_by.has_key(cited) :
				non_self_cited_by[cited] = []
			if patents[citing]['assignee'] == patents[cited]['assignee']:
				self_cited_by[cited].append(int(patents[citing]['grant_year']))
			else:
				non_self_cited_by[cited].append(int(patents[citing]['grant_year']))			
	print len(self_cited_by),len(non_self_cited_by)

	pids = []
	features = {}
	self_count = 0
	non_self_count = 0
	for key in self_cited_by:
		if len(self_cited_by[key]) > 0 and len(non_self_cited_by[key]) > 0:
			pids.append(key)
			features[key] = patents_sample[int(key)]['fea']
			if str(key) == "4232177":
				print "has 4232177"
			if str(key) == "4203444":
				print "has 4203444"
			self_count += len(self_cited_by[key])
			non_self_count += len(non_self_cited_by[key])
	print len(pids)
	print 'self_count =',self_count,' non_self_count =',non_self_count

	txt = []
	for key in pids:
		self = [key] + [x - int(patents[key]['grant_year']) + 1 for x in self_cited_by[key]]
		nonself = [key] + [x - int(patents[key]['grant_year']) + 1 for x in non_self_cited_by[key]]
		feature = [key] + features[key]
		txt.append(self)
		txt.append(nonself)
		txt.append([key,patents[key]['grant_year']])
		txt.append(feature)
		

	with open(patent_config.train_sequence,'w') as f:
		for line in txt:
			f.writelines(["%s," % item for item in line[0:-1]])
			f.write("%s" % line[-1])
			f.write("\n")

def read_large_long(count):
	citation_csv = file(patent_config.citation_csv,'r')
	citing_events = []
	for row in csv.reader(citation_csv):
		row = [float(x) for x in row[0:-1]]
		citing_events.append(row)

	feature_csv = file(patent_config.feature_csv,'r')
	fea = []
	for row in csv.reader(feature_csv):
		row = [float(x) for x in row[0:-1]]
		_max = max(row)
		_min = min(row)
		new_row = [(x - _min)/float(_max - _min) for x in row]
		fea.append(new_row)

	pid_csv = file(patent_config.pid_csv,'r')
	pids = [row for row in csv.reader(pid_csv)]
	
	publish_years = []
	selected_cites = []
	selected_feas = []
	selected_pids = []
	item = -1
	_dict = {}
	while len(selected_pids) < count:
		item += 1
		if item >= len(citing_events):
			break
		if not _dict.has_key(pids[item][0]):
			_dict[pids[item][0]] = 1
		else:
			raise Exception("@2 duplicate id")
		years = citing_events[item]
		if len(years) < 20: # long 20
			continue
		cite_years = [x - years[0] + 1 for x in years[1:-1]]
		publish_years.append(years[0])
		selected_cites.append(cite_years)
		selected_feas.append(fea[item])
		selected_pids.append(pids[item])

	return selected_pids,selected_cites,selected_feas,publish_years

def generate_large_long_multi_csv():
	cite = patent_config.original_text_dir + '/cite75_99.txt'
	pat = patent_config.original_text_dir + '/apat63_99.txt'
	pat_iter = open(pat)
	cite_iter = open(cite)
	pat_iter.next()
	cite_iter.next()


	patents = {}
	for line in pat_iter:
		fields = [str(x) for x in line.split(',')]
		if int(fields[location['grant_year']]) < 1970 :
			continue
		else:
			grant_year = int(fields[location['grant_year']])
		if len((fields[location['assignee']])) <= 0:
			continue
		else:
			assignee = int(fields[location['assignee']])
		pid = int(fields[location['pid']])
		
		patents[pid] = {'grant_year':grant_year,'pid':pid,'assignee':assignee}
	print pid,patents[pid]
	print len(patents)


	# with open(patent_config.params_json[0:-5] + '1995.json') as f:
	# 	param = json.load(f)
	# 	patents_sample = param['patent']
	# print len(patents_sample)
	# print patents_sample.has_key(4198461)
	# print patents_sample.has_key('4198461')

	selected_pids,selected_cites,selected_feas,publish_years = read_large_long(500000)
	del selected_cites
	del publish_years
	patents_sample = {}
	for i,pid in enumerate(selected_pids):
		patents_sample[int(pid[0])] = {}
		patents_sample[int(pid[0])]['fea'] = selected_feas[i]

	self_cited_by = {}
	non_self_cited_by = {}
	for line in cite_iter:
		citing,cited = (int(line.split(',')[0]),int(line.split(',')[1]))
		if patents.has_key(cited) and patents.has_key(citing) and patents_sample.has_key(int(cited)):
			if not self_cited_by.has_key(cited) :
				self_cited_by[cited] = []
			if not non_self_cited_by.has_key(cited) :
				non_self_cited_by[cited] = []
			if patents[citing]['assignee'] == patents[cited]['assignee']:
				self_cited_by[cited].append(int(patents[citing]['grant_year']))
			else:
				non_self_cited_by[cited].append(int(patents[citing]['grant_year']))			
	print len(self_cited_by),len(non_self_cited_by)

	pids = []
	features = {}
	self_count = 0
	non_self_count = 0
	for key in self_cited_by:
		# if len(self_cited_by[key]) > 0 and len(non_self_cited_by[key]) > 0:
		pids.append(key)
		features[key] = patents_sample[int(key)]['fea']
		if str(key) == "4232177":
			print "has 4232177"
		if str(key) == "4203444":
			print "has 4203444"
		self_count += len(self_cited_by[key])
		non_self_count += len(non_self_cited_by[key])
	print len(pids)
	print 'self_count =',self_count,' non_self_count =',non_self_count

	txt = []
	for key in pids:
		self = [key] + [x - int(patents[key]['grant_year']) + 1 for x in self_cited_by[key]]
		nonself = [key] + [x - int(patents[key]['grant_year']) + 1 for x in non_self_cited_by[key]]
		feature = [key] + features[key]
		txt.append(self)
		txt.append(nonself)
		txt.append([key,patents[key]['grant_year']])
		txt.append(feature)
		

	with open(patent_config.train_sequence,'w') as f:
		for line in txt:
			f.writelines(["%s," % item for item in line[0:-1]])
			f.write("%s" % line[-1])
			f.write("\n")

def generate_citation_distribution():
	cite = patent_config.original_text_dir + '/cite75_99.txt'
	pat = patent_config.original_text_dir + '/apat63_99.txt'
	pat_iter = open(pat)
	cite_iter = open(cite)
	pat_iter.next()
	cite_iter.next()


	patents = {}
	patents_sample = {}
	# patents_citing = {}
	for line in pat_iter:
		fields = [str(x) for x in line.split(',')]
		if int(fields[location['grant_year']]) < 1975:
			continue
		else:
			grant_year = int(fields[location['grant_year']])
		if len((fields[location['assignee']])) <= 0:
			continue
		else:
			assignee = int(fields[location['assignee']])
		pid = int(fields[location['pid']])
		
		patents[pid] = {'grant_year':grant_year,'assignee':assignee}
		if grant_year in [1975,1976,1977]:
			patents_sample[pid] = 1
	print pid,patents[pid]
	print len(patents)


	# with open(patent_config.params_json[0:-5] + '1995.json') as f:
	# 	param = json.load(f)
	# 	patents_sample = param['patent']
	# print len(patents_sample)
	# print patents_sample.has_key(4198461)
	# print patents_sample.has_key('4198461')

	# selected_pids,selected_cites,selected_feas,publish_years = read(50000)
	# del selected_cites
	# del publish_years
	# patents_sample = {}
	# for i,pid in enumerate(selected_pids):
	# 	patents_sample[int(pid[0])] = {}
	# 	patents_sample[int(pid[0])]['fea'] = selected_feas[i]

	self_cited_by = {}
	non_self_cited_by = {}
	for line in cite_iter:
		citing,cited = (int(line.split(',')[0]),int(line.split(',')[1]))
		if patents.has_key(cited) and patents.has_key(citing) and patents_sample.has_key(int(cited)):
			if not self_cited_by.has_key(cited) :
				self_cited_by[cited] = []
			if not non_self_cited_by.has_key(cited) :
				non_self_cited_by[cited] = []
			if patents[citing]['assignee'] == patents[cited]['assignee']:
				self_cited_by[cited].append(int(patents[citing]['grant_year']) - patents[cited]['grant_year'] + 1)
			else:
				non_self_cited_by[cited].append(int(patents[citing]['grant_year']) - patents[cited]['grant_year'] + 1)		
	print len(self_cited_by),len(non_self_cited_by)


	distri = [0.0] * 30
	distri_self = [0.0] * 30
	distri_nonself = [0.0] * 30

	self_count = 0.0
	non_self_count = 0.0

	for key in self_cited_by:
		if len(self_cited_by[key]) > 0 and len(non_self_cited_by[key]) > 1:
			for t in self_cited_by[key]:
				distri[t] += 1
				distri_self[t] += 1
				self_count += 1
			for t in non_self_cited_by[key]:
				distri[t] += 1
				distri_nonself[t] += 1		
				non_self_count += 1

	distri = [x/len(self_cited_by) for x in distri]
	distri_self = [x/len(self_cited_by) for x in distri_self]
	distri_nonself = [x/len(self_cited_by) for x in distri_nonself]

	with open(patent_config.citation_distri,'w') as f:
		json.dump({'total':distri,'self':distri_self,'nonself':distri_nonself},f)

def generate_citation_distribution_by_count():
	cite = patent_config.original_text_dir + '/cite75_99.txt'
	pat = patent_config.original_text_dir + '/apat63_99.txt'
	pat_iter = open(pat)
	cite_iter = open(cite)
	pat_iter.next()
	cite_iter.next()


	patents = {}
	patents_sample = {}
	# patents_citing = {}
	for line in pat_iter:
		fields = [str(x) for x in line.split(',')]
		if int(fields[location['grant_year']]) < 1975:
			continue
		else:
			grant_year = int(fields[location['grant_year']])
		if len((fields[location['assignee']])) <= 0:
			continue
		else:
			assignee = int(fields[location['assignee']])
		pid = int(fields[location['pid']])
		
		patents[pid] = {'grant_year':grant_year,'assignee':assignee}
		if grant_year in [1975,1976,1977]:
			patents_sample[pid] = 1
	print pid,patents[pid]
	print len(patents)


	# with open(patent_config.params_json[0:-5] + '1995.json') as f:
	# 	param = json.load(f)
	# 	patents_sample = param['patent']
	# print len(patents_sample)
	# print patents_sample.has_key(4198461)
	# print patents_sample.has_key('4198461')

	# selected_pids,selected_cites,selected_feas,publish_years = read(50000)
	# del selected_cites
	# del publish_years
	# patents_sample = {}
	# for i,pid in enumerate(selected_pids):
	# 	patents_sample[int(pid[0])] = {}
	# 	patents_sample[int(pid[0])]['fea'] = selected_feas[i]

	self_cited_by = {}
	non_self_cited_by = {}
	for line in cite_iter:
		citing,cited = (int(line.split(',')[0]),int(line.split(',')[1]))
		if patents.has_key(cited) and patents.has_key(citing) and patents_sample.has_key(int(cited)):
			if not self_cited_by.has_key(cited) :
				self_cited_by[cited] = []
			if not non_self_cited_by.has_key(cited) :
				non_self_cited_by[cited] = []
			if patents[citing]['assignee'] == patents[cited]['assignee']:
				self_cited_by[cited].append(int(patents[citing]['grant_year']) - patents[cited]['grant_year'] + 1)
			else:
				non_self_cited_by[cited].append(int(patents[citing]['grant_year']) - patents[cited]['grant_year'] + 1)		
	print len(self_cited_by),len(non_self_cited_by)


	distri = [0.0] * 1000
	distri_self = [0.0] * 1000
	distri_nonself = [0.0] * 1000

	self_count = 0.0
	non_self_count = 0.0

	for key in self_cited_by:
		if len(self_cited_by[key]) > 0 and len(self_cited_by[key]) < 1000 and len(non_self_cited_by[key]) > 1 and len(non_self_cited_by[key]) < 1000:
			distri[len(self_cited_by[key])] += 1
			distri_self[len(self_cited_by[key])] += 1
			self_count += 1
			distri[len(non_self_cited_by[key])] += 1
			distri_nonself[len(non_self_cited_by[key])] += 1		
			non_self_count += 1

	distri = [x/len(self_cited_by) for x in distri]
	distri_self = [x/len(self_cited_by) for x in distri_self]
	distri_nonself = [x/len(self_cited_by) for x in distri_nonself]

	with open(patent_config.citation_distri_by_count,'w') as f:
		json.dump({'total':distri,'self':distri_self,'nonself':distri_nonself},f)


if __name__ == '__main__':
	#clear_database()
	# create_database()
	#insert_citation()
	#insert_fake_data()
	# db2csv()
	# insert_text_vector()

	# insert_patents()
	# insert_similarity_citation()
	# insert_text_info()
	# generate_similarity_json()
	# generate_similarity_sequence()
	# generate_multi_csv()
	# get_self_citation_share()
	# generate_long_multi_csv()
	# generate_citation_distribution_by_count()
	generate_large_long_multi_csv()
	pass



