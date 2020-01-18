#coding:utf-8
from __future__ import print_function

import os
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../academic')

import pysolr
from engine.util.env_config import env_config
import json
import sqlite3
from preprocess.database import tables as database_tables

# requirement: every field is varchar
def insert_dict(cur,tablename,dict_):
	items = dict_.items()
	sql = 'insert into '+ "'" + tablename + "' ("
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
		sql += "?"
	sql += ')'
	print (sql)
	cur.execute("%s"%sql,[str(item[1]).encode('utf8') for item in items])

def select_papers():
	conn = sqlite3.connect(env_config.dbfile)
	cur = conn.cursor()

	with open(env_config.params_file) as fr:
		param = json.load(fr)

	
	for table in database_tables:
		cur.execute("delete from '%s'"%table['name'])
		tablefile = open(table['file'])
		if table['name'] in ['paper','paper_url','paper_author_affiliation','paper_keyword','paper_reference']:
			for line in tablefile:
				line = line.split('\t')
				if param['paper'].has_key(line[0]):
					row_dict = {}
					for field in table['fields']:
						row_dict[field] = line[table['fields'].index(field)].strip()
					insert_dict(cur,table['name'],row_dict)
					conn.commit()
					#break
		elif table['name'] in ['author','affiliation','conference_series','journal','study_field']:
			if table['name'] == 'author':
				fieldname = 'author_id'
				cur.execute("select distinct %s from paper_author_affiliation"%fieldname)
			elif table['name'] == 'affiliation':
				fieldname = 'affiliation_id'
				cur.execute("select distinct %s from paper_author_affiliation"%fieldname)
			elif table['name'] in ['conference_series','journal']:
				fieldname = table['name'] + '_id'
				cur.execute("select distinct %s from paper"%fieldname)
			elif table['name'] == 'study_field':
				fieldname = table['name'] + '_id'
				cur.execute("select distinct %s from paper_keyword"%fieldname)
			#cur.execute("select %s from paper_author_affiliation"%fieldname)
			entity_ids = cur.fetchall()
			if not entity_ids:
				continue
			entity_ids = dict([(entity[0],1) for entity in entity_ids])
			print ('len(entity_ids) = ',len(entity_ids))
			for line in tablefile:
				line = line.split('\t')
				if line[0] in entity_ids:
					row_dict = {}
					for field in table['fields']:
						row_dict[field] = line[table['fields'].index(field)].strip()
					insert_dict(cur,table['name'],row_dict)
					conn.commit()
					#break
		cur.execute("create index if not exists %s_id_index on %s(%s)"%(table['name'],table['name'],table['fields'][0]))
		conn.commit()				

	# print (len(paper_urls))
	# with open(os.path.dirname(__file__) + '/' + "papers.json","w") as fw:
	# 	json.dump(papers,fw)

	cur.close()
	conn.close()


def build_solr_index():

	conn = sqlite3.connect(env_config.dbfile)
	cur = conn.cursor()

	keys = [
		'id','original_title','normalized_title','publish_year','publish_date','DOI','original_venue',
		'normalized_venue','journal_id','conference_series_id','paper_rank','paper_id','url','paper_id',
		'author_id','affiliation_id','original_affiliation','normalized_affiliation','author_sequence_number',
		'paper_id','keyword','study_field_id'
	]

	sql = "select *"
	sql += " from paper left outer join paper_url on paper.id = paper_url.paper_id"
	sql += " left outer join paper_author_affiliation on paper.id = paper_author_affiliation.paper_id"
	sql += " left outer join paper_keyword on paper.id = paper_keyword.paper_id group by id"
	#sql += " limit 5"
	#sql = "select id from paper limit 5"
	cur.execute(sql)
	indexed_papers = []
	#print (len(cur.fetchall()))
	for row in cur.fetchall():
		indexed_paper = {}
		indexed_keys = []		
		for i in range(len(row)):
			if keys[i] in indexed_keys:
				continue
			if keys[i] in ['paper_id','publish_date']:
				continue
			if keys[i] in ['journal_id','conference_series_id','author_id','affiliation_id','study_field_id']:
				#print ("select id, name from %s where id = '%s'"%(keys[i].split('_id')[0],row[i]))
				cur.execute("select id, name from %s where id = '%s'"%(keys[i].split('_id')[0],row[i]))
				res = cur.fetchone()
				if res :
					indexed_paper[keys[i].split('_id')[0]] = res[1]
				else:
					indexed_paper[keys[i].split('_id')[0]] = row[i]
			else:
				indexed_paper[keys[i]] = row[i]
			indexed_keys.append(keys[i]) 
		indexed_papers.append(indexed_paper)

	exkeys = ['authors','citation']

	for paper in indexed_papers:
		for key in exkeys:
			if key in ['authors']:
				authors = []
				sql = "select author_id from paper_author_affiliation where paper_id = '"
				sql += str(paper['id']).encode('utf8') + "'"
				cur.execute(sql)
				for author_id in cur.fetchall():
					sql = "select name from author where id = '" + str(author_id[0]).encode('utf8') + "'"
					#print ('@1 author_id[0] = ', author_id[0])
					cur.execute(sql)
					author = cur.fetchone()                        
					if author:
						author = author[0]
						#new_author = capitalize_name(author)
						authors.append(author)
				paper['authors'] = authors
			elif key in ['citation']:
				citations = 0
				sql = "select count(distinct paper_id) from paper_reference where reference_id = '"
				sql += str(paper['id']).encode('utf8') + "'"
				cur.execute(sql)
				count = cur.fetchone()
				if count:
				    count = count[0]
				    citations = int(count)
				paper['citation'] = citations

	print (indexed_papers)
	print (len(indexed_papers))
	#return

	# with open(os.path.dirname(__file__) + '/' + "papers.json","r") as fr:
	# 	papers = json.load(fr)
	# indexed_papers = papers
	# for paper_id in papers:
	# 	paper = papers[paper_id]
	# 	paper['id'] = paper_id
	# 	indexed_papers.append(paper)

	solr = pysolr.Solr(env_config.search_server + '/solr/paper')
	successful_papers = 0
	failed_papers = 0
	for paper in indexed_papers:
		try:
			solr.add([paper])
			successful_papers += 1
		except:
			failed_papers += 1
	print ('successful_papers = ', successful_papers,', failed_papers = ' ,failed_papers)
	
	solr.optimize()

	cur.close()
	conn.close()

def delete_solr_index():
	solr = pysolr.Solr(env_config.search_server + '/solr/paper')
	solr.delete(q='*:*')


if __name__ == '__main__':
	#select_papers()
	#delete_solr_index()
	build_solr_index()
	#build_entire_table('paper_reference')
	pass




