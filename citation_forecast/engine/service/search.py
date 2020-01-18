#coding:utf-8

import os
import sys
import json
import urllib
import pysolr

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/' + '..')
from engine.util.env_config import env_config

autocomplete_url_prefix = env_config.search_server + '/solr/paper'
autocomplete_url_prefix += '/select?rows=0&wt=json&q=*&'
autocomplete_url_prefix += 'facet=true&facet.limit=8&facet.field=normalized_title&'
autocomplete_url_prefix += 'facet.field=author&facet.mincount=1&facet.prefix='

# solr search

field_weights = [
	('normalized_title',10),
	('author',3),
	('keyword',3),
	('publish_year',2),
]

boosting_function = []

highlight_params = [
	('fragsize',10),
]

paging_size = 5

def autocomplete(prefix):
	search_url = urllib.request.urlopen(autocomplete_url_prefix + prefix)
	result = json.loads(search_url.read())
	return result

def search_paper(search_words,start):
	search_text = ''
	for weight in field_weights:
		for w in search_words:
			search_text += weight[0] + ':*' + w + '*^%d '%weight[1]
	for w in search_words:
		search_text += w + ' '

	solr = pysolr.Solr(env_config.search_server + '/solr/paper')
	result = solr.search(search_text, **{'start':start,'rows': paging_size})
# 	search_url = urllib.request.urlopen(env_config.search_server + \
# 		'/solr/paper/select?q=%s&start=%s&rows='%(search_text,start,paging_size))
# 	result = json.loads(search_url.read())
	return result