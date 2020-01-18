#coding:utf-8

import tornado.web

import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/' + '..')

from engine.service import search

class AutocompleteHandler(tornado.web.RequestHandler):
	#@tornado.web.authenticated
	def get(self):
		args = self.request.arguments
		body = self.request.body
		files = self.request.files


		prefix = args.get('prefix',[])
		if isinstance(prefix,list):
			prefix = prefix[0]
		result = search.autocomplete(prefix)
		#results = solr.search(search_text,**{'start':start,'rows':common.paging_size})
		if result:
			suggest = result.get('facet_counts',{}).get('facet_fields',{})
			self.set_header("Content-Type", "application/json")
			self.write( json.dumps({
				'error':0,
				'title':[x for x in suggest.get('normalized_title',[]) if prefix in str(x)],
				'author':[x for x in suggest.get('author',[]) if prefix in str(x)],
				}))
		else :
			self.set_header("Content-Type", "application/json")
			ret_json = {}
			ret_json['error'] = 1
			self.write( json.dumps(ret_json))            

	def post(self):
		self.get()

	def get_current_user(self):
		return self.get_secure_cookie('username')