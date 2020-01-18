#coding:utf-8

import json
import os
import sys

import tornado.web
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/' + '..')

from engine.service import predict, search
from engine.util import str_util


class SubmitHandler(tornado.web.RequestHandler):
	#@tornado.web.authenticated
	def get(self):
		# name=self.current_user
		# if name:
		# 	self.write('hey,'+name)
		# else:
		# 	#self.render('submit.html',**settings)
		# 	self.write('hey,')
		args = self.request.arguments
		body = self.request.body
		files = self.request.files

		search_words = args.get('kw',[])
		# if len(search_words) < 1:
		#     return
		start = args.get('start',['0'])
		if isinstance(start,list):
			start = start[0]
		if start.isdigit():
			start = int(start) - int(start)%5
		else:
			start = 0
		#self.write( {'start':start})
		#return
		results = search.search_paper(search_words,start)
		if results:
			paper_list = []
			# conn = sqlite3.connect(common.dbfile)
			# cur = conn.cursor()
			for r in results:
				paper_id = r['id']
				pred_ct,real_ct = predict.predict_ct(paper_id)

				paper_list.append({
					'id':paper_id,
					'title':r.get('normalized_title',''),
					'url':r.get('url',''),
					'authors':[str_util.capitalize_name(author) for author in r.get('authors',[])],
					'journal':r.get('journal',''),
					'conference_series':r.get('conference_series',''),
					'keyword':[str_util.capitalize_name(author) for author in r.get('keyword','')],
					'publish_year':r.get('publish_year',''),
					'paper_rank':r.get('paper_rank',''),
					'pred_ct':pred_ct,
					'real_ct':real_ct,
					#'authors':authors,
					#'citation':r.get('citation',0),
				})
			# cur.close()
			# conn.close()
			self.set_header("Content-Type", "application/json")
			self.write( json.dumps({'error':0,'start':start,'hits':results.hits,'paper_list':paper_list}))
		else:
			self.set_header("Content-Type", "application/json")
			ret_json = {}
			ret_json['error'] = 1
			self.write( json.dumps(ret_json))

	def post(self):
		self.get()

	def get_current_user(self):
		return self.get_secure_cookie('username')