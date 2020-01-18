#coding:utf-8

import tornado.web

settings = {
	'page_title':'Citation Forcasting',
	'page_entry':'index'
}

class IndexHandler(tornado.web.RequestHandler):
	def get(self):
		self.render('index.html',**settings)