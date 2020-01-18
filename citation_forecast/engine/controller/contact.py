#coding:utf-8

import tornado.web

settings = {
	'page_title':'联系',
	'page_entry':'contact'
}

class ContactHandler(tornado.web.RequestHandler):
	def get(self):
		self.render('contact.html',**settings)