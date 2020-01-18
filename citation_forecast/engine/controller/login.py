#coding:utf-8

import tornado.web

settings = {
	'page_title':'登录',
	'page_entry':'login'
}

class LoginHandler(tornado.web.RequestHandler):
	def get(self):
		self.render('login.html',**settings)
 
	def post(self):
		self.set_secure_cookie('username',self.get_argument('username'),expires_days=1)
		self.redirect('/')