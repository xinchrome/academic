#coding:utf-8

import time
import tornado.web
import tornado.gen
import tornado.ioloop

settings = {
	'page_title':'Citation Forcasting',
	'page_entry':'index'
}

class SleepHandler(tornado.web.RequestHandler):
	@tornado.web.asynchronous
	@tornado.gen.coroutine
	def get(self):
		yield tornado.gen.Task(tornado.ioloop.IOLoop.instance().add_timeout,time.time()+5)
		self.write("@1 after sleeping or 5s")
		self.finish()
