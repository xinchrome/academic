#coding:utf-8

import os,sys,re
import threading
import Queue
import urllib
import urllib2
import cookielib
import database
import json

from preprocess_config import patent_config

def make_cookie(name='org', value='uspto.gov'):
    return cookielib.Cookie(
        version=0,
        name=name,
        value=value,
        port=None,
        port_specified=False,
        domain="xxxxx",
        domain_specified=True,
        domain_initial_dot=False,
        path="/",
        path_specified=True,
        secure=False,
        expires=None,
        discard=False,
        comment=None,
        comment_url=None,
        rest=None
    )

cookie = make_cookie()
jar = cookielib.CookieJar()
jar.set_cookie(cookie)
processor = urllib2.HTTPCookieProcessor(jar)
opener = urllib2.build_opener(processor)

def download_fulltext(task_queue,instruct_queue):
	while True:
		try:
			instruct = instruct_queue.get(block=False)
		except Queue.Empty:
			pass
		else:
			return

		try:
			pid = task_queue.get(block=False)
		except:
			print 'exit download_fulltext'
			return
		else:
			url = 'http://patft.uspto.gov/netacgi/nph-Parser?'
			url += 'Sect1=PTO1&Sect2=HITOFF&d=PALL&p=1&u=%2Fnetahtml%2FPTO%2Fsrchnum.htm&r=1&f=G&l=50&'
			url += 's1=%d.PN.&OS=PN/%d&RS=PN/%d'%(int(pid),int(pid),int(pid))
			html = ''
			try:
				html = urllib.urlopen(url).read()
			except Exception, e:
				print e
			else:
				with open(patent_config.crawed_uspto_text_dir + '/' + str(pid) + '.html','w') as f:
					f.write(html)

def download_lifecycle(task_queue,instruct_queue):
	while True:
		try:
			instruct = instruct_queue.get(block=False)
		except Queue.Empty:
			pass
		else:
			return

		try:
			pid,app_id = task_queue.get(block=False)
		except:
			print 'exit download_lifecycle'
			return
		else:
			url = 'https://fees.uspto.gov/MaintenanceFees/fees/details?'
			url += 'applicationNumber=%d&patentNumber=%d'%(int(app_id),int(pid))
			html = ''
			try:
				html = opener.open(url).read()
			except Exception, e:
				print e
			else:
				with open(patent_config.crawed_uspto_lifecycle_dir + '/' + str(pid) + '.html','w') as f:
					f.write(html)

task_queue = Queue.Queue()
threads = [] # [ (t,q),(t,q),...]

def add_threads(delta,func=download_fulltext):
	added = 0
	for i in range(len(threads)):
		if added >= delta:
			return
		if not threads[i][0].isAlive():
			new_t = threading.Thread(target=func,args=(task_queue,threads[i][1]))
			new_t.start()
			threads[i][0] = new_t
			added += 1
	while added < delta:
		new_q = Queue.Queue()
		new_t = threading.Thread(target=func,args=(task_queue,new_q))
		new_t.start()
		threads.append([new_t,new_q])
		added += 1

def current_download_threads():
	current = 0
	for i in range(len(threads)):
		if threads[i][0].isAlive():
			current += 1
	return current

def crawl():
	files = os.listdir(patent_config.crawed_uspto_text_dir)
	local_pids = [int(x[0:-5]) for x in files if len(x) > 5 and x[-5:] == '.html' and x[0:-5].isdigit()]
	local_pids = dict(zip(local_pids,[0]*len(local_pids)))

	conn = database.get_connect()
	cur = conn.cursor()
	cur.execute('select pid from patent')
	for r in cur.fetchall():
		pid = int(r[0])
		if not local_pids.has_key(pid):
			task_queue.put(pid,block=False)
	conn.close()	

	add_threads(20)
	for thread in threads:
		thread[0].join()

def crawl_citation():
	files = os.listdir(patent_config.crawed_uspto_text_dir)
	local_pids = [int(x[0:-5]) for x in files if len(x) > 5 and x[-5:] == '.html' and x[0:-5].isdigit()]
	local_pids = dict(zip(local_pids,[0]*len(local_pids)))

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
	count = 0
	for pid in pids:
		if not local_pids.has_key(int(pid)):
			task_queue.put(pid,block=False)
			count += 1
	print count
	
	add_threads(20)
	for thread in threads:
		thread[0].join()


def crawl_lifecycle():
	files = os.listdir(patent_config.crawed_uspto_text_dir)
	local_pids = [int(x[0:-5]) for x in files if len(x) > 5 and x[-5:] == '.html' and x[0:-5].isdigit()]
	local_pids = dict(zip(local_pids,[0]*len(local_pids)))

	conn = database.get_connect()
	cur = conn.cursor()
	cur.execute('select pid from patent')
	for r in cur.fetchall():
		pid = int(r[0])
		if not local_pids.has_key(pid):
			task_queue.put(pid,block=False)
	conn.close()	

	add_threads(20)
	for thread in threads:
		thread[0].join()

	all_pids = [int(x[0:-5]) for x in files if len(x) > 5 and x[-5:] == '.html' and x[0:-5].isdigit()]
	files = os.listdir(patent_config.crawed_uspto_lifecycle_dir)
	local_pids = [int(x[0:-5]) for x in files if len(x) > 5 and x[-5:] == '.html' and x[0:-5].isdigit()]
	local_pids = dict(zip(local_pids,[0]*len(local_pids)))

	print len(all_pids),len(local_pids)
	while True:
		try:
			task_queue.get(block=False)
		except Queue.Empty:
			break

	reg = re.compile('\d+\/\d+,\d+')#*Appl\.\s*No\..*\d+').*?(\d+\/\d+,\d+)')
	spliter = re.compile('[/,]')
	for pid in all_pids:
		with open(patent_config.crawed_uspto_text_dir + '/' + str(pid) + '.html') as f:
			m = reg.search(f.read())
			try:
				app_id = int(''.join(spliter.split(m.group())))
			except Exception, e:
				print e
				print 'fail - on pid %d'%pid
				continue
		if not local_pids.has_key(pid):
			task_queue.put([pid,app_id],block=False)
		break

	add_threads(20,download_lifecycle)
	print current_download_threads()
	for thread in threads:
		thread[0].join()

if __name__ == '__main__':
	# crawl()
	crawl_citation()