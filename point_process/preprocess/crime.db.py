#coding=utf8

# from preprocess_config import crime_config
from preprocess_config import CrimeConfig
crime_config = CrimeConfig(2013)
import os,re,matplotlib,xlrd,csv
import cPickle as pickle
from itertools import islice
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np
import json
import sys, os

def replace_time_format(s):
	s = s.replace('-','').replace(':','').replace(' ','.')
	return s

def prepare_data(augument_data = True, hierarchy = True, bptt = 5):

	will_dump_processed_dataset = False
	will_generate_noid = True

	if will_dump_processed_dataset == True:
		book = xlrd.open_workbook(crime_config.dataset_file)
		table = book.sheet_by_name(u'Query')
		nrows = table.nrows  
		device_id = offense_id = table.col_values(1)[1:nrows]
		error_day = occur_date = table.col_values(3)[1:nrows]
		error_sec = occur_time = table.col_values(4)[1:nrows]
		error_type1 = loc_type = table.col_values(17)[1:nrows]
		x = table.col_values(21)[1:nrows]
		y = table.col_values(22)[1:nrows]
		y_cali = np.linspace(33.6,33.9,6)
		x_cali = np.linspace(-84.55,-84.25,6)
		error_type2 = []
		for i in range(len(x)) :
			x_loc = y_loc = 0
			for k in range(len(x_cali)-1):
				if x_cali[k] <= float(x[i]) and float(x[i]) < x_cali[k+1] :
					x_loc = k
					break
			for k in range(len(y_cali)-1):
				if y_cali[k] <= float(y[i]) and float(y[i]) < y_cali[k+1] :
					y_loc = k
					break
			loc = (len(x_cali) - 1) * y_loc + x_loc
			error_type2.append(loc)

		pickle.dump([device_id,error_day,error_sec,error_type1,error_type2,book.datemode], open(
			crime_config.preprocessed_dataset_file,'wb'), protocol=0)


	# get error info
	device_id,error_day,error_sec,error_type1,error_type2,book_datemode = pickle.load(open(
		crime_config.preprocessed_dataset_file,'rb'))

	error_day = map(lambda x:datetime.strptime(x,'%m/%d/%Y'),error_day)
	print error_sec[0]
	error_sec = map(lambda x:datetime.strptime(x,'%H:%M:%S') if len(x) > 0 else datetime.strptime('00:00:00','%H:%M:%S'),error_sec)
	error_time = map(lambda x,y:x.replace(hour=y.hour,minute=y.minute,second=y.second),error_day,error_sec)
	print str(error_time[0])
	
	crime_event = map(lambda x, y, z: {'time':x, 'mark':y, 'feature':z}, error_time, error_type2, error_type1)
	crime_event.sort(key=lambda x:x['time'])

	if will_generate_noid == True:
		result = {}
		for i in range(0,len(crime_event),100):
			id_ = str(i)
			if not result.has_key(id_): result[id_] = []
			result[id_].extend(crime_event[i:i+100])

		data = []
		for id_ in result:
			# result[id_].sort()
			begin = result[id_][0]['time']
			line = []
			for time in result[id_]:
				line.append([float('%.4f'%(int((time['time'] - begin).days) * 24 + (time['time'] - begin).seconds / float(60 * 60))), time['mark']]) # weekly
			data.append(line)
	
		with open(crime_config.data, 'w') as fw:
			stdout_old = sys.stdout
			sys.stdout = fw
			for line in data:
				print line
			sys.stdout = stdout_old

	with open(crime_config.data) as fr:
		data = []
		for i,line in enumerate(fr):
			line = eval(line)
			data.append(line)
	
	print data[0]



if __name__=='__main__':
	precessed_data = prepare_data(augument_data = False, hierarchy = False, bptt = 10)
