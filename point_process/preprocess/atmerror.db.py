#coding=utf8

from preprocess_config import atm_config
import os,re,matplotlib,xlrd,csv
import cPickle as pickle
from itertools import islice
from collections import defaultdict,Counter
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np
import json

def replace_time_format(s):
	s = s.replace('-','').replace(':','').replace(' ','.')
	return s

def prepare_data(augument_data = True, hierarchy = True, bptt = 5):
  
	# book = xlrd.open_workbook(atm_config.dataset_file)
	# table = book.sheet_by_name(u'Sheet1')
	# device_id = table.col_values(0)[1:245055]+table.col_values(0)[245068:]
	# error_day = table.col_values(1)[1:245055]+table.col_values(1)[245068:]
	# error_sec = table.col_values(2)[1:245055]+table.col_values(2)[245068:]
	# error_type1 = table.col_values(3)[1:245055]+table.col_values(3)[245068:]
	# error_type2 = table.col_values(4)[1:245055]+table.col_values(4)[245068:]
	# pickle.dump([device_id,error_day,error_sec,error_type1,error_type2,book.datemode], open(atm_config.preprocessed_dataset_file,'wb'), protocol=0)
   
	# # get error info
	# device_id,error_day,error_sec,error_type1,error_type2,book_datemode = pickle.load(open(atm_config.preprocessed_dataset_file,'rb'))
	# print str(error_type1[0])
	# # exit()
	# print {
	# 	'len_device_id':len(device_id),
	# 	'len_error_day':len(error_day),
	# 	'error_day':error_day[0],
	# 	}
	# error_day = map(lambda x:datetime(*xlrd.xldate_as_tuple(x, book_datemode)),error_day)
	# error_sec = map(lambda x:datetime(*xlrd.xldate_as_tuple(x, book_datemode)),error_sec)
	# error_time = map(lambda x,y:x.replace(hour=y.hour,minute=y.minute,second=y.second),error_day,error_sec)
	# print str(error_time[0])
	# error_type1 = map(lambda x:str(x),error_type1)
	# error_type2 = map(lambda x:str(x),error_type2)
	# pickle.dump([device_id,error_time,error_type1,error_type2], open('errors2','w'), protocol=0)
	# result = {}
	# for i,id_ in enumerate(device_id):
	# 	id_ = str(id_)
	# 	if not result.has_key(id_): result[id_] = []
	# 	result[id_].append(error_time[i])
	# data = []
	# for id_ in result:
	# 	result[id_].sort()
	# 	begin = result[id_][0]
	# 	line = str(id_)
	# 	for time in result[id_]:
			# line += ',' + '%.1f'%((int((time - begin).days) + (time - begin).seconds / float(24 * 60 * 60))/7.0) # weekly
	# 	data.append(line + '\n')
		
	# 	line = str(id_)
	# 	line += ',' + replace_time_format(str(begin))
	# 	data.append(line + '\n')

	# 	line = str(id_)
	# 	line += ',0.0,1.0'
	# 	data.append(line + '\n')
	
	# with open('../data/atmerror2.txt', 'w') as fw:
	# 	fw.writelines(data)

	with open('../data/atmerror2.txt') as fr:
		data = []
		for i,line in enumerate(fr):
			if i % 3 == 0:
				data.append(line.strip().split(',')[0] + ',0.0\n')
			data.append(line)

	with open('../data/atmerror2.txt', 'w') as fw:
		fw.writelines(data)
		



if __name__=='__main__':
	precessed_data = prepare_data(augument_data = False, hierarchy = False, bptt = 10)

