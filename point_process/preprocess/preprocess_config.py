#coding:utf-8


import os,sys

class PaperConfig(object):
	def __init__(self):
		self.params_file = os.path.dirname(os.path.abspath(__file__)) + '/../service/params.json'
		#params_file = os.path.dirname(os.path.abspath(__file__)) + '/' + 'service/param15.json'

		# original dataset
		self.dataset_dir = '/Volumes/exFAT/tmp/academic'
		self.preprocessed_dataset_dir = os.path.dirname(os.path.abspath(__file__)) + '/../dataset/academic'
		self.aln_seq = self.preprocessed_dataset_dir + '/aln_seq.csv'
		self.aln_fea = self.preprocessed_dataset_dir + '/aln_fea.csv'
		self.aln_vau = self.preprocessed_dataset_dir + '/aln_vau.csv'

class PatentConfig(object):
	def __init__(self):
		self.original_text_dir = "d:/ref/data/patent/NBER/text"
		self.crawed_uspto_text_dir = "d:/ref/data/patent/USPTO/text"
		self.db_type = "sqlite"
		self.db_file = os.path.dirname(os.path.abspath(__file__)) + "/../dataset/patent/patent.db"
		self.mysqldb = {'host':'127.0.0.1','user':'root','passwd':'mysql','db':'test','port':3306,'charset':'utf8'}
		self.mongodb = {'host':'127.0.0.1','port':27017}
		self.feature_csv = os.path.dirname(os.path.abspath(__file__)) + "/data/feature.csv"
		self.citation_csv = os.path.dirname(os.path.abspath(__file__)) + "/data/citation.csv"
		self.pid_csv = os.path.dirname(os.path.abspath(__file__)) + "/data/pid.csv"
		self.params_json = os.path.dirname(os.path.abspath(__file__)) + "/data/params.json"
		self.metrics_json = os.path.dirname(os.path.abspath(__file__)) + "/data/metrics.json"
		self.similarity = os.path.dirname(os.path.abspath(__file__)) + "/../dataset/patent/similarity"
		self.sim_distribution = os.path.dirname(os.path.abspath(__file__)) + "/data/distribution.json"
		self.result_img = os.path.dirname(os.path.abspath(__file__)) + "/result/metrics.png"
		self.multihawkes_json = os.path.dirname(os.path.abspath(__file__)) + '/data/hawkes.json'

		self.similarity_correlation = os.path.dirname(os.path.abspath(__file__)) + '/data/similarity_correlation.json'
		self.similarity_sequence = os.path.dirname(os.path.abspath(__file__)) + '/data/similarity_sequence'

		self.train_count = 5000
		self.converge = 0.05
		self.predict_year = 1991

		# print 'train_count =',train_count, 'converge =',converge

		# train
		self.train_sequence = os.path.dirname(os.path.abspath(__file__)) + '/data_new/train_sequence.txt'

		self.multi_self_params_json = os.path.dirname(os.path.abspath(__file__)) + '/data_new/self_params'
		self.multi_non_self_params_json = os.path.dirname(os.path.abspath(__file__)) + '/data_new/non_self_params'
		self.hawkes_decayed_params_json = os.path.dirname(os.path.abspath(__file__)) + '/data_new/train_hawkes_decayed_params'
		self.citation_distri = os.path.dirname(os.path.abspath(__file__)) + '/data_new/citation_distri.json'
		self.citation_distri_by_count =  os.path.dirname(os.path.abspath(__file__)) + '/data_new/citation_distri_by_count.json'

		self.draw_json = os.path.dirname(os.path.abspath(__file__)) + '/data_new/draw.json'

class ATMErrorConfig(object):
	def __init__(self):
		self.dataset_file = 'd:/data/atm/ATM/data/Errors[20140901-20150401].xlsx' # 212 days in total, therefore observe on a weekly basis
		self.preprocessed_dataset_file = 'd:/data/atm/errors'

class CrimeConfig(object):
	def __init__(self,year):
		self.dataset_file = '/home/xin/data/crime/COBRA-YTD'+str(year)+'.xlsx' # 212 days in total, therefore observe on a weekly basis
		self.preprocessed_dataset_file = '/home/xin/data/crime/crimes'
		self.preprocessed_dataset_file2 = '/home/xin/data/crime/crimes2'
		self.data = os.path.dirname(os.path.abspath(__file__)) + '/../data/crime2.'+str(year)+'.txt'


paper_config = PaperConfig()
patent_config = PatentConfig()
atm_config = ATMErrorConfig()
crime_config = CrimeConfig(2016)
