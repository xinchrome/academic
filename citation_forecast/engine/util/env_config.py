#coding:utf-8

class EnvConfig(object):
	def __init__(self):
		# locatoin of dataset
		self.dataset_dir = '/Volumes/SSD1/data/academic'
		self.preprocessed_dataset_dir = 'data'
		self.aln_seq = self.preprocessed_dataset_dir + '/aln_seq.csv'
		self.aln_fea = self.preprocessed_dataset_dir + '/aln_fea.csv'
		self.aln_vau = self.preprocessed_dataset_dir + '/aln_vau.csv'
		self.params_file = self.preprocessed_dataset_dir + '/params.json'
		self.dbfile = self.preprocessed_dataset_dir + '/academic.db'
		
		# training config
		self.train_count = 500
		self.converge = 0.01

		# location of servers
		self.server_port = 8911
		self.search_server = 'http://127.0.0.1:8983'

env_config = EnvConfig()