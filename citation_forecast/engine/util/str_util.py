#coding:utf-8


def capitalize_name(author):
	new_author = ''
	for seg_name in author.split():
		new_author += seg_name.capitalize() + ' '
	return new_author.strip() 