#filename: wbw_utils.py
#Author: Vivek
#Date: March 2017
#Description: Takes care of tokenizing file during word-by-word french->french
#translation




import sys
import nltk
import data_custom_utils as du
import os

# eng_vocab_file = open('eng_vocab','a+')
# dic = {}
#write out a eng vocab file
# lines = open('eng_train_data').readlines()
# for line in lines:
# 	if(sys.version_info < (3,0)):
# 		line = line.decode('utf-8').strip()
# 	words = nltk.word_tokenize(line)
# 	for w in words:
# 		if w not in dic.keys():
# 			dic[w] = len(dic.keys())
# 			eng_vocab_file.write(w+'\n')
# 				# rev_dic[len(rev_dic.keys())] = w
# 	# return dic, rev_dic
# 	# 


#create a eng_french token dictionary _word by word
eng_fr_dic = {}
fr_eng_dic = {}

##change this
home = '/Users/vivek/Google Drive/Colby17S/translate_mini'
lines_eng = open(home+'/data/french_french_data/eng_vocab').readlines()
lines_fr = open(home+'/data/french_french_data/eng_vocab').readlines()

for i in range(len(lines_eng)):
	l = lines_eng[i].strip()
	eng_fr_dic[l] = lines_fr[i].strip()
	fr_eng_dic[lines_fr[i].strip()] = l


# french_source_file = open('fr_wordbword_train_data','a+')
# eng_lines = open('eng_train_data').readlines()
# for sentence in eng_lines:
# 	new_s = []
# 	words = nltk.word_tokenize(sentence)
# 	for w in words:
# 		fr_word = eng_fr_dic[w]

# 		# fr_word = eng_fr_dic[w]
# 		# if fr_word in french_to_token_dic.keys():
# 		# 	token = french_to_token_dic[fr_word]
# 		# elif fr_word.lower() in french_to_token_dic.keys():
# 		# 	token = french_to_token_dic[fr_word.lower()]
# 		# elif fr_word.title() in french_to_token_dic.keys():
# 		# 	token = french_to_token_dic[fr_word.title()]
# 		# else:
# 		# 	print(fr_word)
# 		# 	print('something went wrong')
# 		new_s.append(fr_word)
# 	french_source_file.write(' '.join(str(t) for t in new_s)+'\n')

PAD = 0
EOS = 1
UNK = 2


def get_dic():
	french_to_token_dic = {'_PAD':PAD, '.':EOS, '_UNK':UNK}
	token_to_french_dic = {PAD:'_PAD', EOS:'.',UNK:'_UNK'}

	lines = open(home+'/data/french_french_data/fr_train_data').readlines()
	for line in lines:
		if(sys.version_info < (3,0)):
			line = line.decode('utf-8').strip()
		line = line.replace('\'',' \' ')
		words = nltk.word_tokenize(line)
		for w in words:
			if w not in french_to_token_dic.keys():
				french_to_token_dic[w] = len(french_to_token_dic.keys())
				token_to_french_dic[len(token_to_french_dic)] = w

	# print(len(french_to_token_dic.keys()))



	lines = open(home+'/data/french_french_data/fr_wordbword_train_data').readlines()
	for line in lines:
		if(sys.version_info < (3,0)):
			line = line.decode('utf-8').strip()
		line = line.replace('\'',' \' ')
		words = nltk.word_tokenize(line)
		for w in words:
			if w not in french_to_token_dic.keys():
				if w.lower() not in french_to_token_dic.keys():
					if w.title() not in french_to_token_dic.keys():
						# print(w)
						french_to_token_dic[w] = len(french_to_token_dic.keys())
						token_to_french_dic[len(token_to_french_dic)] = w
	return french_to_token_dic, token_to_french_dic


#write fr_fr token file

#only use to tokenize fr_fr wordby word file and fr_target file
def tokenize():
	f2token, token2f = get_dic()
	train_source_token_file = du.tokenize_file(home+'/data/french_french_data/fr_wordbword_train_data',f2token)
	train_target_token_file = du.tokenize_file(home+'/data/french_french_data/fr_train_data',f2token)
	test_source_token_file = du.tokenize_file(home+'/data/french_french_data/fr_wordbword_test_data',f2token)
	test_target_token_file = du.tokenize_file(home+'/data/french_french_data/fr_test_data',f2token)

	return train_source_token_file, train_target_token_file, test_source_token_file, test_target_token_file,f2token, token2f
	# print(len(french_to_token_dic.keys()))

def get_dictionaries():
	return eng_fr_dic, fr_eng_dic


#write a word by word french_source file



