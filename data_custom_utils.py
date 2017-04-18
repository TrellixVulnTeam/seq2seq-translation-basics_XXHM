import nltk
import os 
fr_dic = {}
eng_dic = {}
import sys

PAD = 0
EOS = 1
UNK = 2
# print(sys.version_info < (3,5))
def tokenize_file(source_file, source_dic):
	'''creates two dictionaries and tokenizes file'''
	
	# for line in lines:
	
	source_token_file = source_file+ ".tokens"
	#source files
	source_lines = open(source_file).readlines()
	source_token_file = open(source_token_file,'a+')
	for line in source_lines:
		if(sys.version_info < (3,0)):
			line = line.decode('utf-8').strip()
		words = nltk.word_tokenize(line)
		token_sentence = ' '.join([ str(source_dic.get(w,UNK)) for w in words ])+'\n'

		source_token_file.write(token_sentence)

	return source_file+ ".tokens"


'''test the tokenizer'''
# def regain_file(source_file, source_dic):
# 	'''creates two dictionaries and tokenizes file'''
	
# 	# for line in lines:
	
# 	source_token_file = source_file+ ".tokens"
# 	#source files
# 	source_lines = open(source_file).readlines()
# 	source_token_file = open(source_token_file,'a+')
# 	for line in source_lines:
# 		if(sys.version_info < (3,0)):
# 			line = line.decode('utf-8').strip()
# 		words = nltk.word_tokenize(line)
# 		token_sentence = ' '.join([ str(source_dic[int(w)]) for w in words])+'\n'
# 		source_token_file.write(token_sentence)

# 	return source_file+ ".tokens"





def create_dic(filename):
	dic = {'_PAD':PAD, '.':EOS, '_UNK':UNK}
	rev_dic = {PAD:'_PAD', EOS:'.',UNK:'_UNK'}
	# dic = {}
	lines = open(filename).readlines()
	for line in lines:
		if(sys.version_info < (3,0)):
			line = line.decode('utf-8').strip()
		words = nltk.word_tokenize(line)
		for w in words:
			if w not in dic.keys():
				dic[w] = len(dic.keys())
				rev_dic[len(rev_dic.keys())] = w
	return dic, rev_dic

def tokenize(source_file):
	word_to_token_dic, token_to_word_dic = create_dic(source_file)	
	token_file= tokenize_file(source_file, word_to_token_dic)
	return token_file, word_to_token_dic, token_to_word_dic

# print(d)
# print(len(d.keys()))
