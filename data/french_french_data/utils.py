import sys
import nltk
eng_vocab_file = open('eng_vocab_test','a+')
dic = {}
#write out a eng vocab file
# def write_eng_vocab(eng_file):
# 	lines = open(eng_file).readlines()
# 	for line in lines:
# 		if(sys.version_info < (3,0)):
# 			line = line.decode('utf-8').strip()
# 		words = nltk.word_tokenize(line)
# 		for w in words:
# 			if w not in dic.keys():
# 				dic[w] = len(dic.keys())
# 				eng_vocab_file.write(w+'\n')
# 					# rev_dic[len(rev_dic.keys())] = w
# 		# return dic, rev_dic
# 		# 

# write_eng_vocab('eng_test_data')
#reate a eng_french token dictionary _word by word
eng_fr_dic = {}

lines_eng = open('eng_vocab_test').readlines()
lines_fr = open('french_vocab_test').readlines()

for i in range(len(lines_eng)):
	l = lines_eng[i].strip()
	eng_fr_dic[l] = lines_fr[i].strip()


# french_source_file = open('fr_wordbword_test_data','a+')
# eng_lines = open('eng_test_data').readlines()
# for sentence in eng_lines:
# 	new_s = []
# 	words = nltk.word_tokenize(sentence)
# 	for w in words:
# 		fr_word = eng_fr_dic.get(w)

# 		fr_word = eng_fr_dic[w]
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

	lines = open('fr_train_data').readlines()
	for line in lines:
		if(sys.version_info < (3,0)):
			line = line.decode('utf-8').strip()
		line = line.replace('\'',' \' ')
		words = nltk.word_tokenize(line)
		for w in words:
			if w not in french_to_token_dic.keys():
				french_to_token_dic[w] = len(french_to_token_dic.keys())
				token_to_french_dic[len(token_to_french_dic)] = w

	print(len(french_to_token_dic.keys()))



	lines = open('fr_wordbword_train_data').readlines()
	for line in lines:
		if(sys.version_info < (3,0)):
			line = line.decode('utf-8').strip()
		line = line.replace('\'',' \' ')
		words = nltk.word_tokenize(line)
		for w in words:
			if w not in french_to_token_dic.keys():
				if w.lower() not in french_to_token_dic.keys():
					if w.title() not in french_to_token_dic.keys():
						print(w)
						french_to_token_dic[w] = len(french_to_token_dic.keys())
						token_to_french_dic[len(token_to_french_dic)] = w



#write fr_fr token file
# def tokenize_file(source_file, source_dic):
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
# 		token_sentence = ' '.join([ str(source_dic.get(w,UNK)) for w in words ])+'\n'

# 		source_token_file.write(token_sentence)

# 	return source_file+ ".tokens"

# tokenize_file('fr_wordbword_train_data',french_to_token_dic)

# print(len(french_to_token_dic.keys()))




#write a word by word french_source file



