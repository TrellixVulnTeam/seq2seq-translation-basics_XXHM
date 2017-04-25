import data_custom_utils as utils
import wbw_utils as wbw_utils
import numpy as np
import tensorflow as tf
import helpers
from tensorflow.contrib.tensorboard.plugins import projector
import os
import random
import math
import csv
import copy
import sys
import nltk
#=====================DEFINE DATA SOURCES============================================#
data_dir = "data"

home =  os.getcwd()
print("home", home)
from_train_path = home+"/data/eng_french_data/eng_train_data"
to_train_path = home+"/data/eng_french_data/fr_train_data"
from_dev_path = home+"/data/eng_french_data/eng_test_data"
to_dev_path = home+"/data/eng_french_data/fr_test_data"
LOGDIR = "./log"


tf.reset_default_graph()

PAD = 0
EOS = 1

NORMAL = 0
WBW = 1



input_embedding_size = 20
encoder_hidden_units = 20 
decoder_hidden_units = 40
max_batches = 8001
batches_in_epoch = 10
number_of_layers = 3

## In the original paper, it was 1000, we can increase it



#=====================PREPARE TRAINING DATA============================================#
arguments = ['wbw','train']
if (len(sys.argv) < 3):
	print("Using default values")
else:
	arguments[0] = sys.argv[1]
	arguments[1] = sys.argv[2]

mode = arguments[0]
user_test = False if arguments[1] == "train" else True
if (mode == 'h' or mode == 'help'):
	print("usage: translate_mini.py <mode> <train/test>")
	exit()
if ( mode== 'normal'):
	CHK_DIR=home+"/data/normal_checkpoints"
	from_train, source_to_token, token_to_source = utils.tokenize(from_train_path)
	to_train, target_to_token, token_to_target = utils.tokenize(to_train_path)
	from_dev = utils.tokenize_file(from_dev_path, source_to_token)
	to_dev = utils.tokenize_file(to_dev_path, target_to_token)
elif (mode == "wbw"):
	CHK_DIR=home+"/data/wbw_checkpoints"
	from_train, to_train, from_dev, to_dev, source_to_token, token_to_source = wbw_utils.tokenize()
	eng_fr_dic, fr_eng_dic = wbw_utils.get_dictionaries()
	target_to_token = source_to_token
	token_to_target = token_to_source
else:
	print("only normal and wbw mode available")
	exit()
vocab_size_encoder = len(source_to_token.keys()) #pseudo vocab size
vocab_size_decoder = len(target_to_token.keys())



#====================BUILD THE TENSORFLOW GRAPH =======================================#
'''
	____  __  ________    ____     ________  ________   __________  ___    ____  __  __
   / __ )/ / / /  _/ /   / __ \   /_  __/ / / / ____/  / ____/ __ \/   |  / __ \/ / / /
  / __  / / / // // /   / / / /    / / / /_/ / __/    / / __/ /_/ / /| | / /_/ / /_/ / 
 / /_/ / /_/ // // /___/ /_/ /    / / / __  / /___   / /_/ / _, _/ ___ |/ ____/ __  /  
/_____/\____/___/_____/_____/    /_/ /_/ /_/_____/   \____/_/ |_/_/  |_/_/   /_/ /_/   
																					   
'''


#======================DEFINE THE PLACEHOLDERS================#
encoder_inputs = tf.placeholder(shape = (None,None), dtype= tf.int32, name = "encoder_inputs")
encoder_inputs_length = tf.placeholder(shape = (None,), dtype= tf.int32, name = "encoder_inputs_length")
decoder_targets= tf.placeholder(shape = (None,None), dtype= tf.int32, name = "decoder_targets")
decoder_inputs_length = tf.placeholder(shape = (None,), dtype= tf.int32, name = "encoder_inputs_length")

# build embeddings
embeddings_source = tf.Variable(tf.random_uniform([vocab_size_encoder, input_embedding_size] ,-1.0,1, dtype = tf.float32),name="embedding_name")
if (mode == "normal"):
	embeddings_target = tf.Variable(tf.random_uniform([vocab_size_decoder, input_embedding_size] ,-1.0,1, dtype = tf.float32),name="embedding_name")
else: #because in word by word, source and target language same
	embeddings_target = embeddings_source



'''				  ___ _  _  ___ ___  ___  ___ ___ 
				 | __| \| |/ __/ _ \|   \| __| _ \
				 | _|| .` | (_| (_) | |) | _||   /
				 |___|_|\_|\___\___/|___/|___|_|_\
												  '''
with tf.variable_scope('encoder') as scope:

	encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_source, encoder_inputs)
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(encoder_hidden_units)
	encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * number_of_layers,state_is_tuple=True)
	# make encoder system using dynamic RNN which does BAsic RNN under the hood
	# 
	# vanilla encoder
	# encoder_outputs,encoder_final_state =  tf.nn.dynamic_rnn(cell=encoder_cell, 
	# 		inputs= encoder_inputs_embedded,
	# 		sequence_length = encoder_inputs_length, 
	# 		dtype =tf.float32,
	# 		time_major = True) 
	# 		# SEE API details, basically if trsue, these Tensors must be shaped [max_time, batch_size, depth].
	# 		# If false, these Tensors must be shaped [batch_size, max_time, depth].True is a bit more efficient because it avoids 
	# 		# transposes at the beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default
	# 		#  this function accepts input and emits output in batch-major form.






	# IMPROVEMENT ON DYMAMIC RNN
	(encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state,encoder_bw_final_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, cell_bw= encoder_cell, inputs= encoder_inputs_embedded,sequence_length = encoder_inputs_length, dtype =tf.float32,time_major = True)



	#Concatenates tensors along one dimension.
	encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

	#letters h and c are commonly used to denote "output value" and "cell state". 
	#http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
	#Those tensors represent combined internal state of the cell, and should be passed together. 
	encoder_final_state = [0]*number_of_layers
	for i in range(len(encoder_final_state)):
		state_c = tf.concat(
		(encoder_fw_final_state[i].c, encoder_bw_final_state[i].c), 1)
		state_h = tf.concat(
		(encoder_fw_final_state[i].h, encoder_bw_final_state[i].h), 1)
		final_state  = tf.contrib.rnn.LSTMStateTuple(
			c=state_c,
			h=state_h
		)
		encoder_final_state[i] = final_state
	encoder_final_state = tuple(encoder_final_state)




'''				  ___  ___ ___ ___  ___  ___ ___ 
				 |   \| __/ __/ _ \|   \| __| _ \
				 | |) | _| (_| (_) | |) | _||   /
				 |___/|___\___\___/|___/|___|_|_\
'''


with tf.variable_scope('decoder') as scope:

	# decoder_cell = tf.contrib.rnn.BasicLSTMCell(decoder_hidden_units)
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(decoder_hidden_units)
	decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * number_of_layers,state_is_tuple=True)

	encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
	decoder_lengths = decoder_inputs_length
	# print(decoder_lengths)

	#define weights and biases
	W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size_decoder], -1,1), dtype = tf.float32)
	b = tf.Variable(tf.zeros([vocab_size_decoder]), dtype = tf.float32)


	#create padded inputs for the decoder from the word embeddings

	#were telling the program to test a condition, and trigger an error if the condition is false.
	assert EOS == 1 and PAD == 0

	eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
	pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

	#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
	eos_step_embedded = tf.nn.embedding_lookup(embeddings_target, eos_time_slice)
	pad_step_embedded = tf.nn.embedding_lookup(embeddings_target, pad_time_slice)



	# We are manually going to implement decoder recurrent neural network using raw_rnn
	# Thios is because we have unlike dynamic rnn, which takes in a list of tensors t1...tn and 
	# feeds them one by one, here we have to generate the sequence ourselves, looking at the previous 
	# output, 
	# Also remember, in raw_rnn api, we have to define a function which does that, that is define the
	#  input and cell state to feed in

	#but first, we have to ,as shown in the encoder-decoder system, feed in iniyial input as EOS, 
	#first cell state = encoder_final state
	def loop_fn_initial():
		initial_elements_finished = (0>= decoder_lengths)
		initial_input = eos_step_embedded
		initial_cell_state = encoder_final_state
		initial_cell_output = None
		initial_loop_state = None
		return(initial_elements_finished,
			 initial_input,
			 initial_cell_state, 
			 initial_cell_output,
			 initial_loop_state)



	#soft attention mechanism,
	def loop_fn_transition(time,previous_output,previous_state, previous_loop_state):

		# to get the next input 
		def get_next_input():
			#dot product between previous output and weights+ biases
			k = tf.matmul(previous_output, W)
			output_logits = tf.add(k, b)


			#returns the index with highest probability
			prediction=tf.argmax(output_logits, axis = 1)

			next_input = tf.nn.embedding_lookup(embeddings_target, prediction)
			return next_input


		elements_finished = (time >= decoder_lengths)
		finished = tf.reduce_all(elements_finished)

		input = tf.cond(finished,lambda:pad_step_embedded,get_next_input) 


		#set previous to current
		state = previous_state
		output = previous_output
		loop_state = None

		return(elements_finished,
				input,
				state,
				output,
				loop_state )


	#use one of the loop functions for raw_rnn
	def loop_fn(time,previous_output,previous_state, previous_loop_state):
		if previous_state is None: #time = 0
			assert previous_output is None and previous_state is None
			return loop_fn_initial()
		else:
			return loop_fn_transition(time,previous_output,previous_state, previous_loop_state)

	decoder_outputs_ta, decoder_final_state, _ =  tf.nn.raw_rnn(decoder_cell,loop_fn)
	decoder_outputs = decoder_outputs_ta.stack()


	#to convert output to human readable prediction
	#we will reshape output tensor

	#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
	#reduces dimensionality
	decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
	#flettened output tensor
	decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
	#pass flattened tensor through decoder
	decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
	#prediction vals
	decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size_decoder))



	#final prediction
	decoder_prediction = tf.argmax(decoder_logits, 2)



	###OPTIMIZE
	stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		labels=tf.one_hot(decoder_targets, depth=vocab_size_decoder, dtype=tf.float32),
		logits=decoder_logits,
	)

	#loss function
	loss = tf.reduce_mean(stepwise_cross_entropy)
	tf.summary.scalar('loss', loss)
	
	#train it 
	# train_op = tf.train.AdagradOptimizer(0.001).minimize(loss)
	train_op = tf.train.AdamOptimizer().minimize(loss)
	# train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter(CHK_DIR, sess.graph)
test_writer = tf.summary.FileWriter(CHK_DIR)


saver = tf.train.Saver()


#gets the next batch of inputs and targets , random batch_size sentences , put padding as required
#from_: the source file
#to_ : the target file
def next_feed(from_,to_,batch_size):
		
	# lines_indices = random.sample(range(0,len(train_from_lines)),batch_size)
	# print(from_,to_)
	batch_source,batch_target = helpers.get_batch(from_, to_, batch_size)

	# [train_from_lines[x].strip().split() for x in lines_indices]
	# batch_target = [train_to_lines[x].strip().split() for x in lines_indices]


	encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch_source, reverse_input=False)
	# print("encoder lengths",encoder_input_lengths_)
	'''encoder_inputs_reverse'''
	
	decoder_targets_, decoder_inputs_length_ = helpers.batch(
		[(sequence) + [EOS] + [PAD] * (2) for sequence in batch_target], reverse_input=False
		)
	# print("decoder lengths",decoder_inputs_length_)
	return { encoder_inputs: encoder_inputs_, encoder_inputs_length: encoder_input_lengths_, decoder_targets: decoder_targets_,
		decoder_inputs_length: decoder_inputs_length_}




###
#  #####  ###       #     # #   # 
#    #    ###     # # #   # # # #
#    #    #  #   #     #  # #   #
#
###########






sess = tf.Session()
global_step = 0


def init_session(load_checkpoint=True):
	ckpt = tf.train.get_checkpoint_state(CHK_DIR)
	print(CHK_DIR)
	print(ckpt)
	if load_checkpoint and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		global_step =int(ckpt.model_checkpoint_path.split("-")[-1])
	else:
		print("Created model with fresh parameters.")
		sess.run(tf.global_variables_initializer())
	return sess

init_session()
def run(batch_size_, user_test= False):
	# print("calling train-size")
	print("user_test is" ,user_test)
	config = tf.ConfigProto()
	if (1 < 2) :
	# config.gpu_options.allocator_type = 'BFC'
	# config.gpu_options.per_process_gpu_memory_fraction=0.5
	# with tf.Session(config = config) as sess:
	# 	global_step = 0
	# 	ckpt = tf.train.get_checkpoint_state(CHK_DIR)
	# 	if load_checkpoint and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
	# 		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
	# 		saver.restore(sess, ckpt.model_checkpoint_path)
	# 		global_step =int(ckpt.model_checkpoint_path.split("-")[-1])
	# 	else:
	# 		print("Created model with fresh parameters.")
	# 		sess.run(tf.global_variables_initializer())
		###Training
		batch_size = batch_size_
		loss_train = []
		loss_dev = []
		
		#tensorboard
		# writer = tf.summary.FileWriter(CHK_DIR, sess.graph)
		
		# _,token_to_source = utils.initialize_vocabulary(from_vocab_path)
		# _,fr_words = utils.initialize_vocabulary(to_vocab_path)
		# fr_words = token_to_target


		try:
			if user_test == False:  #train
				for batch in range(max_batches):
					fd = next_feed(from_train,to_train,batch_size)
					_, l,summary = sess.run([train_op, loss,merged], fd)
					perplexity = math.exp(float(l)) if l < 300 else float("inf")

					loss_train.append(perplexity)
					test_writer.add_summary(summary, batch)


					if batch+global_step == 0 or batch+global_step % batches_in_epoch == 0:
						fd2 = next_feed(from_dev, to_dev,batch_size)
						saver.save(sess, os.path.join(CHK_DIR, "model.ckpt"), batch)
						# sess.run(embeddings)
						l2 = sess.run(loss, fd2)
						p_ = math.exp(float(l2)) if l2 < 300 else float("inf")
						loss_dev.append(p_)

						print('batch {}'.format(batch+global_step))
						print('  minibatch perplexity: {}'.format(p_))
						predict_ = sess.run(decoder_prediction, fd2)
						
						pred_str = []
						for i, (inp, pred) in enumerate(zip(fd2[encoder_inputs].T, predict_.T)):
							print('  sample {}:'.format(i + 1))
							# print('    input     > {}'.format(inp))
							# print('    predicted > {}'.format(pred))

							input_str = [token_to_source[x] for x in inp if x != 0]

							
							target_str =[]
							for x in pred:
								if x != 0:
									target_str.append(token_to_target[x])
							input_string = " ".join([str(x) for x in input_str  ])
							target_string =" ".join([str(x) for x in target_str ])
							print('      input   >',input_string)
							print('    predicted >',target_string)
							if i >= 2:
								break

						print()
			else: #test
				batch_size_ = 1
				sys.stdout.write("> ")
				sys.stdout.flush()
				sentence = sys.stdin.readline()
				while sentence:
					words = nltk.word_tokenize(sentence)
					if mode == "wbw":
						wbw_words = [eng_fr_dic.get(w,"UNK") for w in words]
						print("wbw> ", " ".join([w for w in wbw_words]))
						#if word by word, then first get french word and then tokenize
						#if wbw, source_to_token dictionary will map french words t token
						batch_source = [[source_to_token.get(w,2) for w in wbw_words]]						
					else:
						batch_source = [[source_to_token.get(w,2) for w in words]]

					batch_target = [[1]*len(batch_source)]
					# print(tokenized_sentence)
					# print(ra)
					encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch_source, reverse_input=False)
					decoder_targets_, decoder_inputs_length_ = helpers.batch(
					[(sequence) + [EOS] + [PAD] * (2) for sequence in batch_target], reverse_input=False
					)

					feed = { encoder_inputs: encoder_inputs_, encoder_inputs_length: encoder_input_lengths_, decoder_targets: decoder_targets_, decoder_inputs_length: decoder_inputs_length_}
					predict_ = sess.run(decoder_prediction, feed)
					output_str = [token_to_target[out] for out in predict_.T[0]]
					print(" ".join([s for s in output_str]))
					print("> ", end="")
					sys.stdout.flush()
					sentence = sys.stdin.readline()

		except KeyboardInterrupt:
			print('training interrupted')
		return loss_train, loss_dev

def get_translation(eng_sentence):	
	batch_size_ = 1
	words = nltk.word_tokenize(eng_sentence)
	wbw_words = [eng_fr_dic.get(w,"UNK") for w in words]
	print("wbw> ", " ".join([w for w in wbw_words]))
	#if word by word, then first get french word and then tokenize
	#if wbw, source_to_token dictionary will map french words t token
	batch_source = [[source_to_token.get(w,2) for w in wbw_words]]
	batch_target = [[1]*len(batch_source)]
	encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch_source, reverse_input=False)
	decoder_targets_, decoder_inputs_length_ = helpers.batch(
	[(sequence) + [EOS] + [PAD] * (2) for sequence in batch_target], reverse_input=False
	)

	feed = { encoder_inputs: encoder_inputs_, encoder_inputs_length: encoder_input_lengths_, decoder_targets: decoder_targets_, decoder_inputs_length: decoder_inputs_length_}
	predict_ = sess.run(decoder_prediction, feed)
	output_str = [token_to_target[out] for out in predict_.T[0]]
	output_str= " ".join([s for s in output_str])
	return output_str


	
def write_results(loss_train,loss_test, mode):
	train_loss_file = open("results_train.csv",'a+')
	dev_loss_file = open("results_test.csv",'a+')

	csvwriter1 = csv.writer(train_loss_file, delimiter=',',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
	csvwriter2 = csv.writer(dev_loss_file, delimiter=',',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)

	csvwriter1.writerow([mode+" "+ str(number_of_layers)+"_train"]+loss_train)
	csvwriter2.writerow([mode+" "+ str(number_of_layers)+"_test"]+loss_dev)


	# print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))


	# embedding_conf.metadata_path = os.path.join( 'metadata.tsv')

# loss_train, loss_dev = run(10, user_test=user_test)

# print(get_translation("I am dog."))

