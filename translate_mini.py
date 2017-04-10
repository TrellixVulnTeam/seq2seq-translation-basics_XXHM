import data_utils as utils
import numpy as np
import tensorflow as tf
import helpers
from tensorflow.contrib.tensorboard.plugins import projector
import os
import random

#=====================DEFINE DATA SOURCES============================================#
data_dir = "data"
from_train_path = "data/source_train.en"
to_train_path = "data/target_train.fr"
from_dev_path = "data/source_dev.en"
to_dev_path = "data/target_dev.fr"
CHK_DIR="./data/checkpoints"
LOGDIR = "./log"
training_source_file = "./data/source_train.en.ids10000"


tf.reset_default_graph()

PAD = 0
EOS = 1

file_vocab_size = 10000 #pseudo vocab size
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20 
decoder_hidden_units = 20

## In the original paper, it was 1000, we can increase it



#=====================PREPARE TRAINING DATA============================================#

from_train, to_train, from_dev, to_dev, _, _ = utils.prepare_data(data_dir, 
 	from_train_path, 
 	to_train_path, 
 	from_dev_path, 
 	to_dev_path, 
 	file_vocab_size,
   file_vocab_size)


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

# build embeddings
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size] ,-1.0,1, dtype = tf.float32),name="embedding_name")


'''				  ___ _  _  ___ ___  ___  ___ ___ 
				 | __| \| |/ __/ _ \|   \| __| _ \
				 | _|| .` | (_| (_) | |) | _||   /
				 |___|_|\_|\___\___/|___/|___|_|_\
				                                  '''
with tf.variable_scope('encoder') as scope:

	encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
	encoder_cell = tf.contrib.rnn.BasicLSTMCell(encoder_hidden_units)
	# make encoder system using dynamic RNN which does BAsic RNN under the hood
	# 
	encoder_outputs,encoder_final_state =  tf.nn.dynamic_rnn(cell=encoder_cell, 
			inputs= encoder_inputs_embedded,
			sequence_length = encoder_inputs_length, 
			dtype =tf.float32,
			time_major = True) 
			# SEE API details, basically if trsue, these Tensors must be shaped [max_time, batch_size, depth].
			# If false, these Tensors must be shaped [batch_size, max_time, depth].True is a bit more efficient because it avoids 
			# transposes at the beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default
			#  this function accepts input and emits output in batch-major form.






	# IMPROVEMENT ON DYMAMIC RNN
	# ((encoder_fw_inputs, 
	# 	encoder_bw_inputs, 
	# 	encoder_fw_final_state, 
	# 	encoder_bw_final_state)) = ( tf.nn.bidorectional_dynamic_rnn(cell_fw=encoder_cell, 
	# 		cell_bw= encoder_cell, 
	# 		inputs= encoder_inputs_embedded,
	# 		sequence_length = encoder_inputs_length, 
	# 		dtype =tf.float32,
	# 		time_major = True)


'''				  ___  ___ ___ ___  ___  ___ ___ 
				 |   \| __/ __/ _ \|   \| __| _ \
				 | |) | _| (_| (_) | |) | _||   /
				 |___/|___\___\___/|___/|___|_|_\
'''


with tf.variable_scope('decoder') as scope:

	decoder_cell = tf.contrib.rnn.BasicLSTMCell(decoder_hidden_units)
	encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
	decoder_lengths = encoder_inputs_length+3

	#define weights and biases
	W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1,1), dtype = tf.float32)
	b = tf.Variable(tf.zeros([vocab_size]), dtype = tf.float32)


	#create padded inputs for the decoder from the word embeddings

	#were telling the program to test a condition, and trigger an error if the condition is false.
	assert EOS == 1 and PAD == 0

	eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
	pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

	#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
	eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
	pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)



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

			next_input = tf.nn.embedding_lookup(embeddings, prediction)
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
	decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))



	#final prediction
	decoder_prediction = tf.argmax(decoder_logits, 2)



	###OPTIMIZE
	stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
	    logits=decoder_logits,
	)

	#loss function
	loss = tf.reduce_mean(stepwise_cross_entropy)
	#train it 
	train_op = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver()


with tf.Session() as sess:
	ckpt = tf.train.get_checkpoint_state(CHK_DIR)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		sess.run(tf.global_variables_initializer())




	# sess.run(tf.global_variables_initializer())
	# E = tf.Variable([0.0], name='embedding')
	# place = tf.placeholder(tf.float32, shape=embeddings.shape)
	# set_E = tf.assign(embeddings)
	# sess.run(set_E, feed_dict={place: embeddings})


	###Training
	batch_size = 100
	batches = helpers.random_sequences(length_from=3, length_to=8,
	                                   vocab_lower=2, vocab_upper=10,
	                                   batch_size=batch_size)  #this method returns an iterator
	
	#each iteration returns a list of 100 lists
	# train_from_lines = open(from_train).readlines()
	# train_to_lines = open(to_train).readlines()

	def next_feed():
		
  		# lines_indices = random.sample(range(0,len(train_from_lines)),batch_size)
  		# batch_source,batch_target = helpers.get_batch(from_train, to_train, batch_size)

  		# [train_from_lines[x].strip().split() for x in lines_indices]
  		# batch_target = [train_to_lines[x].strip().split() for x in lines_indices]


		batch = next(batches)
		encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
		decoder_targets_, _ = helpers.batch(
	        [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
	    )
		return { encoder_inputs: encoder_inputs_, encoder_inputs_length: encoder_input_lengths_, decoder_targets: decoder_targets_}

	loss_track = []
	max_batches = 1001
	batches_in_epoch = 100
	

	#tensorboard
	writer = tf.summary.FileWriter('log', sess.graph)
	

	try:
	    for batch in range(max_batches):
	        fd = next_feed()
	        _, l = sess.run([train_op, loss], fd)
	        loss_track.append(l)

	        if batch == 0 or batch % batches_in_epoch == 0:
	            saver.save(sess, os.path.join(CHK_DIR, "model.ckpt"), batch)
	            # sess.run(embeddings)


	            print('batch {}'.format(batch))
	            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
	            predict_ = sess.run(decoder_prediction, fd)
	            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
	                print('  sample {}:'.format(i + 1))
	                print('    input     > {}'.format(inp))
	                print('    predicted > {}'.format(pred))
	                if i >= 2:
	                    break
	            print()

	except KeyboardInterrupt:
	    print('training interrupted')


	#matplotlib inline
	import matplotlib.pyplot as plt
	plt.plot(loss_track)
	plt.show()
	print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))
	config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
	embedding_conf = config.embeddings.add()
	embedding_conf.tensor_name = embeddings.name
	projector.visualize_embeddings(writer, config)

	# embedding_conf.metadata_path = os.path.join( 'metadata.tsv')




