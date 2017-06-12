	                                                                         
	                                                                         
	 .-.           .-.  .-.               .    .         .                   
	(   )         (   )(   )              |\  /|         |      o            
	 `-.  .-. .-.,  .'  `-.  .-. .-.,     | \/ | .-.  .-.|--.   .  .--. .-.  
	(   )(.-'(   | /   (   )(.-'(   |     |    |(   )(   |  |   |  |  |(.-'  
	 `-'  `--'`-'|'---' `-'  `--'`-'|     '    ' `-'`-`-''  `--' `-'  `-`--' 
	            -|-                -|-                                       
	             '                  '                                        

                                             
			                                              
			 .---.                .      .                
			   |                  |     _|_   o           
			   |.--..-.  .--. .--.| .-.  |    .  .-. .--. 
			   ||  (   ) |  | `--.|(   ) |    | (   )|  | 
			   ''   `-'`-'  `-`--'`-`-'`-`-'-' `-`-' '  `-
			                                              
                                             



___________________________________________________________
                                                           |
ONLY USE ON PYTHON 3, DUE TO UNICODE ISSUES WITH PYTHON 2  |
___________________________________________________________|

# Directory Structure

|
|-----translate_mini.py           The main script to train/test
|-----------run(batch_size, test_boolean):
|								  the main function, if test_boolean is true, opens up interactive mode where user can input sentence and get
|								  translation back
|-----------get_translation(sentence):
|                      			  helper function to get direct translation from default setup
|-----helpers.py                  Used to convert the data from tokenized files into arrays/matrices
|-----data_custom_utils.py 		  Used to tokenize a source file
|-----wbw_utils.py  			  Takes care of tokenizing file during word-by-word french->french (uses data_custom_utils)
|------data
       |------eng_french-data     data files for eng-french training/testing
       |------french_french-data  data files for wbw french-french training/testing
       |------normal_checkpoints  saves training values for eng-french training
       |------wbw_checkpoints     saves training values for wbw french-french training
|------data_analysis              scripts which plot the data for each experiment (effect of optimizer,
								  bidirectional encoder, making it deep and finally word-by-word model )


# How to run:

python translate_mini.py <mode normal or wbw>  <train or test>


Eg: 

python translate_mini.py normal train   

  ------------>trains normal english-french model

  Eg output:
		batch 0
		  minibatch perplexity: 320.6136991519436
		  sample 1:
		      input   > He was in good _UNK last summer .
		    predicted > Je Je Je Je . . . . . . .
		  sample 2:
		      input   > _UNK !
		    predicted > . . . . .
		  sample 3:
		      input   > It 's going _UNK !
		    predicted > Je . . . . .

python translate_mini.py normal test

 ------------>tests the normal setup

 Eg: output

		 Reading model parameters from /Users/vivek/Google Drive/Colby17S/translate_mini/data/normal_checkpoints/model.ckpt-0
		> hello
		. . . .
		> who is this?
		Je . . .
		>

python translate_mini.py wbw train


------------>trains word-by-word english-french model

python translate_mini.py wbw test

------------>tests word-by-word english-french model




# DESCRIPTION OF SETUP

The translate_mini.py script uses data from data folder, tokenizes it and trains it. After training, it 
saves the values into appropriate checkpoints directory. When the test mode is run or training is resumed, it
restores teh values from  latest checkpoints. To reset the training, delete the files in checkpoints directory.

