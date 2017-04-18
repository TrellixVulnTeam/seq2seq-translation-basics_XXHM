import numpy as np
import random

def batch(inputs, max_sequence_length=None, reverse_input=True):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            if reverse_input:
                inputs_batch_major[i, (max_sequence_length-1)- j] = element
            else:
                inputs_batch_major[i,  j] = element


    # [batch_size, max_time] -> [max_time, batch_size]
    # print(inputs_batch_major)
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)


    return inputs_time_major, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]




# gets random sentences and pads them
def get_batch(filename1, filename2,batch_size):
    lines1 = open(filename1).readlines()
    lines2 = open(filename2).readlines()
    lines_indices = random.sample(range(0,len(lines1)),batch_size)
    batch_source = [lines1[x].strip().split() for x in lines_indices]
    batch_target = [lines2[x].strip().split() for x in lines_indices]
    batch_source = [list(map(int,x)) for x in batch_source]
    batch_target = [list(map(int,x)) for x in batch_target]
    return batch_source, batch_target


# x,y = get_batch("./data/eng_french_data/eng_test_data.ids1700", "./data/eng_french_data/fr_test_data.ids2300", 5)
# print("=======")
# print(x)
# print(y)
# x = [[11, 791, 7, 26, 383, 4], [12, 55, 100, 260, 353, 496, 44, 201, 4], [136, 117, 44, 1201, 4], [82, 13, 39, 1101, 6], [136, 326, 10, 233, 342, 685, 53, 44, 4]]


# print(batch(x, reverse_input=False))
# print(batch(x, reverse_input=True))
    # lines = open("./data/source_dev.en.ids10000").readlines()[:5]
