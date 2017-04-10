import numpy as np
import random

def batch(inputs, max_sequence_length=None):
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
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
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

# batches = random_sequences(length_from=3, length_to=8,
#                                    vocab_lower=2, vocab_upper=10,
#                                    batch_size=5)
# b = next(batches)
# print(b)

# print(batch(b))
# def get_batch_size(file)
def get_batch(filename1, filename2,batch_size):
    lines1 = open(filename1).readlines()
    lines2 = open(filename2).readlines()
    lines_indices = random.sample(range(0,len(lines1)),batch_size)
    batch_source = [lines1[x].strip().split() for x in lines_indices]
    batch_target = [lines2[x].strip().split() for x in lines_indices]
    batch_source = [map(int,x) for x in batch_source]
    batch_target = [map(int,x) for x in batch_target]
    return batch_source, batch_target
#     batch_1 = [][lines1[x].strip().split() for x in lines_indices]
#     for line in lines1:
#         l = []

# x,y = get_batch("./data/source_dev.en.ids10000", "./data/target_dev.fr.ids10000", 5)
# print("=======")
# print(x)
# print(y)
    # lines = open("./data/source_dev.en.ids10000").readlines()[:5]
