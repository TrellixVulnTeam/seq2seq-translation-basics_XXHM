import data_utils as utils

data_dir = "data"
from_train_path = "data/source_train.en"
to_train_path = "data/target_train.fr"
from_dev_path = "data/source_dev.en"
to_dev_path = "data/target_dev.fr"
from_vocabulary_size = 10000
to_vocabulary_size = 10000


from_train, to_train, from_dev, to_dev, _, _ = utils.prepare_data(data_dir, 
 	from_train_path, 
 	to_train_path, 
 	from_dev_path, 
 	to_dev_path, 
 	from_vocabulary_size,
    to_vocabulary_size)


