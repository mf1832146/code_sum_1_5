

def get_config():
    config = {
        # data set path
        'origin_data_dir': '../data',
        'output_data_dir': '../data_set',

        # the min occur time of word occur in the word vocab
        'min_occur': 20,
        # the max size of tree
        'max_tree_size': 100,
        # the max size of nl
        'max_nl_len': 30,
        # max path len
        'k': 5
    }

    return config
