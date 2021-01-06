from collections import Counter
import pickle
from utils import read_pickle


class Vocab(object):
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self, output_data_dir):
        self.ast2index = None
        self.index2ast = None

        self.nl2index = None
        self.index2nl = None

        self.code2index = None
        self.index2code = None

        self.output_data_dir = output_data_dir

    def generate_vocab(self, ast_tokens, nl_tokens, code_tokens, min_occur):
        # code_vocab
        code_counter = Counter([x for c in code_tokens for x in c])

        self.code2index = {w: i for i, w in enumerate(
            ['<PAD>', '<UNK>'] +
            sorted([x[0] for x in code_counter.items() if x[1] > min_occur]
                   ))}

        # word_vocab
        nl_counter = Counter([x for w in nl_tokens for x in w])
        self.nl2index = {w: i for i, w in enumerate(
            ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] +
            sorted([x[0] for x in nl_counter.items() if x[1] > min_occur]
                   ))}

        ast_counter = Counter([l for s in ast_tokens for l in s])
        self.ast2index = {w: i for i, w in enumerate(
            ['<PAD>', '<UNK>'] +
            sorted([x[0] for x in ast_counter.items() if x[1] > min_occur]
                   ))}

        # save to file
        pickle.dump(self.ast2index, open(self.output_data_dir + "/ast_w2i.pkl", "wb"))
        pickle.dump(self.nl2index, open(self.output_data_dir + "/nl_w2i.pkl", "wb"))
        pickle.dump(self.code2index, open(self.output_data_dir + "/code_w2i.pkl", "wb"))

    def load_vocab(self):
        self.ast2index = read_pickle(self.output_data_dir + "/ast_w2i.pkl")
        self.nl2index = read_pickle(self.output_data_dir + "/nl_w2i.pkl")
        self.code2index = read_pickle(self.output_data_dir + "/code_w2i.pkl")

        self.index2ast = {v: k for k, v in self.ast2index.items()}
        self.index2nl = {v: k for k, v in self.nl2index.items()}
        self.index2code = {v: k for k, v in self.code2index.items()}



