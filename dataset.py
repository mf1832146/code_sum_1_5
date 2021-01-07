import torch.utils.data as data
import torch
import numpy as np
from torch.autograd import Variable
from utils import subsequent_mask, load_json
import math
import random


class TreeDataSet(data.Dataset):
    def __init__(self, data_dir, data_set_name, matrix_list, max_ast_size, max_nl_size, k,
                 slice_file_size, shuffle=False, with_more_than_k=True):
        print('loading ' + data_set_name + ' data...')
        self.matrices_path = self.data_dir + '/' + str(self.max_ast_size) + '_' + str(self.k)
        if with_more_than_k:
            self.matrices_path += '_' + 'ignore_max_than_k'
        else:
            self.matrices_path += '_' + 'with_max_than_k'
        self.matrices_path += '/' + self.data_set_name

        self.data = load_json(data_dir + '/' + data_set_name + '.json')
        self.ast_seq = load_json(self.matrices_path + '_pre_order_seq.json')
        self.data_dir = data_dir
        self.data_set_name = data_set_name
        self.data_set_len = len(self.data)

        self.slice_file_size = slice_file_size
        if self.slice_file_size == -1:
            self.slice_file_size = self.data_set_len

        self.shuffle = shuffle
        slice_file_len = int(math.ceil(self.data_set_len / self.slice_file_size))
        self.slices = list(range(slice_file_len))
        if self.shuffle:
            random.shuffle(self.slices)

        self.matrix_list = matrix_list
        self.max_ast_size = max_ast_size
        self.max_nl_size = max_nl_size
        self.k = k

        self.current_slice_index = 0
        self.loaded_data_size = 0


    def __len__(self):
        return self.data_set_len

    def get_slice_data(self):
        matrices = np.load(self.matrices_path + '_matrices_' + str(self.slices[self.current_slice_index]) + '.npz')

        self.par_matrix_slice = matrices['par']
        self.bro_matrix_slice = matrices['bro']
        self.sem_matrix_slice = matrices['sem']
        self.sem_mask_slice = matrices['mask']

        current_begin = self.slices[self.current_slice_index] * self.slice_file_size
        self.loaded_data_size += self.par_matrix_slice.shape[0]
        current_end = current_begin + self.par_matrix_slice.shape[0]

        self.data_slice = self.data[current_begin: current_end]
        self.ast_seq_slice = self.ast_seq[current_begin: current_end]

        if self.shuffle:
            shuffle_ix = np.random.permutation(np.arange(len(self.data_slice)))
            self.data_slice = [self.data_slice[id_] for id_ in shuffle_ix]
            self.ast_seq_slice = [self.ast_seq_slice[id_] for id_ in shuffle_ix]
            self.par_matrix_slice = self.par_matrix_slice[shuffle_ix]
            self.bro_matrix_slice = self.bro_matrix_slice[shuffle_ix]
            self.sem_matrix_slice = self.sem_matrix_slice[shuffle_ix]
            self.sem_mask_slice = self.sem_mask_slice[shuffle_ix]

    def __getitem__(self, index):
        if index == 0:
            self.current_slice_index = 0
            self.loaded_data_size = 0
            random.shuffle(self.slices)
            self.get_slice_data()
        if index >= self.loaded_data_size:
            self.current_slice_index += 1
            self.get_slice_data()

        index = index - self.current_slice_index * self.slice_file_size
        pre_order_seq = self.ast_seq_slice[index]
        nl = self.data_slice[index]['nl']
        nl = nl[:self.max_nl_size-2]
        nl = ['<SOS>'] + nl + ['<EOS>']

        if 'par' in self.matrix_list:
            rel_par_ids = torch.from_numpy(self.par_matrix_slice[index]).long()
        else:
            rel_par_ids = None
        if 'bro' in self.matrix_list:
            rel_bro_ids = torch.from_numpy(self.bro_matrix_slice[index]).long()
        else:
            rel_bro_ids = None
        if 'sem' in self.matrix_list:
            rel_semantic_ids = torch.from_numpy(self.sem_matrix_slice[index]).long()
        else:
            rel_semantic_ids = None
        semantic_mask = torch.from_numpy(self.sem_mask_slice[index]).long()

        pre_order_seq = pre_order_seq + ['<PAD>' for i in range(self.max_ast_size - len(pre_order_seq))]
        nl = nl + ['<PAD>' for i in range(self.max_nl_size - len(nl))]

        return pre_order_seq, nl, rel_par_ids, rel_bro_ids, rel_semantic_ids, semantic_mask

    @staticmethod
    def collect_fn(inputs, vocab):
        extra_vocab = {}

        expanded_x = []
        expanded_y = []

        batch_seq = []
        batch_nl = []
        batch_rel_par_ids = []
        batch_rel_bro_ids = []
        batch_rel_semantic_ids = []
        batch_semantic_mask = []

        for i in range(len(inputs)):
            pre_order_seq, nl, rel_par_ids, rel_bro_ids, rel_semantic_ids, semantic_mask = inputs[i]

            vec_seq = []
            vec_nl = []
            for j, token in enumerate(pre_order_seq):
                if token in extra_vocab.keys():
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # vec_seq.append(extra_vocab[token])
                    vec_seq.append(vocab.UNK)
                elif token not in vocab.nl2index:
                    # ignore non-leaf nodes
                    if semantic_mask[j] == 0:
                        vec_seq.append(vocab.UNK)
                    else:
                        extra_vocab[token] = len(vocab.nl2index) + len(extra_vocab)
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        #vec_seq.append(extra_vocab[token])
                        vec_seq.append(vocab.UNK)
                else:
                    vec_seq.append(vocab.nl2index[token])

            for token in nl:
                if token in extra_vocab.keys():
                    vec_nl.append(extra_vocab[token])
                elif token not in vocab.nl2index:
                    extra_vocab[token] = len(vocab.nl2index) + len(extra_vocab)
                    vec_nl.append(extra_vocab[token])
                else:
                    vec_nl.append(vocab.nl2index[token])

            expanded_x.append(torch.LongTensor(vec_seq))
            expanded_y.append(torch.LongTensor(vec_nl))

            batch_seq.append(
                [vocab.ast2index[x] if x in vocab.ast2index else vocab.ast2index['<UNK>'] for x in pre_order_seq])
            batch_nl.append([vocab.nl2index[x] if x in vocab.nl2index else vocab.nl2index['<UNK>'] for x in nl])

            if rel_par_ids is not None:
                batch_rel_par_ids.append(rel_par_ids)
            if rel_bro_ids is not None:
                batch_rel_bro_ids.append(rel_bro_ids)
            if rel_semantic_ids is not None:
                batch_rel_semantic_ids.append(rel_semantic_ids)
            batch_semantic_mask.append(semantic_mask)

        batch_seq = torch.LongTensor(batch_seq)
        batch_nl = torch.LongTensor(batch_nl)

        rel_ids = []

        if len(batch_rel_par_ids) > 0:
            batch_rel_par_ids = torch.stack(batch_rel_par_ids, dim=0)
            rel_ids.append(batch_rel_par_ids)
        if len(batch_rel_bro_ids) > 0:
            batch_rel_bro_ids = torch.stack(batch_rel_bro_ids, dim=0)
            rel_ids.append(batch_rel_bro_ids)
        if len(batch_rel_semantic_ids) > 0:
            batch_rel_semantic_ids = torch.stack(batch_rel_semantic_ids, dim=0)
            rel_ids.append(batch_rel_semantic_ids)
        batch_semantic_mask = torch.stack(batch_semantic_mask, dim=0).unsqueeze(-2).unsqueeze(1)
        expanded_x = torch.stack(expanded_x, dim=0)
        expanded_y = torch.stack(expanded_y, dim=0)

        batch_predicts = batch_nl[:, 1:]
        batch_nl = batch_nl[:, :-1]
        predicts = expanded_y[:, 1:]

        extra_vocab = {k: torch.IntTensor([v]) for k, v in extra_vocab.items()}

        return get_batch_data(batch_seq, batch_nl, vocab.PAD, rel_ids, dict(), expanded_x, batch_semantic_mask), batch_predicts


def get_batch_data(src, tgt, pad, rel_ids=[], extra_vocab=dict(), expanded_x=[], semantic_mask=[]):

    src_mask = (src != pad).unsqueeze(-2).unsqueeze(1)

    tgt_mask = (tgt != pad).unsqueeze(-2).unsqueeze(1)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

    batch = {
        'src': src,
        'tgt': tgt,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'rel_ids': rel_ids,
        'extra_vocab': extra_vocab,
        'expanded_x': expanded_x,
        'semantic_mask': semantic_mask
    }

    return batch
