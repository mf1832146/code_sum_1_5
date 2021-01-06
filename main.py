import argparse

from config import get_config
from solver import Solver
from data_pre_process.vocab import Vocab

config = get_config()


def parse():
    parser = argparse.ArgumentParser(description='ast transformer')
    parser.add_argument('-model_dir', default='train_model', help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-model', default='standard_transformer')
    parser.add_argument('-num_step', type=int, default=250)
    parser.add_argument('-num_layers', type=int, default=6, help='layer num')
    parser.add_argument('-model_dim', type=int, default=384)
    parser.add_argument('-num_heads', type=int, default=12)
    parser.add_argument('-ffn_dim', type=int, default=1536)

    parser.add_argument('-k', type=int, default=config['k'])
    parser.add_argument('-max_tree_size', type=int, default=config['max_tree_size'])
    parser.add_argument('-max_nl_len', type=int, default=config['max_nl_len'])

    parser.add_argument('-use_par', type=int, default=1)
    parser.add_argument('-use_bro', type=int, default=1)
    parser.add_argument('-use_semantic', type=int, default=1)

    parser.add_argument('-data_dir', default='../data_set')
    parser.add_argument('-dropout', type=float, default=0.3)

    parser.add_argument('-load', action='store_true', help='load pretrained model')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-visual', action='store_true')
    parser.add_argument('-gold_test', action='store_true')
    parser.add_argument('-generate_matrix', action='store_true')

    parser.add_argument('-load_epoch', type=str, default='0')

    parser.add_argument('-log_dir', default='train_log/')

    parser.add_argument('-g', type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    vocab = Vocab(get_config()['output_data_dir'])
    vocab.load_vocab()
    solver = Solver(args, vocab)

    if args.train:
        solver.train()
    elif args.test:
        solver.test(load_epoch=args.load_epoch)
    elif args.visual:
        solver.visualize(load_epoch=args.load_epoch)
    elif args.gold_test:
        solver.gold_test(load_epoch=args.load_epoch)
