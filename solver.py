from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch.utils.data.dataloader import DataLoader
import torch

from blue import BLEU4
from dataset import TreeDataSet
from ignite.contrib.handlers.tensorboard_logger import *

from data_pre_process.vocab import Vocab
from module.train_component import LabelSmoothing, Train, GreedyEvaluate
from module.transformer import make_model


class Solver:
    def __init__(self, args, vocab):
        self.vocab = vocab
        self.args = args
        self.epoch = 0

        self.feature_list = []
        if self.args.model.startswith('ast'):
            if self.args.use_par == 1:
                self.feature_list.append('par')
            if self.args.use_bro == 1:
                self.feature_list.append('bro')
            if self.args.use_semantic == 1:
                self.feature_list.append('sem')

        self.model = make_model(src_vocab=len(self.vocab.ast2index),
                                tgt_vocab=len(self.vocab.nl2index),
                                N=self.args.num_layers,
                                d_model=self.args.model_dim,
                                d_ff=self.args.ffn_dim,
                                k=self.args.k,
                                h=self.args.num_heads,
                                num_features=len(self.feature_list),
                                dropout=self.args.dropout,
                                model_name=self.args.model)

    def train(self):
        with_more_than_k = True if self.args.mtk == 1 else False
        train_data_set = TreeDataSet(self.args.data_dir,
                                     'train',
                                     self.feature_list,
                                     self.args.max_tree_size,
                                     self.args.max_nl_len,
                                     self.args.k,
                                     4096,
                                     shuffle=True,
                                     with_more_than_k=with_more_than_k)
        valid_data_set = TreeDataSet(self.args.data_dir,
                                     'valid',
                                     self.feature_list,
                                     self.args.max_tree_size,
                                     self.args.max_nl_len,
                                     self.args.k,
                                     -1,
                                     shuffle=False,
                                     with_more_than_k=with_more_than_k)
        train_loader = DataLoader(dataset=train_data_set,
                                  batch_size=self.args.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda x: TreeDataSet.collect_fn(x, self.vocab))
        valid_loader = DataLoader(dataset=valid_data_set,
                                  batch_size=self.args.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda x: TreeDataSet.collect_fn(x, self.vocab))

        device = "cpu"

        print(torch.cuda.is_available())

        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.g)
            device = "cuda"
            print('use gpu')

        self.model.to(device)

        if self.args.load_epoch != '0':
            self.load_model(load_epoch=self.args.load_epoch)
            print('load epoch ', self.args.load_epoch)

        model_opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        criterion = LabelSmoothing(padding_idx=0, smoothing=0.0)

        train_model = Train(self.model)
        greedy_evaluator = GreedyEvaluate(self.model, self.args.max_nl_len, Vocab.SOS)

        trainer = create_supervised_trainer(train_model, model_opt, criterion, device)

        metric_valid = {"bleu": BLEU4(self.vocab.index2nl)}

        """
                train + generator
                validation + generator
                """
        validation_evaluator = create_supervised_evaluator(greedy_evaluator, metric_valid, device)

        # save model
        save_handler = ModelCheckpoint('checkpoint/' + self.args.model + '_' + ' '.join(self.feature_list), n_saved=10,
                                       filename_prefix='',
                                       create_dir=True,
                                       global_step_transform=lambda e, _: e.state.epoch + int(self.args.load_epoch),
                                       require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), save_handler, {self.args.model: self.model})

        # early stop
        early_stop_handler = EarlyStopping(patience=20, score_function=self.score_function, trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            print('Train Epoch ' + str(self.epoch + int(self.args.load_epoch)) + ' end')
            validation_evaluator.run(valid_loader)
            self.epoch += 1

        tb_logger = TensorboardLogger(self.args.log_dir + self.args.model + '_' + '_'.join(self.feature_list) + '/')

        tb_logger.attach(
            validation_evaluator,
            log_handler=OutputHandler(tag="validation", metric_names=["bleu"], global_step_transform=global_step_from_engine(trainer)),
            event_name=Events.EPOCH_COMPLETED
        )

        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(
                tag="training", output_transform=lambda loss: {"batchloss": loss}, metric_names="all"
            ),
            event_name=Events.ITERATION_COMPLETED(every=50)
        )

        trainer.run(train_loader, max_epochs=self.args.num_step)
        tb_logger.close()

    @staticmethod
    def score_function(engine):
        bleu = engine.state.metrics['bleu']
        return bleu

    def load_model(self, load_epoch):
        model_path = 'checkpoint/'+ self.args.model + '/' + self.args.model + '_' + load_epoch + '.pth'

        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
