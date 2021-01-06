import nltk
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np


def batch_evaluate(comments, predicts, nl_i2w, extra_vocab):
    batch_size = comments.size(0)
    references = []
    hypothesises = []
    for i in range(batch_size):
        reference = [nl_i2w[c.item()] if c.item() in nl_i2w else extra_vocab[c.item()] for c in comments[i]]
        if '<EOS>' in reference and reference.index('<EOS>') < len(reference):
            reference = reference[:reference.index('<EOS>')]
        hypothesis = [nl_i2w[c.item()] if c.item() in nl_i2w else extra_vocab[c.item()] for c in predicts[i]][1:]
        if '<EOS>' in hypothesis and hypothesis.index('<EOS>') < len(hypothesis):
            hypothesis = hypothesis[:hypothesis.index('<EOS>')]
        references.append(reference)
        hypothesises.append(hypothesis)
    return references, hypothesises


def batch_bleu(comments, predicts, nl_i2w, extra_vocab,  i):
    references, hypothesises = batch_evaluate(comments, predicts, nl_i2w, extra_vocab)
    if i == 0:
        for j in range(3):
            print("真实:", references[j], "\n猜测:", hypothesises[j])
    scores = []
    for i in range(len(references)):
        bleu_score = nltk_sentence_bleu(hypothesises[i], references[i])
        scores.append(bleu_score)
    return scores


def nltk_sentence_bleu(hypothesis, reference, order=4):
    cc = SmoothingFunction()
    if len(reference) < 4:
        return 0
    try:
        score = nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method1)
    except Exception as e:
        print('reference: ', reference)
        print('hypothesis: ', hypothesis)
        print(e)
    return score


class BLEU4(Metric):
    def __init__(self, id2nl, output_transform=lambda x: x, device=None):
        super(BLEU4, self).__init__(output_transform, device=device)
        self._id2nl = id2nl

    @reinit__is_reduced
    def reset(self):
        self._bleu_scores = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        (y_pred, extra_vocab), y = output
        extra_vocab = {v.item(): k for k, v in extra_vocab.items()}
        scores = batch_bleu(y, y_pred, self._id2nl, extra_vocab, self._num_examples)
        self._bleu_scores += np.sum(scores)
        self._num_examples += len(scores)

    @sync_all_reduce("_bleu_scores", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("BLEU4 must have "
                                     "at least one example before it can be computed.")
        return self._bleu_scores / self._num_examples