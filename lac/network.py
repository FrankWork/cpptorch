"""Defines a Bi-LSMT CRF model."""

import argparse

import torch
import torch as th
from torch import nn

from crf import ConditionalRandomField


def sequence_mask(lens: torch.Tensor, max_len: int = None) -> torch.ByteTensor:
    """
    Compute sequence mask.

    Parameters
    ----------
    lens : torch.Tensor
        Tensor of sequence lengths ``[batch_size]``.

    max_len : int, optional (default: None)
        The maximum length (optional).

    Returns
    -------
    torch.ByteTensor
        Returns a tensor of 1's and 0's of size ``[batch_size x max_len]``.

    """
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().item()

    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = torch.autograd.Variable(ranges)

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask

class Config(object):
    def __init__(self):
        self.word_dim = 128
        self.hidden_dim = 256
        self.num_gru_layers = 2
	self.num_labels = 57

class BiGruLayer(nn.Module):
    def __init__(self, args):
        super(BiGruLayer, self).__init__()

        word_dim = args.word_dim
        hidden_dim = args.hidden_dim

        self.fc = nn.Linear(word_dim, hidden_dim *3)
        self.gru = nn.GRU(hidden_dim*3, hidden_dim,
                batch_first=True)
        self.fc_r = nn.Linear(word_dim, hidden_dim*3)
        self.gru_r = nn.GRU(hidden_dim*3, hidden_dim,
                batch_first=True)

    def forward(self, x):
        forward, _ = self.gru(self.fc(x))
        x_backward = ?
        backward, _ = self.gru_r(self.fc_r(x_backward))
        return th.cat([forward, backward], dim=-1)

class LacNet(nn.Module):
    def __init__(self, args):
        super(LacNet, self).__init__()

        vocab_size = args.vocab_size
        word_dim = args.word_dim
        num_gru_layers = args.num_gru_layers
        num_labels = args.num_labels
	hidden_dim = args.hidden_dim

        self.word_emb = nn.Embedding(vocab_size, word_dim)
        self.gru_layers = nn.ModuleList([
                BiGruLayer(args) for _ in num_gru_layers])
        self.emission = nn.Linear(hidden_dim*2, num_labels)
        
        self.crf  = ConditionalRandomField(num_labels)
        self.crf_decode = crf_decoding()
        self.crf_cost = linear_chain_crf()
    
    def forward(self, x):
        x = self.word_emb(x)
        for gru in self.gru_layers:
            x = gru(x)
        x = self.emission(x)

        if lens is None:
            lens = torch.tensor([words.size(0)], device=words.device)
        mask = sequence_mask(lens)

        # Run features through Viterbi decode algorithm.
        preds = self.crf.viterbi_tags(feats, mask)
        
        # loglik = self.crf(feats, labs, mask=mask)
        # loss = -1. * loglik
        return preds

    def get_trainable_params(self):
        module_params = [
            self.char_feats_layer.parameters(),
            self.rnn.parameters(),
            self.rnn_to_crf.parameters(),
            self.crf.parameters()
        ]
        return module_params

