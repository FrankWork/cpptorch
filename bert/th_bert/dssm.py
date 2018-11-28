import json
import os
import torch as th
import torch
from torch import nn
import pickle
import argparse
from modeling import BertConfig
from logger import logger

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class SiameseModel(nn.Module):
    def __init__(self, config, num_labels):
        super(SiameseModel, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*4, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0,
                        std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0,
                        std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0,
                        std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, a, b, labels=None):
        x = torch.cat([a, b, torch.abs(a-b), a*b], -1)

        x = self.dropout(x)
        logits = self.classifier(x)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

    #dtype = torch.long 
    #unique_id = torch.tensor([f.unique_id for f in features],dtype=dtype)
    #tokens_a = torch.tensor([f.tokens_a for f in features], dtype=dtype)
    #types_a = torch.tensor([f.types_a for f in features], dtype=dtype)
    #mask_a = torch.tensor([f.mask_a for f in features], dtype=dtype)
    #tokens_b = torch.tensor([f.tokens_b for f in features], dtype=dtype)
    #types_b = torch.tensor([f.types_b for f in features], dtype=dtype)
    #mask_b = torch.tensor([f.mask_b for f in features], dtype=dtype)
    #label_ids = torch.tensor([f.label_id for f in features], dtype=dtype)
    #data = TensorDataset(unique_id, tokens_a, types_a, mask_a,
    #        tokens_b, types_b, mask_b, label_ids)

    #sampler = None
    #if random_sample:
    #    sampler = RandomSampler(data)
    #else:
    #    sampler = SequentialSampler(data)

    #loader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    #return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default=None,type=str,required=True,help="The input data dir")
    parser.add_argument("--bert_config_file", default=None, type=str,
            required=True, help='')
    parser.add_argument('--gpu_id', default=0, type=int, help='')
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="")
    parser.add_argument("--num_train_epochs", default=3, type=int, help=".")
    parser.add_argument("--output_dir",default=None,type=str,required=True,help="")
    args = parser.parse_args()
    
    device = torch.device("cuda", args.gpu_id)

    #bert_config = BertConfig.from_json_file(args.bert_config_file)

    train_path = os.path.join(args.data_dir, 'train.feat')
    logger.info('load data from {}'.format(train_path))
    with open(train_path, 'rb') as f:
        train_feat = pickle.load(f)

    dev_path = os.path.join(args.data_dir, 'dev.feat')
    logger.info('load data from {}'.format(dev_path))
    with open(dev_path, 'rb') as f:
        dev_feat = pickle.load(f)
    logger.info(dev_feat[0])
    # id_, a, b, label

if __name__ == '__main__':
    main()
