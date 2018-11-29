import json
import os
import torch as th
import torch
from torch import nn
import numpy as np
# import pickle
import argparse
from modeling import BertConfig
from logger import logger
from tqdm import tqdm, trange

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_score, recall_score, f1_score

class SiameseModel(nn.Module):
    def __init__(self, config, num_labels):
        super(SiameseModel, self).__init__()
        #self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*4, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0,
                        std=config.initializer_range)
            # elif isinstance(module, BERTLayerNorm):
            #     module.beta.data.normal_(mean=0.0,
            #             std=config.initializer_range)
            #     module.gamma.data.normal_(mean=0.0,
            #             std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, a, b, labels=None):
        x = torch.cat([a, b, torch.abs(a-b), a*b], -1)

        x = self.dropout(x)
        logits = self.classifier(x)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        else:
            return logits

def load_data(path, batch_size, random_sample=True):
    buf = np.load(path)

    unique_id = th.from_numpy(buf['ids'])
    feat_a = th.from_numpy(buf['feat_a'])
    feat_b = th.from_numpy(buf['feat_b'])
    labels = th.from_numpy(buf['labels'])

    data = TensorDataset(unique_id, feat_a, feat_b, labels)

    sampler = None
    if random_sample:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    loader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default=None,type=str,required=True,help="The input data dir")
    parser.add_argument("--bert_config_file", default=None, type=str,
            required=True, help='')
    parser.add_argument('--gpu_id', default=0, type=int, help='')
    parser.add_argument('--do_train', default=True, type=bool, help='')
    parser.add_argument('--do_eval', default=True, type=bool, help='')
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="")
    parser.add_argument("--num_train_epochs", default=3, type=int, help=".")
    parser.add_argument("--output_dir",default=None,type=str,required=True,help="")
    args = parser.parse_args()
    
    logger.info('build model')
    device = torch.device("cuda", args.gpu_id)
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    model = SiameseModel(bert_config, 2)
    model = model.to(device)
    
    if args.do_train:
        optimizer = th.optim.Adam(model.parameters(), lr = args.learning_rate)
        
        train_path = os.path.join(args.data_dir, 'train.npz')
        logger.info('load data from {}'.format(train_path))
        train_data = load_data(train_path, args.batch_size, True)


        model.train()
        for _ in trange(args.num_train_epochs, desc="Epoch"):
            with tqdm(train_data) as t:
                for batch in t: 
                    id_, a, b, label = batch
                    loss, logits = model(a.to(device), b.to(device),label.to(device))
                    logits = logits.detach().cpu()
                    preds = th.argmax(logits, dim=1)
                    acc = th.sum(preds ==  label)
                    t.set_postfix(loss=loss.item(), acc=acc.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        logger.info('save model to {}'.format(args.output_dir))
        th.save(model.state_dict(), 
                os.path.join(args.output_dir, 'dssm_model.bin'))

    if args.do_eval:
        dev_path = os.path.join(args.data_dir, 'dev.npz')
        logger.info('load data from {}'.format(dev_path))
        dev_data = load_data(dev_path, args.batch_size, True)

        model.eval()
        y_true = []
        y_pred = []
        for batch in tqdm(dev_data):
            id_, a, b, label = batch
            with th.no_grad():
                logits = model(a.to(device), b.to(device))
                logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)
            y_pred.extend(preds)
            y_true.extend(label.cpu().numpy())

        logger.info('eval results:')
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        logger.info("p {} r {} f1 {}".format(p, r, f1))

if __name__ == '__main__':
    main()
