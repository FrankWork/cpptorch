# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import collections
import json
import os
import torch
import numpy as np
import pickle

import tokenization
from modeling import BertConfig, BertModel
from logger import logger
from atec_data import AtecProcessor, convert_examples_to_siamese_features, \
        convert_siamese_features_to_dataset
from tqdm import tqdm

def to_device(tokens, types, mask, device):
    return tokens.to(device), types.to(device), mask.to(device)

def bert_feature(output_file, model, dataloader, device):
    features = []
    
    logger.info('Total {} batches.'.format(len(dataloader)))
    for batch in tqdm(dataloader):
        unique_id, tokens_a, types_a, mask_a, \
            tokens_b, types_b, mask_b, label_ids = batch

        with torch.no_grad():
            _, out_a = model(tokens_a.to(device), token_type_ids=None, 
                    attention_mask=mask_a.to(device))
            out_a = out_a.cpu()
            torch.cuda.empty_cache()

        with torch.no_grad():
            _, out_b = model(tokens_b.to(device), token_type_ids=None,
                    attention_mask=mask_b.to(device))
            out_b = out_b.cpu()
            torch.cuda.empty_cache()

        for b, id_ in enumerate(unique_id):
            id_ = int(id_.item())
            features.append( (id_, out_a[b], out_b[b], label_ids[b]))
            
    logger.info('write features to {}'.format(output_file))
    with open(output_file, "wb") as f:
        pickle.dump(features, f, protocol=2)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",default=None,type=str,required=True,help="The input data dir")
    parser.add_argument("--vocab_file", default=None, type=str, required=True, 
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                            "This specifies the model architecture.")
    parser.add_argument("--init_checkpoint", default=None, type=str, required=True, 
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument('--gpu_id', default=0, type=int, help='')
    parser.add_argument('--no_cuda', default=False, type=bool, help='')
    ## Other parameters
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_lower_case", default=True, action='store_true', 
                        help="Whether to lower case the input text. Should be True for uncased "
                            "models and False for cased models.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")

    args = parser.parse_args()

    if not args.no_cuda:
        device = torch.device("cuda", args.gpu_id)
        n_gpu = 1 # torch.cuda.device_count()
    else:
        device = torch.device('cpu')
        n_gpu = 0 
    logger.info("device {} n_gpu {}".format(device, n_gpu))

    layer_indexes = [int(x) for x in args.layers.split(",")]

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    cache_path = os.path.join(args.output_dir, 'tmp_data.pkl')
    if not os.path.exists(cache_path):
        tokenizer = tokenization.FullTokenizer(
            vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

        processor = AtecProcessor()
        label_list = processor.get_labels()

        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_siamese_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Dataset info *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.batch_size) 

        train_dataloader = convert_siamese_features_to_dataset(
                train_features, args.batch_size)
       
        dev_examples = processor.get_dev_examples(args.data_dir)
        dev_features = convert_examples_to_siamese_features(
            dev_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Dataset info *****")
        logger.info("  Num examples = %d", len(dev_examples))
        logger.info("  Batch size = %d", args.batch_size) 
        dev_dataloader = convert_siamese_features_to_dataset(
                dev_features, args.batch_size)

        with open(cache_path, 'wb') as f:
            pickle.dump([train_dataloader, dev_dataloader], f)
    else:
        logger.info("load data from cache file: {}".format(cache_path))
        with open(cache_path, 'rb') as f:
            train_dataloader, dev_dataloader = pickle.load(f)
   

    model = BertModel(bert_config)
    if args.init_checkpoint is not None:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    model.to(device)
    model.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    logger.info('extract train features.')
    bert_feature(os.path.join(args.output_dir, "train.feat"), model, train_dataloader, device)
    logger.info('extract dev features.')
    bert_feature(os.path.join(args.output_dir, "dev.feat"), model, dev_dataloader, device)
    
if __name__ == "__main__":
    main()
