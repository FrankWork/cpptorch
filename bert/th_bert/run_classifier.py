
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, \
        accuracy_score

import torch
import torch as th
from torch.utils.data import TensorDataset, DataLoader

import tokenization
from modeling import BertConfig, BertForSequenceClassification
from modeling import BertSiameseModel
from optimization import BERTAdam
import collections

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class AtecProcessor(DataProcessor):
    """Processor for the ATEC nlp sim task"""
    def get_train_examples(self, data_dir):
        path = os.path.join(data_dir,"train.tsv")
        logger.info("Looking at {}".format(path))
                    
        return self._create_examples(
                self._read_tsv(path), "train")

    def get_dev_examples(self, data_dir):
        path = os.path.join(data_dir, "dev.tsv")
        return self._create_examples(
                self._read_tsv(path), "dev")
    
    def get_test_examples(self, file_path):
        lines = self._read_tsv(file_path)
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            examples.append(
                    InputExample(guid, text_a, text_b, '0'))
        return examples

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[3])
            examples.append(
                    InputExample(guid, text_a, text_b, label))
        return examples

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features


def convert_examples_to_siamese_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP]; [CLS], [SEP] with "- 4"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)

        tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
        types_a = [0] * len(tokens_a)
        types_b = [0] * len(tokens_b)
        
        ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        mask_a = [1] * len(ids_a)
        mask_b = [1] * len(ids_b)

        # Zero-pad up to the sequence length.
        while len(ids_a) < max_seq_length:
            ids_a.append(0)
            mask_a.append(0)
            types_a.append(0)
        while len(ids_b) < max_seq_length:
            ids_b.append(0)
            mask_b.append(0)
            types_b.append(0)

        assert len(ids_a) == max_seq_length
        assert len(ids_b) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens_a: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens_a]))
            logger.info("tokens_b: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens_b]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputSiameseFeatures(
                    ids_a, types_a, mask_a,
                    ids_b, types_b, mask_b,
                    label_id))
    return features


class InputSiameseFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens_a, types_a, mask_a,
                       tokens_b, types_b, mask_b, label_id):
        self.tokens_a = tokens_a
        self.types_a  = types_a
        self.mask_a = mask_a
        self.tokens_b = tokens_b
        self.types_b  = types_b
        self.mask_b = mask_b
        self.label_id = label_id
        
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, 
            named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, 
            named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file", default=None,  type=str,   required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name", default=None,  type=str,   required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file", default=None,  type=str,   required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir", default=None,  type=str,   required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint", default=None,  type=str,   
                                help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case", default=False,  action='store_true',
                                help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=128,   type=int,
                                help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", default=False,   action='store_true',
                                help="Whether to run training.")
    parser.add_argument("--do_eval", default=False,   action='store_true',
                                help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32,   type=int,
                                help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8,   type=int,
                                help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5,   type=float,
                                help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0,   type=float,
                                help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1,   type=float,
                                help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps", default=1000,   type=int,
                                help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda", default=False,   action='store_true',
                                help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                            help="local_rank for distributed training on gpus")
    parser.add_argument("--gpu_ids", type=str, default="0",
                            help="select one gpu to use")
    parser.add_argument('--seed',  type=int,  default=42,
                            help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumualte before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu', default=False,action='store_true',
                            help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16', default=False, action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=128,
                            help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    ################## atec 
    parser.add_argument('--do_submit', default=False, action='store_true',  help="submit to results to atec cloud")
    parser.add_argument('--train_devset', default=False, action='store_true',  help="")
    parser.add_argument("--test_in_file", default=None, type=str,  help="")
    parser.add_argument("--test_out_file", default=None, type=str,  help="")

    args = parser.parse_args()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "atec": AtecProcessor,
    }

    if args.local_rank == -1 and not args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
       
        #device = torch.device("cuda", args.gpu_id)
        n_gpu = len(args.gpu_ids.split(','))#torch.cuda.device_count()
    elif args.no_cuda:
        device = torch.device('cpu')
        n_gpu = 0
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    #if not args.do_train and not args.do_eval:
    #    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir) #, exist_ok=True)


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        if args.train_devset:
            eval_examples = processor.get_dev_examples(args.data_dir)
            train_examples += eval_examples

        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    logger.info('build model')
    #model = BertForSequenceClassification(bert_config, len(label_list))
    model = BertSiameseModel(bert_config, len(label_list))
    if args.init_checkpoint is not None:
        try:
            # just model.bert
            model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
        except RuntimeError as e:
            # all model
            import re
            new_state_dict = collections.OrderedDict()
            state_dict = torch.load(args.init_checkpoint, map_location='cpu')
            for key in state_dict.keys():
                new_key = re.sub("module\.", "", key)
                new_state_dict[new_key] = state_dict[key]
            model.load_state_dict(new_state_dict)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=[int(x) for x in
            args.gpu_ids.split(',')])

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
        ]
    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    tr_loss=None
    if args.do_train:
        from tqdm import tqdm, trange
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
        from torch.utils.data.distributed import DistributedSampler

        # train_features = convert_examples_to_features(
        #    train_examples, label_list, args.max_seq_length, tokenizer)
        train_features = convert_examples_to_siamese_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        # all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        # all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        # all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        # train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        all_tokens_a = torch.tensor([f.tokens_a for f in train_features], dtype=torch.long)
        all_types_a = torch.tensor([f.types_a for f in train_features], dtype=torch.long)
        all_mask_a = torch.tensor([f.mask_a for f in train_features], dtype=torch.long)
        all_tokens_b = torch.tensor([f.tokens_b for f in train_features], dtype=torch.long)
        all_types_b = torch.tensor([f.types_b for f in train_features], dtype=torch.long)
        all_mask_b = torch.tensor([f.mask_b for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_tokens_a, all_types_a, all_mask_a,
                all_tokens_b, all_types_b, all_mask_b, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            with tqdm(train_dataloader, desc="Iteration") as pbar:
                for step, batch in enumerate(pbar):
                    batch = tuple(t.to(device) for t in batch)
                    # input_ids, input_mask, segment_ids, label_ids = batch
                    # loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                    tokens_a, types_a, mask_a, tokens_b, types_b, mask_b, label_ids = batch
                    loss, logits = model(tokens_a, types_a, mask_a, tokens_b, types_b,
                            mask_b, label_ids)

                    if n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.fp16 and args.loss_scale != 1.0:
                        # rescale loss for fp16 training
                        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                        loss = loss * args.loss_scale
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    tr_loss += loss.item()
                    
                    preds = th.argmax(logits, dim=1)
                    acc = accuracy_score(label_ids.detach().cpu().numpy(),
                                preds.detach().cpu().numpy())

                    pbar.set_postfix(loss=loss.item(), acc=acc.item())
                    #nb_tr_examples += input_ids.size(0)
                    nb_tr_examples += tokens_a.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16 or args.optimize_on_cpu:
                            if args.fp16 and args.loss_scale != 1.0:
                                # scale down gradients for fp16 training
                                for param in model.parameters():
                                    param.grad.data = param.grad.data / args.loss_scale
                            is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                            if is_nan:
                                logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                                args.loss_scale = args.loss_scale / 2
                                model.zero_grad()
                                continue
                            optimizer.step()
                            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                        else:
                            optimizer.step()
                        model.zero_grad()
                        global_step += 1

        torch.save(model.state_dict(), os.path.join(args.output_dir,
                                            "pytorch_model.bin"))

    if args.do_eval:
        from tqdm import tqdm, trange
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
        from torch.utils.data.distributed import DistributedSampler

        eval_examples = processor.get_dev_examples(args.data_dir)
        # eval_features = convert_examples_to_features(
        #    eval_examples, label_list, args.max_seq_length, tokenizer)
        eval_features = convert_examples_to_siamese_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        # all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        # all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        # eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        all_tokens_a = torch.tensor([f.tokens_a for f in eval_features], dtype=torch.long)
        all_types_a = torch.tensor([f.types_a for f in eval_features], dtype=torch.long)
        all_mask_a = torch.tensor([f.mask_a for f in eval_features], dtype=torch.long)
        all_tokens_b = torch.tensor([f.tokens_b for f in eval_features], dtype=torch.long)
        all_types_b = torch.tensor([f.types_b for f in eval_features], dtype=torch.long)
        all_mask_b = torch.tensor([f.mask_b for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_tokens_a, all_types_a, all_mask_a,
                all_tokens_b, all_types_b, all_mask_b, all_label_ids)

        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        for batch in tqdm(eval_dataloader):
            tokens_a, types_a, mask_a, tokens_b, types_b, mask_b, label_ids = batch
            # input_ids = input_ids.to(device)
            # input_mask = input_mask.to(device)
            # segment_ids = segment_ids.to(device)
            # label_ids = label_ids.to(device)

            with torch.no_grad():
                # tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)
                tmp_eval_loss, logits = model(tokens_a, types_a, mask_a,
                        tokens_b, types_b, mask_b, label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            preds = np.argmax(logits, axis=1)

            y_true.extend(label_ids)
            y_pred.extend(preds)

            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            #nb_eval_examples += input_ids.size(0)
            nb_eval_examples += tokens_a.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'precision': precision_score(y_true, y_pred),
                  'recall': recall_score(y_true, y_pred), 
                  'f1': f1_score(y_true, y_pred)}
        if tr_loss is not None:
            result['loss'] = tr_loss/nb_tr_steps

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results *****")
            writer.write("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_submit:
        eval_examples = processor.get_test_examples(args.test_in_file)
        # eval_features = convert_examples_to_features(
        #    eval_examples, label_list, args.max_seq_length, tokenizer)
        eval_features = convert_examples_to_siamese_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        # all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        # all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        # all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        #eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        
        all_tokens_a = torch.tensor([f.tokens_a for f in eval_features], dtype=torch.long)
        all_types_a = torch.tensor([f.types_a for f in eval_features], dtype=torch.long)
        all_mask_a = torch.tensor([f.mask_a for f in eval_features], dtype=torch.long)
        all_tokens_b = torch.tensor([f.tokens_b for f in eval_features], dtype=torch.long)
        all_types_b = torch.tensor([f.types_b for f in eval_features], dtype=torch.long)
        all_mask_b = torch.tensor([f.mask_b for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        # eval_data = TensorDataset(all_tokens_a, all_types_a, all_mask_a,
        #        all_tokens_b, all_types_b, all_mask_b, all_label_ids)


        #eval_sampler = SequentialSampler(eval_data)
        #eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        y_pred = []
        batch_size = args.eval_batch_size
        for i in range(0, len(all_label_ids), batch_size):
            # input_ids = all_input_ids[i:i+batch_size]
            # input_mask = all_input_mask[i:i+batch_size]
            # segment_ids = all_segment_ids[i:i+batch_size]
            # label_ids = all_label_ids[i:i+batch_size]
            tokens_a = all_tokens_a[i:i+batch_size]
            types_a = all_types_a[i:i+batch_size]
            mask_a = all_mask_a[i:i+batch_size]
            tokens_b = all_tokens_b[i:i+batch_size]
            types_b = all_types_b[i:i+batch_size]
            mask_b = all_mask_b[i:i+batch_size]

            with torch.no_grad():
                # _, logits = model(input_ids, segment_ids, input_mask, label_ids)
                _, logits = model(tokens_a, types_a, mask_a,
                        tokens_b, types_b, mask_b)

            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)

            y_pred.extend(preds)

        with open(args.test_out_file, "w") as writer:
            for i, y in enumerate(y_pred):
                writer.write('{}\t{}\n'.format(i+1, y))

if __name__ == "__main__":
    main()
