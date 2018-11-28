import csv
import os
import tokenization
import torch
from logger import logger
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

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
            guid = i #"%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[3])
            examples.append(
                    InputExample(guid, text_a, text_b, label))
        return examples

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b, label):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

class InputSiameseFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens_a, types_a, mask_a,
                       tokens_b, types_b, mask_b, label_id):
        self.unique_id = unique_id
        self.tokens_a = tokens_a
        self.types_a  = types_a
        self.mask_a = mask_a
        self.tokens_b = tokens_b
        self.types_b  = types_b
        self.mask_b = mask_b
        self.label_id = label_id

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

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
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

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
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
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
            logger.info("guid: %s" % (example.unique_id))
            logger.info("tokens_a: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens_a]))
            logger.info("tokens_b: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens_b]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputSiameseFeatures(
                    example.unique_id,
                    ids_a, types_a, mask_a,
                    ids_b, types_b, mask_b,
                    label_id))
    return features

def convert_siamese_features_to_dataset(features, batch_size,
        random_sample=False):
    dtype = torch.long 
    unique_id = torch.tensor([f.unique_id for f in features],dtype=dtype)
    tokens_a = torch.tensor([f.tokens_a for f in features], dtype=dtype)
    types_a = torch.tensor([f.types_a for f in features], dtype=dtype)
    mask_a = torch.tensor([f.mask_a for f in features], dtype=dtype)
    tokens_b = torch.tensor([f.tokens_b for f in features], dtype=dtype)
    types_b = torch.tensor([f.types_b for f in features], dtype=dtype)
    mask_b = torch.tensor([f.mask_b for f in features], dtype=dtype)
    label_ids = torch.tensor([f.label_id for f in features], dtype=dtype)
    data = TensorDataset(unique_id, tokens_a, types_a, mask_a,
            tokens_b, types_b, mask_b, label_ids)

    sampler = None
    if random_sample:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    loader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return loader



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


