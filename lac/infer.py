#! encoding: utf-8

import numpy as np
import paddle.fluid as fluid
import paddle
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser("Run inference.")
    parser.add_argument(
            '--batch_size', type=int,  default=5,
                    help='The size of a batch. (default: %(default)d)')
    parser.add_argument(
            '--model_dir', type=str,  default='conf', help='A path to the model dir')
    parser.add_argument(
            '--input_path', type=str,  default=None, help='dir or file')
    args = parser.parse_args()
    return args

   
def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print(('%s: %s' % (arg, value)))
    print('------------------------------------------------')


def get_real_tag(origin_tag):
    if origin_tag == "O":
        return "O"
    return origin_tag[0:len(origin_tag) - 2]

def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

def load_dict(dict_path):
    """
    Load a dict. The first column is the key and the second column is the value.
    """
    result_dict = {}
    for line in open(dict_path, "r"):
        terms = line.strip("\n").split("\t")
        if len(terms) != 2:
            continue
        result_dict[terms[0].decode("utf-8")] = terms[1].decode("utf-8")
    return result_dict


def load_reverse_dict(dict_path):
    """
    Load a dict. The first column is the value and the second column is the key.
    """
    result_dict = {}
    for line in open(dict_path, "r"):
        terms = line.strip("\n").split("\t")
        if len(terms) != 2:
            continue
        result_dict[terms[1].decode("utf-8")] = terms[0].decode("utf-8")
    return result_dict

def test_reader(input_path,
                word2id_dict,
                label2id_dict,
                word_replace_dict,
                filename_feature=""):
    """
    define the reader to read test files in file_dir
    """
    def read_file(file_path):
        for line in open(file_path, 'r'):
            line = line.strip("\n")
            if len(line) == 0:
                # yield [], [] # paddle will break
                continue
            word_part = line
            word_idx = []
            words = word_part.decode("utf-8")
            for word in words:
                if ord(word) < 0x20:
                    word = ' '.decode("utf-8")
                if word in word_replace_dict:
                    word = word_replace_dict[word]
                if word in word2id_dict:
                    word_idx.append(int(word2id_dict[word]))
                else:
                    word_idx.append(int(word2id_dict["OOV"]))
            yield word_idx, words

    def reader():
        if os.path.isdir(input_path):
            for root, dirs, files in os.walk(input_path):
                for filename in files:
                    if not filename.startswith(filename_feature):
                        continue
                    else:
                        return read_file(os.path.join(root, filename))
        else:
            return read_file(input_path)

    return reader


def infer(args):
    word_dict_path = os.path.join(args.model_dir, "word.dic")
    label_dict_path = os.path.join(args.model_dir, "tag.dic")
    word_rep_dict_path = os.path.join(args.model_dir, 'q2b.dic')
    paddle_model_path = os.path.join(args.model_dir, "model")
    

    id2word_dict = load_dict(word_dict_path)
    word2id_dict = load_reverse_dict(word_dict_path) 

    id2label_dict = load_dict(label_dict_path)
    label2id_dict = load_reverse_dict(label_dict_path)
    q2b_dict = load_dict(word_rep_dict_path) # 全角转半角

    test_data = paddle.batch(
                    test_reader(args.input_path,
                        word2id_dict,
                        label2id_dict,
                        q2b_dict),
                    batch_size = args.batch_size,
                    drop_last=False)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(paddle_model_path, exe)
        for data in test_data():
            full_out_str = ""
            word_idx = to_lodtensor([x[0] for x in data], place)
            word_list = [x[1] for x in data]
            (crf_decode, ) = exe.run(inference_program,
                                 feed={"word":word_idx},
                                 fetch_list=fetch_targets,
                                 return_numpy=False)
            lod_info = (crf_decode.lod())[0]
            np_data = np.array(crf_decode)
            assert len(data) == len(lod_info) - 1
            for sen_index in range(len(data)):
                assert len(data[sen_index][0]) == lod_info[
                    sen_index + 1] - lod_info[sen_index]
                word_index = 0
                outstr = ""
                cur_full_word = ""
                cur_full_tag = ""
                words = word_list[sen_index]
                for tag_index in range(lod_info[sen_index],
                                        lod_info[sen_index + 1]):
                    cur_word = words[word_index]
                    cur_tag = id2label_dict[str(np_data[tag_index][0])]
                    if cur_tag.endswith("-B") or cur_tag.endswith("O"):
                        if len(cur_full_word) != 0:
                            #outstr += cur_full_word.encode('utf8') + "/" + cur_full_tag.encode('utf8') + " "
                            outstr += cur_full_word.encode('utf8') + " "
                        cur_full_word = cur_word
                        cur_full_tag = get_real_tag(cur_tag)
                    else:
                        cur_full_word += cur_word
                    word_index += 1
                # outstr += cur_full_word.encode('utf8') + "/" + cur_full_tag.encode('utf8') + " "    
                outstr += cur_full_word.encode('utf8') + " "    
                outstr = outstr.strip()
                full_out_str += outstr + "\n"
            print(full_out_str.strip())

if __name__ == "__main__":
    args = parse_args()
    infer(args)
