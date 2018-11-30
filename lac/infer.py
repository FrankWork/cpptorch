# encoding: utf-8

import numpy as np
# import paddle.fluid as fluid
# import paddle
import argparse
import os
from network import LacNet, Config


def parse_args():
    parser = argparse.ArgumentParser("Run inference.")
    parser.add_argument(
            '--batch_size', type=int,  default=5,
                    help='The size of a batch. (default: %(default)d)')
    parser.add_argument(
            '--paddle_model_dir', type=str,  default='./baidu-lac-python/conf',
                    help='A path to the model. (default: %(default)s)')
    parser.add_argument(
            '--test_data_dir', type=str,  default='./data/test_data',
                    help='A directory with test data files. (default: %(default)s)')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    print(args.batch_size)
    print(args.paddle_model_dir)
    print(args.test_data_dir)
    print('------------------------------------------------')


def load_dict(dict_path):
    """
    Load a dict. The first column is the key and 
    the second column is the value.
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
    Load a dict. The first column is the value and 
    the second column is the key.
    """
    result_dict = {}
    for line in open(dict_path, "r"):
        terms = line.strip("\n").split("\t")
        if len(terms) != 2:
            continue
        result_dict[terms[1].decode("utf-8")] = terms[0].decode("utf-8")
    return result_dict

def get_real_tag(origin_tag):
    if origin_tag == "O":
        return "O"
    return origin_tag[0:len(origin_tag) - 2]


def infer(args):
    word_dict_path = os.path.join(args.paddle_model_dir, "word.dic")
    label_dict_path = os.path.join(args.paddle_model_dir, "tag.dic")
    # 全角转半角符号
    word_rep_dict_path = os.path.join(args.paddle_model_dir, 'q2b.dic')
    paddle_model_path = os.path.join(args.paddle_model_dir, "model")
    
    cfg = Config()
    model = LacNet(cfg)

    exit()

    # id2word_dict = load_dict(word_dict_path)
    # word2id_dict = load_reverse_dict(word_dict_path) 

    # id2label_dict = load_dict(label_dict_path)
    # label2id_dict = load_reverse_dict(label_dict_path)
    # 
    # q2b_dict = load_dict(word_rep_dict_path)


    # test_data = paddle.batch(
    #				 reader.test_reader(args.test_data_dir,
    #					 word2id_dict,
    #					 label2id_dict,
    #					 q2b_dict),
    #				 batch_size = args.batch_size)

    # place = fluid.CPUPlace()
    # exe = fluid.Executor(place)

    # inference_scope = fluid.core.Scope()
    # with fluid.scope_guard(inference_scope):
    #     [inference_program, feed_target_names,
    #      fetch_targets] = fluid.io.load_inference_model(
    #                     paddle_model_path, exe)
    #     #print(inference_program)
    #     vars_map = {}
    #     for var in inference_program.list_vars():
    #         if hasattr(var, 'shape') and var.shape[0]!=-1:
    #             (np_var,) = exe.run(fetch_list=[var])
    #             vars_map[var.name] = np_var
    #     for name in sorted(vars_map.keys()):
    #         print('{0}\t{1}'.format(name, vars_map[name].shape))
    #     print('='*40)
    #     print(feed_target_names)
    #     print('='*40)
    #     for var in fetch_targets:
    #         print('{0}\t{1}'.format(var.name, var.shape))
    #     print('='*40)
    #     exit()

    #     for data in test_data():
    #         full_out_str = ""
    #         word_idx = to_lodtensor([x[0] for x in data], place)
    #         word_list = [x[1] for x in data]
    #         (crf_decode, ) = exe.run(inference_program,
    #                                                  feed={"word":word_idx},
    #                                                  fetch_list=fetch_targets,
    #                                                  return_numpy=False)
    #         lod_info = (crf_decode.lod())[0]
    #         np_data = np.array(crf_decode)
    #         assert len(data) == len(lod_info) - 1
    #         for sen_index in xrange(len(data)):
    #             assert len(data[sen_index][0]) == lod_info[
    #                     sen_index + 1] - lod_info[sen_index]
    #             word_index = 0
    #             outstr = ""
    #             cur_full_word = ""
    #             cur_full_tag = ""
    #             words = word_list[sen_index]
    #             for tag_index in xrange(lod_info[sen_index],
    #                                                             lod_info[sen_index + 1]):
    #                     cur_word = words[word_index]
    #                     cur_tag = id2label_dict[str(np_data[tag_index][0])]
    #                     if cur_tag.endswith("-B") or cur_tag.endswith("O"):
    #                         if len(cur_full_word) != 0:
    #                                 outstr += cur_full_word.encode('utf8') + "/" + cur_full_tag.encode('utf8') + " "
    #                         cur_full_word = cur_word
    #                         cur_full_tag = get_real_tag(cur_tag)
    #                     else:
    #                         cur_full_word += cur_word
    #                     word_index += 1
    #             outstr += cur_full_word.encode('utf8') + "/" + cur_full_tag.encode('utf8') + " "	
    #             outstr = outstr.strip()
    #             full_out_str += outstr + "\n"
    #         print full_out_str.strip()

if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    infer(args)
# crfw	(59L, 57L)
# fc_0.b_0	(768L,)
# fc_0.w_0	(128L, 768L)
# fc_1.b_0	(768L,)
# fc_1.w_0	(128L, 768L)
# fc_2.b_0	(768L,)
# fc_2.w_0	(512L, 768L)
# fc_3.b_0	(768L,)
# fc_3.w_0	(512L, 768L)
# fc_4.b_0	(57L,)
# fc_4.w_0	(512L, 57L)
# gru_0.b_0	(1L, 768L)
# gru_0.w_0	(256L, 768L)
# gru_1.b_0	(1L, 768L)
# gru_1.w_0	(256L, 768L)
# gru_2.b_0	(1L, 768L)
# gru_2.w_0	(256L, 768L)
# gru_3.b_0	(1L, 768L)
# gru_3.w_0	(256L, 768L)
# word_emb	(20941L, 128L)
# ========================================
# [u'word']
# ========================================
# crf_decoding_0.tmp_0	(-1L, 1L)
# ========================================
