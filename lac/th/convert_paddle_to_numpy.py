# encoding: utf-8

import numpy as np
import paddle.fluid as fluid
import paddle
import argparse
import os

parser = argparse.ArgumentParser("Run inference.")
parser.add_argument(
        '--paddle_model_dir',
        type=str,
        default='./baidu-lac-python/conf/model',
        help='A path to the model. (default: %(default)s)'
)
parser.add_argument(
        '--numpy_arrays_path',
        type=str,
        default="lac_vars_map.npz",
        help='filename.npz'
        )
args = parser.parse_args()

place = fluid.CPUPlace()
exe = fluid.Executor(place)

inference_scope = fluid.core.Scope()
with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
                        args.paddle_model_dir, exe)
        #print(inference_program)
        vars_map = {}
        for var in inference_program.list_vars():
                if hasattr(var, 'shape') and var.shape[0]!=-1:
                    (np_var,) = exe.run(fetch_list=[var])
                    vars_map[var.name] = np_var
        for name in sorted(vars_map.keys()):
                print('{0}\t{1}'.format(name, vars_map[name].shape))
        np.savez(args.numpy_arrays_path, vars_map=vars_map)
        print('save model to {}'.format(args.numpy_arrays_path))
        print('='*40)
        print(feed_target_names)
        print('='*40)
        for var in fetch_targets:
                print('{0}\t{1}'.format(var.name, var.shape))
        print('='*40)

# word_emb	(20941, 128)

# fc_0.w_0	(128, 768)
# fc_0.b_0	(768,)
# gru_0.w_0	(256, 768)
# gru_0.b_0	(1, 768)
# fc_1.w_0	(128, 768)
# fc_1.b_0	(768,)
# gru_1.w_0	(256, 768)
# gru_1.b_0	(1, 768)

# fc_2.w_0	(512, 768)
# fc_2.b_0	(768,)
# gru_2.w_0	(256, 768)
# gru_2.b_0	(1, 768)
# fc_3.w_0	(512, 768)
# fc_3.b_0	(768,)
# gru_3.w_0	(256, 768)
# gru_3.b_0	(1, 768)

# fc_4.w_0	(512, 57)
# fc_4.b_0	(57,)
# crfw	(59, 57)
# ========================================
# [u'word']
# ========================================
# crf_decoding_0.tmp_0	(-1L, 1L)
# ========================================
