# encoding: utf-8

import numpy as np
import argparse

parser = argparse.ArgumentParser("Run inference.")
parser.add_argument(
        '--numpy_arrays_path',
        type=str,
        default="lac_vars_map.npz",
        help='filename.npz'
        )
args = parser.parse_args()



# restore `vars_map` of type `dict` from .npz file
data = np.load(args.numpy_arrays_path, encoding='bytes')
vars_map = data['vars_map'][()]

for name in sorted(vars_map.keys()):
    print('{0}\t{1}'.format(name, vars_map[name].shape))
#print(type(vars_map), vars_map.shape)


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
