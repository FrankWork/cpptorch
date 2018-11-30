# encoding: utf-8

import numpy as np
import argparse
from network import LacNet, Config

parser = argparse.ArgumentParser("Run inference.")
parser.add_argument(
        '--numpy_arrays_path',
        type=str,
        default="lac_vars_map.npz",
        help='filename.npz'
        )
args = parser.parse_args()

cfg = Config()
model = LacNet(cfg)

for name, param in model.named_parameters():
    print(name, param.shape)

exit()

# word_emb.weight 				  word_emb	(20941, 128)
# gru_layers.0.fc.weight            [768, 128])	  fc_0.w_0	(128, 768)
# gru_layers.0.fc.bias         	    [768])	  fc_0.b_0	(768,)
# gru_layers.0.gru.weight_ih_l0     [768, 768])   gru_0.w_0     (256, 768)
# gru_layers.0.gru.weight_hh_l0     [768, 256])   gru_0.b_0     (1, 768)
# gru_layers.0.gru.bias_ih_l0       [768])
# gru_layers.0.gru.bias_hh_l0       [768])

# gru_layers.0.fc_r.weight          [768, 128])
# gru_layers.0.fc_r.bias            [768])
# gru_layers.0.gru_r.weight_ih_l0           [768, 768])
# gru_layers.0.gru_r.weight_hh_l0           [768, 256])
# gru_layers.0.gru_r.bias_ih_l0             [768])
# gru_layers.0.gru_r.bias_hh_l0             [768])

# gru_layers.1.fc.weight            [768, 128])
# gru_layers.1.fc.bias          [768])
# gru_layers.1.gru.weight_ih_l0             [768, 768])
# gru_layers.1.gru.weight_hh_l0             [768, 256])
# gru_layers.1.gru.bias_ih_l0           [768])
# gru_layers.1.gru.bias_hh_l0           [768])
# gru_layers.1.fc_r.weight          [768, 128])
# gru_layers.1.fc_r.bias            [768])
# gru_layers.1.gru_r.weight_ih_l0           [768, 768])
# gru_layers.1.gru_r.weight_hh_l0           [768, 256])
# gru_layers.1.gru_r.bias_ih_l0             [768])
# gru_layers.1.gru_r.bias_hh_l0             [768])
# 
# 
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
# emission.weight torch.Size([57, 512])
# emission.bias torch.Size([57])
# crf.transitions torch.Size([57, 57])
# crf._constraint_mask torch.Size([59, 59])
# crf.start_transitions torch.Size([57])
# crf.end_transitions torch.Size([57])


# crfw	(59, 57)
# restore `vars_map` of type `dict` from .npz file
data = np.load(args.numpy_arrays_path, encoding='bytes')
vars_map = data['vars_map'][()]

# for name in sorted(vars_map.keys()):
#    print('{0}\t{1}'.format(name, vars_map[name].shape))
#print(type(vars_map), vars_map.shape)




