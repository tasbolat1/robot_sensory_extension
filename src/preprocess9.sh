#!/bin/bash
##### Downsample 30

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 2000 --tool_type 30 --device 'cuda:1' &

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 3000 --tool_type 30 --device 'cuda:1' && fg