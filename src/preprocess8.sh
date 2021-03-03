
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 250 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0 --device 'cuda:2'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 500 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0 --device 'cuda:2'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 1000 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0 --device 'cuda:2'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 2000 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0 --device 'cuda:2'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 3000 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0 --device 'cuda:2'

