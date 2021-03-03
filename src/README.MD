Instructions to generate kernel features from raw spikes

rodTap 20 4000
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 4000 --tool_type 20

rodTap 20 50
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 50 --tool_type 20

handover 0 4000
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 4000 --tool_type 0

foodPoking
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 4000 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0



### CHECK!
rodTap 20 4000
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 4000 --tool_type 20