Instructions to generate kernel features from raw spikes


##### ROD

rodTap 20 4000
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 4000 --tool_type 20

rodTap 30 4000
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 4000 --tool_type 30

rodTap 50 4000
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 4000 --tool_type 50

##### Downsample 20
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 5 --tool_type 20

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 50 --tool_type 20

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 100 --tool_type 20

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 250 --tool_type 20

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 500 --tool_type 20

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 1000 --tool_type 20

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 2000 --tool_type 20

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 3000 --tool_type 20


##### Downsample 30
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 5 --tool_type 30 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 50 --tool_type 30 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 100 --tool_type 30 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 250 --tool_type 30 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 500 --tool_type 30 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 1000 --tool_type 30 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 2000 --tool_type 30 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 3000 --tool_type 30 --device 'cuda:1'

##### Downsample 50
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 5 --tool_type 50 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 50 --tool_type 50 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 100 --tool_type 50 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 250 --tool_type 50 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 500 --tool_type 50 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 1000 --tool_type 50 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 2000 --tool_type 50 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/tool_neutouch_1k/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task rodTap --frequency 3000 --tool_type 50 --device 'cuda:1'


### Handover

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 4000 --tool_type 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 4000 --tool_type 1

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 4000 --tool_type 2

#### downsample handover 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 5 --tool_type 0 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 50 --tool_type 0 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 100 --tool_type 0 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 250 --tool_type 0 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 500 --tool_type 0 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 1000 --tool_type 0 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 2000 --tool_type 0 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 3000 --tool_type 0 --device 'cuda:1'


#### downsample handover 1

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 5 --tool_type 1 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 50 --tool_type 1 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 100 --tool_type 1 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 250 --tool_type 1 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 500 --tool_type 1 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 1000 --tool_type 1 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 2000 --tool_type 1 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 3000 --tool_type 1 --device 'cuda:1'

#### downsample handover 3
python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 5 --tool_type 2 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 50 --tool_type 2 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 100 --tool_type 2 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 250 --tool_type 2 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 500 --tool_type 2 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 1000 --tool_type 2 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 2000 --tool_type 2 --device 'cuda:1'

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/neutouch_handover/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task handover --frequency 3000 --tool_type 2 --device 'cuda:1'


### foodPoking

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 4000 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 5 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 50 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 100 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 250 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 100 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 250 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 500 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 1000 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 2000 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0

python kernel_preprocess.py --data_dir '/datasets/sensory_ext/data/neutouch/food/' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task foodPoking --frequency 3000 --tool_type -1 --FUTURE_T 6000 --DELAY_T 0
