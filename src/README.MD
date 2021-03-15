Instructions to generate kernel features from raw spikes

There are 3 tasks (task):(tool_type)

1. Tapping on rod (rodTap):(20,30,50)
2. Handover (handover):(0,1,2)
3. Food Poking (foodPoking):(-1)

The data can be sampled at different frequencies:
freq = {4000, 3000, 2000, 1000, 500, 200, 100, 50, 10, 5}


python kernel_preprocess.py --data_dir '{data_dir}' --save_dir_raw_spikes '../data/raw_spikes/' --save_dir_kernel '../data/kernel_features/' --task {task} --frequency {freq} --tool_type {tool_type}

