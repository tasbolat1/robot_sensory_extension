from functools import partial

import sys

import numpy as np
import pandas as pd


def preprocess(dataset, time_past=0.05, time_future=0.25, time_interval=0.005, samples_past=100, samples_future=500):
    
    preprocessor_neutouch = {
        'neutouch_table'           : _preprocess_neutouch_table,
        'neutouch_tool_20'         : partial(_preprocess_neutouch_tool, 20),
        'neutouch_tool_30'         : partial(_preprocess_neutouch_tool, 30),
        'neutouch_tool_50'         : partial(_preprocess_neutouch_tool, 50),
        'neutouch_handover_rod'    : partial(_preprocess_neutouch_handover, 'rod'),
        'neutouch_handover_box'    : partial(_preprocess_neutouch_handover, 'box'),
        'neutouch_handover_plate'  : partial(_preprocess_neutouch_handover, 'plate'),
        'neutouch_food_apple'      : partial(_preprocess_neutouch_food, 'apple'),
        'neutouch_food_banana'     : partial(_preprocess_neutouch_food, 'banana'),
        'neutouch_food_empty'      : partial(_preprocess_neutouch_food, 'empty'),
        'neutouch_food_pepper'     : partial(_preprocess_neutouch_food, 'pepper'),
        'neutouch_food_tofu'       : partial(_preprocess_neutouch_food, 'tofu'),
        'neutouch_food_water'      : partial(_preprocess_neutouch_food, 'water'),
        'neutouch_food_watermelon' : partial(_preprocess_neutouch_food, 'watermelon')
    }
        
    preprocessor_biotac = {
        'biotac_table'             : _preprocess_biotac_table,
        'biotac_tool_20'           : partial(_preprocess_biotac_tool, 20),
        'biotac_tool_30'           : partial(_preprocess_biotac_tool, 30),
        'biotac_tool_50'           : partial(_preprocess_biotac_tool, 50),
        'biotac_handover_rod'      : partial(_preprocess_biotac_handover, 'rod'),
        'biotac_handover_box'      : partial(_preprocess_biotac_handover, 'box'),
        'biotac_handover_plate'    : partial(_preprocess_biotac_handover, 'plate'),
        'biotac_food_apple'        : partial(_preprocess_biotac_food, 'apple'),
        'biotac_food_banana'       : partial(_preprocess_biotac_food, 'banana'),
        'biotac_food_empty'        : partial(_preprocess_biotac_food, 'empty'),
        'biotac_food_pepper'       : partial(_preprocess_biotac_food, 'pepper'),
        'biotac_food_tofu'         : partial(_preprocess_biotac_food, 'tofu'),
        'biotac_food_water'        : partial(_preprocess_biotac_food, 'water'),
        'biotac_food_watermelon'   : partial(_preprocess_biotac_food, 'watermelon')
    }
    
    if dataset in preprocessor_neutouch:
        signals, labels = preprocessor_neutouch[dataset](time_past, time_future, time_interval)
        
    elif dataset in preprocessor_biotac:
        signals, labels = preprocessor_biotac[dataset](samples_past, samples_future)
        
    else: raise Exception('Dataset not found')
    
    print(signals.shape, labels.shape)
    np.savez(f'preprocessed/{dataset}.npz', signals=signals, labels=labels)
    print(f'Preprocessing for {dataset} completed with signals.shape = {signals.shape} and labels.shape = {labels.shape}')


#    _   _            _                   _       _____                                                       
#   | \ | |          | |                 | |     |  __ \                                                      
#   |  \| | ___ _   _| |_ ___  _   _  ___| |__   | |__) | __ ___ _ __  _ __ ___   ___ ___  ___ ___  ___  _ __ 
#   | . ` |/ _ \ | | | __/ _ \| | | |/ __| '_ \  |  ___/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __|/ _ \| '__|
#   | |\  |  __/ |_| | || (_) | |_| | (__| | | | | |   | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ (_) | |   
#   |_| \_|\___|\__,_|\__\___/ \__,_|\___|_| |_| |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/\___/|_|   
#                                                               | |                                           


def _preprocess_neutouch_tool(tool_length, time_past, time_future, time_interval):
    
    signals = None
    labels = None
        
    for trial in range(1, 11):
        
        print('Preprocessing trial {:02d} for neutouch_tool_{:02d}'.format(trial, tool_length))
        
        df_essentials = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/tool_neutouch_1k/trial{trial}_{tool_length}_essentials.csv')
        df_raw = _read_neutouch_raw(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/tool_neutouch_1k/trial{trial}_{tool_length}.tact')
        
        signals_temp = _bin_neutouch_signal(df_essentials.t.values, df_raw, time_past, time_future, time_interval)
        labels_temp = df_essentials.label_y.values
        
        signals = np.append(signals, signals_temp, axis=0) if signals is not None else signals_temp
        labels = np.append(labels, labels_temp, axis=0) if labels is not None else labels_temp
    
    return signals, labels


def _preprocess_neutouch_table(time_past=0.05, time_future=0.25, time_interval=0.005):
    
    signals = None
    labels = None
        
    for trial in range(1, 31):
        
        print('Preprocessing trial {:02d} for neutouch_table'.format(trial))
        
        df_essentials = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/table_1k/trial{trial}.csv')
        df_raw = _read_neutouch_raw(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/table_1k/trial{trial}.tact')
        
        signals_temp = _bin_neutouch_signal(df_essentials.t.values, df_raw, time_past, time_future, time_interval)
        labels_temp = df_essentials[['label_x', 'label_z', 'd_NL', 'd_NR']].values
        
        signals = np.append(signals, signals_temp, axis=0) if signals is not None else signals_temp
        labels = np.append(labels, labels_temp, axis=0) if labels is not None else labels_temp
    
    return signals, labels


def _preprocess_neutouch_handover(item, time_past=0.05, time_future=0.25, time_interval=0.005):
    
    signals = None
    labels = None
    
    df_essentials = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/neutouch_handover/nt_essentials.csv')
    df_essentials = df_essentials[df_essentials.obj == item]
    
    for _, row in df_essentials.iterrows():
        
        print(f'Preprocessing {row.fname} for neutouch_handover_{row.obj}')
        
        df_raw = _read_neutouch_raw(f'/home/shuisong/main_ws/sensory_ext/data/neutouch/neutouch_handover/{row.fname}.tact')
        tap_time = row.tapped_time
        
        signals_temp = _bin_neutouch_signal(np.array([tap_time]), df_raw, time_past, time_future, time_interval)
        labels_temp = row[['isPos', 'label_x_thumb', 'label_y_thumb', 'label_z_thumb', 'label_x_thumb_d', 'label_y_thumb_d', 'label_z_thumb_d', 'label_x_index', 'label_y_index', 'label_z_index', 'label_x_index_d', 'label_y_index_d', 'label_z_index_d']].values
        
        signals = np.vstack((signals, signals_temp)) if signals is not None else signals_temp
        labels = np.vstack((labels, labels_temp)) if labels is not None else labels_temp
    
    return signals, labels


def _preprocess_neutouch_food(item, time_past=0.05, time_future=0.25, time_interval=0.005):
    
    pass


def _read_neutouch_raw(filepath):
    
    df = pd.read_csv(filepath,
                     names=['isPos', 'taxel', 'removable', 't'],
                     dtype={'isPos': int , 'taxel': int, 'removable': int, 't': float},
                     sep=' ')
    
    df.drop(['removable'], axis=1, inplace=True)
    df.drop(df.tail(1).index, inplace=True)
    df.drop(df.head(1).index, inplace=True)
    
    return df.reset_index(drop=True)


def _bin_neutouch_signal(tap_times, df_raw, time_past, time_future, time_interval):

    n_bins = int((time_past + time_future) / time_interval) + 1
    signals = np.zeros([len(tap_times), 80, n_bins], dtype=int)

    for i, tap_time in enumerate(tap_times):

        df_timespan = df_raw[(df_raw.t >= (tap_time - time_past)) & (df_raw.t < (tap_time + time_future))]
        df_positive = df_timespan[df_timespan.isPos == 1]
        df_negative = df_timespan[df_timespan.isPos == 0]

        t = tap_time - time_past
        k = 0

        while t < (tap_time + time_future):
            
            positive_taxels = df_positive[((df_positive.t >= t) & (df_positive.t < t + time_interval))].taxel
            if len(positive_taxels):
                for taxel in positive_taxels:
                    signals[i, taxel - 1, k] += 1
                    
            negative_taxels = df_negative[((df_negative.t >= t) & (df_negative.t < t + time_interval))].taxel
            if len(negative_taxels):
                for taxel in negative_taxels:
                    signals[i, taxel - 1, k] -= 1
                    
            t += time_interval
            k += 1
    
    return signals      


#    ____  _       _               _____                                                       
#   |  _ \(_)     | |             |  __ \                                                      
#   | |_) |_  ___ | |_ __ _  ___  | |__) | __ ___ _ __  _ __ ___   ___ ___  ___ ___  ___  _ __ 
#   |  _ <| |/ _ \| __/ _` |/ __| |  ___/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __|/ _ \| '__|
#   | |_) | | (_) | || (_| | (__  | |   | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ (_) | |   
#   |____/|_|\___/ \__\__,_|\___| |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/\___/|_|   
#                                                | |                                           


def _preprocess_biotac_tool(tool_length, samples_past, samples_future):
    
    signals = None
    labels = None
        
    for trial in range(1, 21):
        
        print('Preprocessing trial {:02d} for biotac_tool_{:02d}'.format(trial, tool_length))
        
        df_essentials = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/biotac/tool_1k/trial{trial}_{tool_length}_essentials.csv')
        df_raw = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/biotac/tool_1k/trial{trial}_{tool_length}.csv')
        
        signals_trial = _crop_biotac_signal(df_essentials.orignal_index.values, df_raw, samples_past, samples_future)
        labels_trial = df_essentials.label_y.values
        
        signals = np.append(signals, signals_trial, axis=0) if signals is not None else signals_trial
        labels = np.append(labels, labels_trial, axis=0) if labels is not None else labels_trial
    
    return signals, labels


def _preprocess_biotac_table(samples_past, samples_future):
    
    signals = None
    labels = None
        
    for trial in range(1, 21):
        
        print('Preprocessing trial {:02d} for biotac_table'.format(trial))
        
        df_essentials = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/biotac/table_1k/essentials/trial{trial}_essentials.csv')
        df_raw = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/biotac/table_1k/essentials/trial{trial}.csv')
        
        signals_trial = _crop_biotac_signal(df_essentials.original_index.values, df_raw, samples_past, samples_future)
        labels_trial = df_essentials[['label_x', 'label_z', 'd_B']].values
        
        signals = np.append(signals, signals_trial, axis=0) if signals is not None else signals_trial
        labels = np.append(labels, labels_trial, axis=0) if labels is not None else labels_trial
    
    return signals, labels


def _preprocess_biotac_handover(item, samples_past, samples_future):
    
    signals = None
    labels = None
    
    df_essentials = pd.read_csv('/home/shuisong/main_ws/sensory_ext/data/biotac/biotac_handover/bt_essentials.csv')
    df_essentials = df_essentials[df_essentials.obj == item]
    print(len(df_essentials))
    
    for _, row in df_essentials.iterrows():
        
        print(f'Preprocessing {row.fname} for biotac_handover_{row.obj}')
        
        df_raw = pd.read_csv(f'/home/shuisong/main_ws/sensory_ext/data/biotac/biotac_handover/{row.fname}.csv')
        tap_index = np.abs(df_raw.t - row.tapped_time).argmin()
        
        signals_temp = _crop_biotac_signal(np.array([tap_index]), df_raw, samples_past, samples_future)
        labels_temp = row[['isPos', 'label_x_thumb', 'label_y_thumb', 'label_z_thumb', 'label_x_thumb_d', 'label_y_thumb_d', 'label_z_thumb_d', 'label_x_index', 'label_y_index', 'label_z_index', 'label_x_index_d', 'label_y_index_d', 'label_z_index_d']].values
        
        signals = np.vstack((signals, signals_temp)) if signals is not None else signals_temp
        labels = np.vstack((labels, labels_temp)) if labels is not None else labels_temp
    
    return signals, labels


def _preprocess_biotac_food(item, samples_past, samples_future):
    
    pass


def _crop_biotac_signal(tap_indices, df_raw, samples_past, samples_future):
    
    signals = np.zeros((len(tap_indices), samples_past + samples_future))
    
    for i, tap_index in enumerate(tap_indices):
        signals[i] = (df_raw.iloc[tap_index-100:tap_index+500].pac.values)
    
    return signals


if __name__ == '__main__':
    preprocess(sys.argv[1])
