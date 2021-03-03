from functools import partial
from sklearn.model_selection import train_test_split, GridSearchCV

import numpy as np


def evaluate(experiment_name, device='cuda'):
    
    experiments = {
        
        'neutouch_tool_20_svm_linear'               : partial(_evaluate_neutouch_tool_svm, 20, 'linear', False),
        'neutouch_tool_30_svm_linear'               : partial(_evaluate_neutouch_tool_svm, 30, 'linear', False),
        'neutouch_tool_50_svm_linear'               : partial(_evaluate_neutouch_tool_svm, 50, 'linear', False),
        
        'neutouch_tool_20_svm_rbf'                  : partial(_evaluate_neutouch_tool_svm, 20, 'rbf', False),
        'neutouch_tool_30_svm_rbf'                  : partial(_evaluate_neutouch_tool_svm, 30, 'rbf', False),
        'neutouch_tool_50_svm_rbf'                  : partial(_evaluate_neutouch_tool_svm, 50, 'rbf', False),
        
        'neutouch_tool_20_fftsvm_linear'            : partial(_evaluate_neutouch_tool_svm, 20, 'linear', True),
        'neutouch_tool_30_fftsvm_linear'            : partial(_evaluate_neutouch_tool_svm, 30, 'linear', True),
        'neutouch_tool_50_fftsvm_linear'            : partial(_evaluate_neutouch_tool_svm, 50, 'linear', True),
        
        'neutouch_tool_20_fftsvm_rbf'               : partial(_evaluate_neutouch_tool_svm, 20, 'rbf', True),
        'neutouch_tool_30_fftsvm_rbf'               : partial(_evaluate_neutouch_tool_svm, 30, 'rbf', True),
        'neutouch_tool_50_fftsvm_rbf'               : partial(_evaluate_neutouch_tool_svm, 50, 'rbf', True),
        
        'neutouch_tool_20_lstm'                     : partial(_evaluate_neutouch_tool_lstm, 20, device),
        'neutouch_tool_30_lstm'                     : partial(_evaluate_neutouch_tool_lstm, 30, device),
        'neutouch_tool_50_lstm'                     : partial(_evaluate_neutouch_tool_lstm, 50, device),
        
        'neutouch_tool_autoencoder'                 : partial(_evaluate_neutouch_tool_autoencoder, device),
        
        'neutouch_tool_20_autoencoder_svm_linear'   : partial(_evaluate_neutouch_tool_autoencoder_svm, 20, 'linear', device),
        'neutouch_tool_30_autoencoder_svm_linear'   : partial(_evaluate_neutouch_tool_autoencoder_svm, 30, 'linear', device),
        'neutouch_tool_50_autoencoder_svm_linear'   : partial(_evaluate_neutouch_tool_autoencoder_svm, 50, 'linear', device),
        
        'neutouch_tool_20_autoencoder_svm_rbf'      : partial(_evaluate_neutouch_tool_autoencoder_svm, 20, 'rbf', device),
        'neutouch_tool_30_autoencoder_svm_rbf'      : partial(_evaluate_neutouch_tool_autoencoder_svm, 30, 'rbf', device),
        'neutouch_tool_50_autoencoder_svm_rbf'      : partial(_evaluate_neutouch_tool_autoencoder_svm, 50, 'rbf', device),
        
        'neutouch_tool_20_autoencoder_mlp'          : partial(_evaluate_neutouch_tool_autoencoder_mlp, 20, device),
        'neutouch_tool_30_autoencoder_mlp'          : partial(_evaluate_neutouch_tool_autoencoder_mlp, 30, device),
        'neutouch_tool_50_autoencoder_mlp'          : partial(_evaluate_neutouch_tool_autoencoder_mlp, 50, device),
        
        'neutouch_halftool_20_svm_linear'           : partial(_evaluate_neutouch_halftool_svm, 20, 'linear', False),
        'neutouch_halftool_30_svm_linear'           : partial(_evaluate_neutouch_halftool_svm, 30, 'linear', False),
        'neutouch_halftool_50_svm_linear'           : partial(_evaluate_neutouch_halftool_svm, 50, 'linear', False),
        
        'neutouch_halftool_20_svm_rbf'              : partial(_evaluate_neutouch_halftool_svm, 20, 'rbf', False),
        'neutouch_halftool_30_svm_rbf'              : partial(_evaluate_neutouch_halftool_svm, 30, 'rbf', False),
        'neutouch_halftool_50_svm_rbf'              : partial(_evaluate_neutouch_halftool_svm, 50, 'rbf', False),
        
        'neutouch_halftool_20_fftsvm_linear'        : partial(_evaluate_neutouch_halftool_svm, 20, 'linear', True),
        'neutouch_halftool_30_fftsvm_linear'        : partial(_evaluate_neutouch_halftool_svm, 30, 'linear', True),
        'neutouch_halftool_50_fftsvm_linear'        : partial(_evaluate_neutouch_halftool_svm, 50, 'linear', True),
        
        'neutouch_halftool_20_fftsvm_rbf'           : partial(_evaluate_neutouch_halftool_svm, 20, 'rbf', True),
        'neutouch_halftool_30_fftsvm_rbf'           : partial(_evaluate_neutouch_halftool_svm, 30, 'rbf', True),
        'neutouch_halftool_50_fftsvm_rbf'           : partial(_evaluate_neutouch_halftool_svm, 50, 'rbf', True),
        
        'neutouch_halftool_20_lstm'                 : partial(_evaluate_neutouch_halftool_lstm, 20, device),
        'neutouch_halftool_30_lstm'                 : partial(_evaluate_neutouch_halftool_lstm, 30, device),
        'neutouch_halftool_50_lstm'                 : partial(_evaluate_neutouch_halftool_lstm, 50, device),
        
        'neutouch_singletool_20'                    : partial(_evaluate_neutouch_singletool, 20),
        'neutouch_singletool_30'                    : partial(_evaluate_neutouch_singletool, 30),
        'neutouch_singletool_50'                    : partial(_evaluate_neutouch_singletool, 50),
        
        'neutouch_handover_rod_svm_linear'          : partial(_evaluate_neutouch_handover_svm, 'rod', 'linear', False),
        'neutouch_handover_box_svm_linear'          : partial(_evaluate_neutouch_handover_svm, 'box', 'linear', False),
        'neutouch_handover_plate_svm_linear'        : partial(_evaluate_neutouch_handover_svm, 'plate', 'linear', False),
        
        'neutouch_handover_rod_svm_rbf'             : partial(_evaluate_neutouch_handover_svm, 'rod', 'rbf', False),
        'neutouch_handover_box_svm_rbf'             : partial(_evaluate_neutouch_handover_svm, 'box', 'rbf', False),
        'neutouch_handover_plate_svm_rbf'           : partial(_evaluate_neutouch_handover_svm, 'plate', 'rbf', False),
        
        'neutouch_handover_rod_fftsvm_linear'       : partial(_evaluate_neutouch_handover_svm, 'rod', 'linear', True),
        'neutouch_handover_box_fftsvm_linear'       : partial(_evaluate_neutouch_handover_svm, 'box', 'linear', True),
        'neutouch_handover_plate_fftsvm_linear'     : partial(_evaluate_neutouch_handover_svm, 'plate', 'linear', True),
        
        'neutouch_handover_rod_fftsvm_rbf'          : partial(_evaluate_neutouch_handover_svm, 'rod', 'rbf', True),
        'neutouch_handover_box_fftsvm_rbf'          : partial(_evaluate_neutouch_handover_svm, 'box', 'rbf', True),
        'neutouch_handover_plate_fftsvm_rbf'        : partial(_evaluate_neutouch_handover_svm, 'plate', 'rbf', True),
        
        'neutouch_handover_rod_lstm'                : partial(_evaluate_neutouch_handover_lstm, 'rod', device),
        'neutouch_handover_box_lstm'                : partial(_evaluate_neutouch_handover_lstm, 'box', device),
        'neutouch_handover_plate_lstm'              : partial(_evaluate_neutouch_handover_lstm, 'plate', device),
        
        'neutouch_handover_rod_mlp'                 : partial(_evaluate_neutouch_handover_mlp, 'rod', device, False),
        'neutouch_handover_box_mlp'                 : partial(_evaluate_neutouch_handover_mlp, 'box', device, False),
        'neutouch_handover_plate_mlp'               : partial(_evaluate_neutouch_handover_mlp, 'plate', device, False),
        
        'neutouch_handover_rod_fftmlp'              : partial(_evaluate_neutouch_handover_mlp, 'rod', device, True),
        'neutouch_handover_box_fftmlp'              : partial(_evaluate_neutouch_handover_mlp, 'box', device, True),
        'neutouch_handover_plate_fftmlp'            : partial(_evaluate_neutouch_handover_mlp, 'plate', device, True),
        
        'neutouch_halfhandover_rod_svm_linear'      : partial(_evaluate_neutouch_halfhandover_svm, 'rod', 'linear', False),
        'neutouch_halfhandover_box_svm_linear'      : partial(_evaluate_neutouch_halfhandover_svm, 'box', 'linear', False),
        'neutouch_halfhandover_plate_svm_linear'    : partial(_evaluate_neutouch_halfhandover_svm, 'plate', 'linear', False),
        
        'neutouch_halfhandover_rod_svm_rbf'         : partial(_evaluate_neutouch_halfhandover_svm, 'rod', 'rbf', False),
        'neutouch_halfhandover_box_svm_rbf'         : partial(_evaluate_neutouch_halfhandover_svm, 'box', 'rbf', False),
        'neutouch_halfhandover_plate_svm_rbf'       : partial(_evaluate_neutouch_halfhandover_svm, 'plate', 'rbf', False),
        
        'neutouch_halfhandover_rod_fftsvm_linear'   : partial(_evaluate_neutouch_halfhandover_svm, 'rod', 'linear', True),
        'neutouch_halfhandover_box_fftsvm_linear'   : partial(_evaluate_neutouch_halfhandover_svm, 'box', 'linear', True),
        'neutouch_halfhandover_plate_fftsvm_linear' : partial(_evaluate_neutouch_halfhandover_svm, 'plate', 'linear', True),
        
        'neutouch_halfhandover_rod_fftsvm_rbf'      : partial(_evaluate_neutouch_halfhandover_svm, 'rod', 'rbf', True),
        'neutouch_halfhandover_box_fftsvm_rbf'      : partial(_evaluate_neutouch_halfhandover_svm, 'box', 'rbf', True),
        'neutouch_halfhandover_plate_fftsvm_rbf'    : partial(_evaluate_neutouch_halfhandover_svm, 'plate', 'rbf', True),
        
        'neutouch_food_svm_linear'                  : partial(_evaluate_neutouch_food_svm, 'linear', False),
        'neutouch_food_svm_rbf'                     : partial(_evaluate_neutouch_food_svm, 'rbf', False),
        'neutouch_food_fftsvm_linear'               : partial(_evaluate_neutouch_food_svm, 'linear', True),
        'neutouch_food_fftsvm_rbf'                  : partial(_evaluate_neutouch_food_svm, 'rbf', True),
        'neutouch_food_lstm'                        : partial(_evaluate_neutouch_food_lstm, device),
        'neutouch_food_mlp'                         : partial(_evaluate_neutouch_food_mlp, device, False),
        'neutouch_food_fftmlp'                      : partial(_evaluate_neutouch_food_mlp, device, True),
        
        'neutouch_halffood_svm_linear'              : partial(_evaluate_neutouch_halffood_svm, 'linear', False),
        'neutouch_halffood_svm_rbf'                 : partial(_evaluate_neutouch_halffood_svm, 'rbf', False),
        'neutouch_halffood_fftsvm_linear'           : partial(_evaluate_neutouch_halffood_svm, 'linear', True),
        'neutouch_halffood_fftsvm_rbf'              : partial(_evaluate_neutouch_halffood_svm, 'rbf', True),
        'neutouch_halffood_lstm'                    : partial(_evaluate_neutouch_halffood_lstm, device),
        'neutouch_halffood_mlp'                     : partial(_evaluate_neutouch_halffood_mlp, device, False),
        'neutouch_halffood_fftmlp'                  : partial(_evaluate_neutouch_halffood_mlp, device, True),
        
        'biotac_tool_20_svm_linear'                 : partial(_evaluate_biotac_tool_svm, 20, 'linear', False),
        'biotac_tool_30_svm_linear'                 : partial(_evaluate_biotac_tool_svm, 30, 'linear', False),
        'biotac_tool_50_svm_linear'                 : partial(_evaluate_biotac_tool_svm, 50, 'linear', False),
        
        'biotac_tool_20_svm_rbf'                    : partial(_evaluate_biotac_tool_svm, 20, 'rbf', False),
        'biotac_tool_30_svm_rbf'                    : partial(_evaluate_biotac_tool_svm, 30, 'rbf', False),
        'biotac_tool_50_svm_rbf'                    : partial(_evaluate_biotac_tool_svm, 50, 'rbf', False),

        'biotac_tool_20_fftsvm_linear'              : partial(_evaluate_biotac_tool_svm, 20, 'linear', True),
        'biotac_tool_30_fftsvm_linear'              : partial(_evaluate_biotac_tool_svm, 30, 'linear', True),
        'biotac_tool_50_fftsvm_linear'              : partial(_evaluate_biotac_tool_svm, 50, 'linear', True),
        
        'biotac_tool_20_fftsvm_rbf'                 : partial(_evaluate_biotac_tool_svm, 20, 'rbf', True),
        'biotac_tool_30_fftsvm_rbf'                 : partial(_evaluate_biotac_tool_svm, 30, 'rbf', True),
        'biotac_tool_50_fftsvm_rbf'                 : partial(_evaluate_biotac_tool_svm, 50, 'rbf', True),
        
        'biotac_tool_20_lstm'                       : partial(_evaluate_biotac_tool_lstm, 20, device),
        'biotac_tool_30_lstm'                       : partial(_evaluate_biotac_tool_lstm, 30, device),
        'biotac_tool_50_lstm'                       : partial(_evaluate_biotac_tool_lstm, 50, device),
        
        'biotac_tool_autoencoder'                   : partial(_evaluate_biotac_tool_autoencoder, device),
        
        'biotac_tool_20_autoencoder_svm_linear'     : partial(_evaluate_biotac_tool_autoencoder_svm, 20, 'linear', device),
        'biotac_tool_30_autoencoder_svm_linear'     : partial(_evaluate_biotac_tool_autoencoder_svm, 30, 'linear', device),
        'biotac_tool_50_autoencoder_svm_linear'     : partial(_evaluate_biotac_tool_autoencoder_svm, 50, 'linear', device),
        
        'biotac_tool_20_autoencoder_svm_rbf'        : partial(_evaluate_biotac_tool_autoencoder_svm, 20, 'rbf', device),
        'biotac_tool_30_autoencoder_svm_rbf'        : partial(_evaluate_biotac_tool_autoencoder_svm, 30, 'rbf', device),
        'biotac_tool_50_autoencoder_svm_rbf'        : partial(_evaluate_biotac_tool_autoencoder_svm, 50, 'rbf', device),
        
        'biotac_tool_20_autoencoder_mlp'            : partial(_evaluate_biotac_tool_autoencoder_mlp, 20, device),
        'biotac_tool_30_autoencoder_mlp'            : partial(_evaluate_biotac_tool_autoencoder_mlp, 30, device),
        'biotac_tool_50_autoencoder_mlp'            : partial(_evaluate_biotac_tool_autoencoder_mlp, 50, device),
        
        'biotac_handover_rod_svm_linear'            : partial(_evaluate_biotac_handover_svm, 'rod', 'linear', False),
        'biotac_handover_box_svm_linear'            : partial(_evaluate_biotac_handover_svm, 'box', 'linear', False),
        'biotac_handover_plate_svm_linear'          : partial(_evaluate_biotac_handover_svm, 'plate', 'linear', False),
        
        'biotac_handover_rod_svm_rbf'               : partial(_evaluate_biotac_handover_svm, 'rod', 'rbf', False),
        'biotac_handover_box_svm_rbf'               : partial(_evaluate_biotac_handover_svm, 'box', 'rbf', False),
        'biotac_handover_plate_svm_rbf'             : partial(_evaluate_biotac_handover_svm, 'plate', 'rbf', False),
        
        'biotac_handover_rod_fftsvm_linear'         : partial(_evaluate_biotac_handover_svm, 'rod', 'linear', True),
        'biotac_handover_box_fftsvm_linear'         : partial(_evaluate_biotac_handover_svm, 'box', 'linear', True),
        'biotac_handover_plate_fftsvm_linear'       : partial(_evaluate_biotac_handover_svm, 'plate', 'linear', True),
        
        'biotac_handover_rod_fftsvm_rbf'            : partial(_evaluate_biotac_handover_svm, 'rod', 'rbf', True),
        'biotac_handover_box_fftsvm_rbf'            : partial(_evaluate_biotac_handover_svm, 'box', 'rbf', True),
        'biotac_handover_plate_fftsvm_rbf'          : partial(_evaluate_biotac_handover_svm, 'plate', 'rbf', True),
        
        'biotac_handover_rod_lstm'                  : partial(_evaluate_biotac_handover_lstm, 'rod', device),
        'biotac_handover_box_lstm'                  : partial(_evaluate_biotac_handover_lstm, 'box', device),
        'biotac_handover_plate_lstm'                : partial(_evaluate_biotac_handover_lstm, 'plate', device),
        
        'biotac_handover_rod_mlp'                   : partial(_evaluate_biotac_handover_mlp, 'rod', device, False),
        'biotac_handover_box_mlp'                   : partial(_evaluate_biotac_handover_mlp, 'box', device, False),
        'biotac_handover_plate_mlp'                 : partial(_evaluate_biotac_handover_mlp, 'plate', device, False),
                
        'biotac_handover_rod_fftmlp'                : partial(_evaluate_biotac_handover_mlp, 'rod', device, True),
        'biotac_handover_box_fftmlp'                : partial(_evaluate_biotac_handover_mlp, 'box', device, True),
        'biotac_handover_plate_fftmlp'              : partial(_evaluate_biotac_handover_mlp, 'plate', device, True),
        
        'biotac_food_svm_linear'                    : partial(_evaluate_biotac_food_svm, 'linear', False),
        'biotac_food_svm_rbf'                       : partial(_evaluate_biotac_food_svm, 'rbf', False),
        'biotac_food_fftsvm_linear'                 : partial(_evaluate_biotac_food_svm, 'linear', True),
        'biotac_food_fftsvm_rbf'                    : partial(_evaluate_biotac_food_svm, 'rbf', True),
        'biotac_food_lstm'                          : partial(_evaluate_biotac_food_lstm, device),
        'biotac_food_mlp'                           : partial(_evaluate_biotac_food_mlp, device, False),
        'biotac_food_fftmlp'                        : partial(_evaluate_biotac_food_mlp, device, True),
        
    }
    
    if experiment_name not in experiments: raise Exception('Experiment not found')
    
    test_loss_mean, test_loss_std = experiments[experiment_name]()
    
    print('Result for {:s}: {:0.4f} ± {:0.4f}'.format(experiment_name, test_loss_mean, test_loss_std))


#    ______          _             _                 
#   |  ____|        | |           | |                
#   | |____   ____ _| |_   _  __ _| |_ ___  _ __ ___ 
#   |  __\ \ / / _` | | | | |/ _` | __/ _ \| '__/ __|
#   | |___\ V / (_| | | |_| | (_| | || (_) | |  \__ \
#   |______\_/ \__,_|_|\__,_|\__,_|\__\___/|_|  |___/
#                                                    


def _create_sklearn_evaluator(estimator, param_grid, scoring, K=4, N=5, callback=None):
    
    gs_estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=K, n_jobs=1, refit=True)
    
    def evaluate(X, y, verbose=True):
        
        test_losses = np.zeros(N)
        
        for n in range(N):
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=n)
            gs_estimator.fit(X_train, y_train)
            test_loss = -gs_estimator.score(X_test, y_test)
            test_losses[n] = test_loss
            
            if callback is not None: callback(gs_estimator, X_test, y_test)
            if verbose: print('Iteration {:d} | Test Loss = {:0.4f}'.format(n, test_loss))

        return np.mean(test_losses), np.std(test_losses)

    return evaluate


def _create_skorch_evaluator(estimator, test_scoring, K=4, N=5):
    
    def evaluate(X, y, verbose=True):
        
        test_losses = np.zeros(N)
        
        for n in range(N):
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=n)
            estimator.fit(X_train, y_train)
            test_loss = -test_scoring(estimator, X_test, y_test)
            test_losses[n] = test_loss
            
            if verbose: print('Iteration {:d} | Test Loss = {:0.4f}'.format(n, test_loss))

        return np.mean(test_losses), np.std(test_losses)

    return evaluate


#    _   _            _                   _        ______                      _                      _       
#   | \ | |          | |                 | |      |  ____|                    (_)                    | |      
#   |  \| | ___ _   _| |_ ___  _   _  ___| |__    | |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_ ___ 
#   | . ` |/ _ \ | | | __/ _ \| | | |/ __| '_ \   |  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ __|
#   | |\  |  __/ |_| | || (_) | |_| | (__| | | |  | |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_\__ \
#   |_| \_|\___|\__,_|\__\___/ \__,_|\___|_| |_|  |______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|___/
#                                                             | |                                             


def _evaluate_neutouch_tool_svm(tool_length, kernel, perform_fft):
    
    from sklearn.svm import SVR
    
    npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
    
    X = npzfile['signals'] / 40
    y = npzfile['labels'] * 100
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = np.reshape(X, (X.shape[0], -1))
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVR(kernel=kernel), param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_neutouch_tool_lstm(tool_length, device):
    
    from skorch import NeuralNetRegressor
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
    
    X = npzfile['signals'] / 40
    y = npzfile['labels']
    
    X = torch.Tensor(np.swapaxes(X, 1, 2))
    y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    callbacks = [
        EpochScoring(scoring='neg_mean_absolute_error'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetRegressor(LSTMRegressor,
                                   module__input_dim=80,
                                   module__output_dim=1,
                                   optimizer=torch.optim.Adam,
                                   train_split=CVSplit(10),
                                   max_epochs=1000,
                                   device=device,
                                   lr=0.001,
                                   verbose=0,
                                   callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return -np.mean(np.abs(estimator.evaluation_step(X_test).cpu().numpy() - y_test.cpu().numpy())) * 100
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_neutouch_tool_autoencoder(device):
    
    from skorch import NeuralNetRegressor
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, Checkpoint
    
    npzfile20 = np.load('preprocessed/neutouch_tool_20.npz')
    npzfile30 = np.load('preprocessed/neutouch_tool_30.npz')
    npzfile50 = np.load('preprocessed/neutouch_tool_50.npz')
    
    X = np.vstack((npzfile20['signals'], npzfile30['signals'], npzfile50['signals'])) / 40
    X = torch.Tensor(np.reshape(X, (X.shape[0], -1)))
    
    callbacks = [
        EarlyStopping(patience=100),
        Checkpoint(dirname='models/neutouch_tool_autoencoder')
    ]
    
    estimator = NeuralNetRegressor(Autoencoder,
                                   module__input_dim=X.shape[1],
                                   module__latent_dim=32,
                                   optimizer=torch.optim.Adam,
                                   train_split=CVSplit(10),
                                   max_epochs=1000,
                                   device=device,
                                   lr=0.001,
                                   verbose=1,
                                   callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return -np.mean((estimator.evaluation_step(X_test).cpu().numpy() - y_test.cpu().numpy()) ** 2)
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, X)


def _evaluate_neutouch_tool_autoencoder_svm(tool_length, kernel, device):
    
    from sklearn.svm import SVR
    
    npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
    
    X = npzfile['signals'] / 40
    y = npzfile['labels'] * 100
    
    X = torch.Tensor(np.reshape(X, (X.shape[0], -1))).to(device)
    
    with torch.no_grad():
        autoencoder = Autoencoder(X.shape[1], 32).to(device)
        autoencoder.load_state_dict(torch.load('models/neutouch_tool_autoencoder/params.pt'))
        X = autoencoder.encode(X)
        X = X.cpu().numpy()
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVR(kernel=kernel), param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_neutouch_tool_autoencoder_mlp(tool_length, device):
    
    from skorch import NeuralNetRegressor
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
    
    X = npzfile['signals'] / 40
    y = npzfile['labels']
    
    X = torch.Tensor(np.reshape(X, (X.shape[0], -1))).to(device)
    y = torch.Tensor(np.reshape(y, (-1, 1))).to(device)
    
    with torch.no_grad():
        autoencoder = Autoencoder(X.shape[1], 32).to(device)
        autoencoder.load_state_dict(torch.load('models/neutouch_tool_autoencoder/params.pt'))
        X = autoencoder.encode(X)
    
    callbacks = [
        EpochScoring(scoring='neg_mean_absolute_error'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetRegressor(MLPRegressor,
                                   module__input_dim=32,
                                   module__output_dim=1,
                                   optimizer=torch.optim.Adam,
                                   train_split=CVSplit(10),
                                   max_epochs=1000,
                                   device=device,
                                   lr=0.0001,
                                   verbose=1,
                                   callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return -np.mean(np.abs(estimator.evaluation_step(X_test).cpu().numpy() - y_test.cpu().numpy())) * 100
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_neutouch_halftool_svm(tool_length, kernel, perform_fft):
    
    from sklearn.svm import SVR
    
    npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
    
    X = npzfile['signals'][:, 0:40, :] / 40
    y = npzfile['labels'] * 100
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = np.reshape(X, (X.shape[0], -1))
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVR(kernel=kernel), param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_neutouch_halftool_lstm(tool_length, device):
    
    from skorch import NeuralNetRegressor
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
    
    X = npzfile['signals'][:, 0:40, :] / 40
    y = npzfile['labels']
    
    X = torch.Tensor(np.swapaxes(X, 1, 2))
    y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    callbacks = [
        EpochScoring(scoring='neg_mean_absolute_error'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetRegressor(LSTMRegressor,
                                   module__input_dim=40,
                                   module__output_dim=1,
                                   optimizer=torch.optim.Adam,
                                   train_split=CVSplit(10),
                                   max_epochs=1000,
                                   device=device,
                                   lr=0.001,
                                   verbose=1,
                                   callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return -np.mean(np.abs(estimator.evaluation_step(X_test).cpu().numpy() - y_test.cpu().numpy())) * 100
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_neutouch_singletool(tool_length):
    
    from sklearn.svm import SVR
    
    npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
    
    X = npzfile['signals'] / 40
    y = npzfile['labels'] * 100
    
    best_taxel = 0
    best_test_loss_mean = float('inf')
    best_test_loss_std = float('inf')
    
    with open(f'results/neutouch_singletool_{tool_length}.csv', 'w') as file:
    
        for taxel in range(1, 81):

            evaluate = _create_sklearn_evaluator(SVR(kernel='rbf', C=10), {}, 'neg_mean_absolute_error')
            test_loss_mean, test_loss_std = evaluate(X[:, taxel-1, :], y, verbose=False)
            file.write(f'{taxel},{test_loss_mean},{test_loss_std}\n')

            if test_loss_mean < best_test_loss_mean:

                best_taxel = taxel
                best_test_loss_mean = test_loss_mean
                best_test_loss_std = test_loss_std

            print('Result for taxel {:02d}: {:0.4f} ± {:0.4f}'.format(taxel, test_loss_mean, test_loss_std), flush=True)
    
    print(f'Best performing taxel is {best_taxel}')
    
    return best_test_loss_mean, best_test_loss_std


def _evaluate_neutouch_handover_svm(item, kernel, perform_fft):
    
    from sklearn.svm import SVC
    
    npzfile = np.load(f'preprocessed/neutouch_handover_{item}.npz')
    
    X = npzfile['signals'] / 40
    y = npzfile['labels'][:, 0]
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = np.reshape(X, (X.shape[0], -1))
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVC(kernel=kernel), param_grid, 'accuracy')
    
    return evaluate(X, y)


def _evaluate_neutouch_handover_lstm(item, device):
    
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'preprocessed/neutouch_handover_{item}.npz')
    
    X = npzfile['signals'] / 40
    y = npzfile['labels'][:, 0].astype(np.compat.long)
    
    X = torch.Tensor(np.swapaxes(X, 1, 2))
    
    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(LSTMClassifier,
                                    module__input_dim=80,
                                    module__output_dim=2,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    optimizer__weight_decay=0.01,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.0001,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_neutouch_handover_mlp(item, device, perform_fft):
    
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'preprocessed/neutouch_handover_{item}.npz')
    
    X = npzfile['signals'] / 40
    y = npzfile['labels'][:, 0].astype(np.compat.long)
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
        
    X = torch.Tensor(np.reshape(X, (X.shape[0], -1)))
    
    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(MLPRegressor,
                                    module__input_dim=X.shape[1],
                                    module__output_dim=7,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    optimizer__weight_decay=0.01,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.0001,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_neutouch_halfhandover_svm(item, kernel, perform_fft):
    
    from sklearn.svm import SVC
    
    npzfile = np.load(f'preprocessed/neutouch_handover_{item}.npz')
    
    X = npzfile['signals'] / 40
    y = npzfile['labels'][:, 0]
    
    X = X[:, 0:40, :]
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = np.reshape(X, (X.shape[0], -1))
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVC(kernel=kernel), param_grid, 'accuracy')
    
    return evaluate(X, y)


def _evaluate_neutouch_food_svm(kernel, perform_fft):
    
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    
    import seaborn as sn
    import matplotlib.pyplot as plt
    import pandas as pd
    
    X, y, classes = _load_food_data('neutouch')

    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = np.reshape(X, (X.shape[0], -1))
    
    cmatrix = None
    
    def confusion_matrix_callback(gs_estimator, X_test, y_test):

        nonlocal cmatrix
        
        matrix = confusion_matrix(gs_estimator.predict(X_test), y_test)
        cmatrix = matrix if cmatrix is None else cmatrix + matrix 
        plt.figure(figsize=(6, 5))
        sn.heatmap(pd.DataFrame(cmatrix, index=classes, columns=classes), annot=True)
        plt.savefig('diagrams/neutouch_food_cmatrix.pdf', bbox_inches='tight')
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVC(kernel=kernel), param_grid, 'accuracy', callback=confusion_matrix_callback)
    
    return evaluate(X, y)


def _evaluate_neutouch_food_lstm(device):
    
    from sklearn.preprocessing import OneHotEncoder
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    X, y, classes = _load_food_data('neutouch')
    X = torch.Tensor(np.swapaxes(X, 1, 2))
    
    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(LSTMClassifier,
                                    module__input_dim=80,
                                    module__output_dim=7,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.0009,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_neutouch_food_mlp(device, perform_fft):
    
    from sklearn.preprocessing import OneHotEncoder
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    X, y, classes = _load_food_data('neutouch')
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
        
    X = torch.Tensor(np.reshape(X, (X.shape[0], -1)))
    print(X.shape)
    
    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(MLPRegressor,
                                    module__input_dim=X.shape[1],
                                    module__output_dim=7,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    optimizer__weight_decay=0.001,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.0003,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_neutouch_halffood_svm(kernel, perform_fft):
    
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    
    import seaborn as sn
    import matplotlib.pyplot as plt
    import pandas as pd
    
    X, y, classes = _load_food_data('neutouch')
    X = X[:, 0:40, :]
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = np.reshape(X, (X.shape[0], -1))
    
    cmatrix = None
    
    def confusion_matrix_callback(gs_estimator, X_test, y_test):

        nonlocal cmatrix
        
        matrix = confusion_matrix(gs_estimator.predict(X_test), y_test)
        cmatrix = matrix if cmatrix is None else cmatrix + matrix 
        plt.figure(figsize=(6, 5))
        sn.heatmap(pd.DataFrame(cmatrix, index=classes, columns=classes), annot=True)
        plt.savefig('diagrams/neutouch_halffood_cmatrix.pdf', bbox_inches='tight')
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVC(kernel=kernel), param_grid, 'accuracy', callback=confusion_matrix_callback)
    
    return evaluate(X, y)


def _evaluate_neutouch_halffood_lstm(device):
    
    from sklearn.preprocessing import OneHotEncoder
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    X, y, classes = _load_food_data('neutouch')
    X = X[:, 0:40, :]
    X = torch.Tensor(np.swapaxes(X, 1, 2))
    
    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(LSTMClassifier,
                                    module__input_dim=40,
                                    module__output_dim=7,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.0001,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_neutouch_halffood_mlp(device, perform_fft):
    
    from sklearn.preprocessing import OneHotEncoder
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    X, y, classes = _load_food_data('neutouch')
    X = X[:, 0:40, :]
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
        
    X = torch.Tensor(np.reshape(X, (X.shape[0], -1)))
    print(X.shape)
    
    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(MLPRegressor,
                                    module__input_dim=X.shape[1],
                                    module__output_dim=7,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    optimizer__weight_decay=0.001,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.001,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


#    ____  _       _                ______                      _                      _       
#   |  _ \(_)     | |              |  ____|                    (_)                    | |      
#   | |_) |_  ___ | |_ __ _  ___   | |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_ ___ 
#   |  _ <| |/ _ \| __/ _` |/ __|  |  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ __|
#   | |_) | | (_) | || (_| | (__   | |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_\__ \
#   |____/|_|\___/ \__\__,_|\___|  |______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|___/
#                                              | |                                             


def _evaluate_biotac_tool_svm(tool_length, kernel, perform_fft):
    
    from sklearn.svm import SVR
    
    npzfile = np.load(f'preprocessed/biotac_tool_{tool_length}.npz')
    
    X = npzfile['signals'] / 100
    y = npzfile['labels'] * 100
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = np.reshape(X, (X.shape[0], -1))
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVR(kernel=kernel), param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_biotac_tool_lstm(tool_length, device):
    
    from skorch import NeuralNetRegressor
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'preprocessed/biotac_tool_{tool_length}.npz')
    
    X = npzfile['signals'] / 100
    y = npzfile['labels']
    
    X = torch.Tensor(np.expand_dims(X, 2))
    y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    callbacks = [
        EpochScoring(scoring='neg_mean_absolute_error'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetRegressor(LSTMRegressor,
                                   module__input_dim=1,
                                   module__output_dim=1,
                                   optimizer=torch.optim.Adam,
                                   train_split=CVSplit(10),
                                   max_epochs=1000,
                                   device=device,
                                   lr=0.003,
                                   verbose=1,
                                   callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return -np.mean(np.abs(estimator.evaluation_step(X_test).cpu().numpy() - y_test.cpu().numpy())) * 100
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_biotac_tool_autoencoder(device):
    
    from skorch import NeuralNetRegressor
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, Checkpoint
    
    npzfile20 = np.load('preprocessed/biotac_tool_20.npz')
    npzfile30 = np.load('preprocessed/biotac_tool_30.npz')
    npzfile50 = np.load('preprocessed/biotac_tool_50.npz')
    
    X = np.vstack((npzfile20['signals'], npzfile30['signals'], npzfile50['signals'])) / 100
    X = torch.Tensor(np.reshape(X, (X.shape[0], -1)))
    
    callbacks = [
        EarlyStopping(patience=100),
        Checkpoint(dirname='models/biotac_tool_autoencoder')
    ]
    
    estimator = NeuralNetRegressor(Autoencoder,
                                   module__input_dim=X.shape[1],
                                   module__latent_dim=32,
                                   optimizer=torch.optim.Adam,
                                   train_split=CVSplit(10),
                                   max_epochs=1000,
                                   device=device,
                                   lr=0.001,
                                   verbose=1,
                                   callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return -np.mean((estimator.evaluation_step(X_test).cpu().numpy() - y_test.cpu().numpy()) ** 2)
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, X)


def _evaluate_biotac_tool_autoencoder_svm(tool_length, kernel, device):
    
    from sklearn.svm import SVR
    
    npzfile = np.load(f'preprocessed/biotac_tool_{tool_length}.npz')
    
    X = npzfile['signals'] / 100
    y = npzfile['labels'] * 100
    
    X = torch.Tensor(np.reshape(X, (X.shape[0], -1))).to(device)
    
    with torch.no_grad():
        autoencoder = Autoencoder(X.shape[1], 32).to(device)
        autoencoder.load_state_dict(torch.load('models/biotac_tool_autoencoder/params.pt'))
        X = autoencoder.encode(X) / 20
        X = X.cpu().numpy()
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVR(kernel=kernel), param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_biotac_tool_autoencoder_mlp(tool_length, device):
    
    from skorch import NeuralNetRegressor
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'preprocessed/biotac_tool_{tool_length}.npz')
    
    X = npzfile['signals'] / 100
    y = npzfile['labels']
    
    X = torch.Tensor(np.reshape(X, (X.shape[0], -1))).to(device)
    y = torch.Tensor(np.reshape(y, (-1, 1))).to(device)
    
    with torch.no_grad():
        autoencoder = Autoencoder(X.shape[1], 32).to(device)
        autoencoder.load_state_dict(torch.load('models/biotac_tool_autoencoder/params.pt'))
        X = autoencoder.encode(X) / 20
    
    callbacks = [
        EpochScoring(scoring='neg_mean_absolute_error'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetRegressor(MLPRegressor,
                                   module__input_dim=32,
                                   module__output_dim=1,
                                   optimizer=torch.optim.Adam,
                                   train_split=CVSplit(10),
                                   max_epochs=1000,
                                   device=device,
                                   lr=0.0001,
                                   verbose=1,
                                   callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return -np.mean(np.abs(estimator.evaluation_step(X_test).cpu().numpy() - y_test.cpu().numpy())) * 100
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_biotac_handover_svm(item, kernel, perform_fft):
    
    from sklearn.svm import SVC
    
    npzfile = np.load(f'preprocessed/biotac_handover_{item}.npz')
    
    X = npzfile['signals'] / 1000
    y = npzfile['labels'][:, 0]
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = np.reshape(X, (X.shape[0], -1))
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVC(kernel=kernel), param_grid, 'accuracy')
    
    return evaluate(X, y)


def _evaluate_biotac_handover_lstm(item, device):
    
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'preprocessed/biotac_handover_{item}.npz')
    
    X = npzfile['signals'] / 1000
    y = npzfile['labels'][:, 0].astype(np.compat.long)
    X = torch.Tensor(np.expand_dims(X, 2))
    
    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(LSTMClassifier,
                                    module__input_dim=1,
                                    module__output_dim=2,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    optimizer__weight_decay=0,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.003,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_biotac_handover_mlp(item, device, perform_fft):
    
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'preprocessed/biotac_handover_{item}.npz')
    
    X = npzfile['signals'] / 1000
    y = npzfile['labels'][:, 0].astype(np.compat.long)
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
        
    X = torch.Tensor(X)

    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(MLPRegressor,
                                    module__input_dim=X.shape[1],
                                    module__output_dim=2,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    optimizer__weight_decay=0.01,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.0001,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_biotac_food_svm(kernel, perform_fft):
    
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix
    
    import seaborn as sn
    import matplotlib.pyplot as plt
    import pandas as pd
    
    X, y, classes = _load_food_data('biotac')

    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = np.reshape(X, (X.shape[0], -1))
    
    cmatrix = None
    
    def confusion_matrix_callback(gs_estimator, X_test, y_test):

        nonlocal cmatrix
        
        matrix = confusion_matrix(gs_estimator.predict(X_test), y_test)
        cmatrix = matrix if cmatrix is None else cmatrix + matrix 
        plt.figure(figsize=(6, 5))
        sn.heatmap(pd.DataFrame(cmatrix, index=classes, columns=classes), annot=True)
        plt.savefig('diagrams/biotac_food_cmatrix.pdf', bbox_inches='tight')
    
    param_grid = { 'C': [1, 3, 10, 30, 100] }
    evaluate = _create_sklearn_evaluator(SVC(kernel=kernel), param_grid, 'accuracy', callback=confusion_matrix_callback)
    
    return evaluate(X, y)


def _evaluate_biotac_food_lstm(device):
    
    from sklearn.preprocessing import OneHotEncoder
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    X, y, classes = _load_food_data('biotac')
    X = X[:, 100:1100:3]
    X = torch.Tensor(np.expand_dims(X, 2))
    
    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(LSTMClassifier,
                                    module__input_dim=1,
                                    module__output_dim=7,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.0005,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


def _evaluate_biotac_food_mlp(device, perform_fft):
    
    from sklearn.preprocessing import OneHotEncoder
    from skorch import NeuralNetClassifier
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    X, y, classes = _load_food_data('biotac')
    
    if perform_fft:
        X = np.abs(np.fft.fft(X)) / 10
    
    X = torch.Tensor(X)
    
    callbacks = [
        EpochScoring(scoring='accuracy'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetClassifier(MLPRegressor,
                                    module__input_dim=X.shape[1],
                                    module__output_dim=7,
                                    criterion=nn.CrossEntropyLoss,
                                    optimizer=torch.optim.Adam,
                                    optimizer__weight_decay=0.001,
                                    train_split=CVSplit(10),
                                    max_epochs=1000,
                                    device=device,
                                    lr=0.001,
                                    verbose=1,
                                    callbacks=callbacks)
    
    def test_scoring(estimator, X_test, y_test):
        return (np.count_nonzero(np.argmax(estimator.evaluation_step(X_test).cpu().numpy(), axis=1) == y_test) / len(y_test))
    
    evaluate = _create_skorch_evaluator(estimator, test_scoring)
    
    return evaluate(X, y)


#    _    _      _                    ______                _   _                 
#   | |  | |    | |                  |  ____|              | | (_)                
#   | |__| | ___| |_ __   ___ _ __   | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
#   |  __  |/ _ \ | '_ \ / _ \ '__|  |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#   | |  | |  __/ | |_) |  __/ |     | |  | |_| | | | | (__| |_| | (_) | | | \__ \
#   |_|  |_|\___|_| .__/ \___|_|     |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#                 | |                                                             


def _load_food_data(sensor_type):
    
    classes = [ 'empty', 'water', 'tofu', 'watermelon', 'banana', 'apple', 'pepper' ]
    
    npzfile0 = np.load(f'preprocessed/{sensor_type}_food_empty.npz')
    npzfile1 = np.load(f'preprocessed/{sensor_type}_food_water.npz')
    npzfile2 = np.load(f'preprocessed/{sensor_type}_food_tofu.npz')
    npzfile3 = np.load(f'preprocessed/{sensor_type}_food_watermelon.npz')
    npzfile4 = np.load(f'preprocessed/{sensor_type}_food_banana.npz')
    npzfile5 = np.load(f'preprocessed/{sensor_type}_food_apple.npz')
    npzfile6 = np.load(f'preprocessed/{sensor_type}_food_pepper.npz')
    
    X = np.vstack((npzfile0['signals'],
                   npzfile1['signals'],
                   npzfile2['signals'],
                   npzfile3['signals'],
                   npzfile4['signals'],
                   npzfile5['signals'],
                   npzfile6['signals'])) / 100
    
    if sensor_type == 'biotac':   X = X / 100
    if sensor_type == 'neutouch': X = X / 40
    
    y = np.concatenate((npzfile0['labels'] * 0,
                        npzfile1['labels'] * 1,
                        npzfile2['labels'] * 2,
                        npzfile3['labels'] * 3,
                        npzfile4['labels'] * 4,
                        npzfile5['labels'] * 5,
                        npzfile6['labels'] * 6)).astype(int)
    
    return X, y, classes


#    _   _ _   _    __  __           _      _     
#   | \ | | \ | |  |  \/  |         | |    | |    
#   |  \| |  \| |  | \  / | ___   __| | ___| |___ 
#   | . ` | . ` |  | |\/| |/ _ \ / _` |/ _ \ / __|
#   | |\  | |\  |  | |  | | (_) | (_| |  __/ \__ \
#   |_| \_|_| \_|  |_|  |_|\___/ \__,_|\___|_|___/
#                                                 


import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):

    def __init__(self, input_dim, output_dim):
        
        super(LSTMRegressor, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, 16, batch_first=True)
        self.linear1 = nn.Linear(16, 8)
        self.linear2 = nn.Linear(8, output_dim)

    def forward(self, X):
        
        X, _ = self.lstm(X)
        X = torch.squeeze(X[:, -1, :])
        X = torch.relu(self.linear1(X))
        
        return self.linear2(X)


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, 16, batch_first=True)
        self.linear1 = nn.Linear(16, 8)
        self.linear2 = nn.Linear(8, output_dim)

    def forward(self, X):
        
        X, _ = self.lstm(X)
        X = torch.squeeze(X[:, -1, :])
        X = torch.relu(self.linear1(X))
        
        return self.linear2(X)

class MLPRegressor(nn.Module):

    def __init__(self, input_dim, output_dim):
        
        super(MLPRegressor, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, 16)
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, output_dim)

    def forward(self, X):
        
        X = torch.relu(self.linear1(X))
        X = torch.relu(self.linear2(X))
        
        return self.linear3(X)


class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        
        super(Autoencoder, self).__init__()
        
        self.encoder1 = nn.Linear(input_dim, 128)
        self.encoder2 = nn.Linear(128, 64)
        self.encoder3 = nn.Linear(64, latent_dim)
        
        self.decoder1 = nn.Linear(latent_dim, 64)
        self.decoder2 = nn.Linear(64, 128)
        self.decoder3 = nn.Linear(128, input_dim)

    def forward(self, X):
        
        latent = self.encode(X)
        X_reconstruct = self.decode(latent)
        
        return X_reconstruct
    
    def encode(self, X):
        
        latent = torch.relu(self.encoder1(X))
        latent = torch.relu(self.encoder2(latent))
        latent = self.encoder3(latent)
        
        return latent
    
    def decode(self, latent):
        
        X_reconstruct = torch.relu(self.decoder1(latent))
        X_reconstruct = torch.relu(self.decoder2(X_reconstruct))
        X_reconstruct = self.decoder3(X_reconstruct)
        
        return X_reconstruct


#     _____ _      _____    _____       _             __               
#    / ____| |    |_   _|  |_   _|     | |           / _|              
#   | |    | |      | |      | |  _ __ | |_ ___ _ __| |_ __ _  ___ ___ 
#   | |    | |      | |      | | | '_ \| __/ _ \ '__|  _/ _` |/ __/ _ \
#   | |____| |____ _| |_    _| |_| | | | ||  __/ |  | || (_| | (_|  __/
#    \_____|______|_____|  |_____|_| |_|\__\___|_|  |_| \__,_|\___\___|
#                                                                      


if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) == 2: evaluate(sys.argv[1])
    if len(sys.argv) == 3: evaluate(sys.argv[1], sys.argv[2])
