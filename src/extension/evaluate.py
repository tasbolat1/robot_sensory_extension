from functools import partial

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC

from skorch import NeuralNetRegressor
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping, EpochScoring

import numpy as np

def evaluate(experiment_name, device='cuda'):
    
    experiments = {
        'neutouch_kernel_rodTap_20_4000_svm_linear':partial(_evaluate_tool_svm, 'rodTap', 20, 4000,'linear'),
        'neutouch_kernel_rodTap_20_4000_svm_rbf':partial(_evaluate_tool_svm, 'rodTap', 20, 4000,'rbf'),
        'neutouch_kernel_rodTap_20_4000_lstm':partial(_evaluate_tool_lstm, 'rodTap', 20, 4000, 'cuda'),
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
    
    gs_estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring, cv=K, n_jobs=-1, refit=True)
    
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
            #print(X_train[0])
            estimator.fit(X_train, y_train)
            
            test_loss = -test_scoring(estimator, X_test, y_test)
            test_losses[n] = test_loss
            
            if verbose: print('Iteration {:d} | Test Loss = {:0.4f}'.format(n, test_loss))

        return np.mean(test_losses), np.std(test_losses)

    return evaluate


#    _______          _    ______                      _                      _       
#   |__   __|        | |  |  ____|                    (_)                    | |      
#      | | ___   ___ | |  | |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_ ___ 
#      | |/ _ \ / _ \| |  |  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ __|
#      | | (_) | (_) | |  | |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_\__ \
#      |_|\___/ \___/|_|  |______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|___/
#                                     | |                                             


def _load_tool(tool_length, signal_type, transformation):
    
    if signal_type == 'biotac':
        
        npzfile = np.load(f'preprocessed/biotac_tool_{tool_length}.npz')
        X = npzfile['signals'] / 1000
        y = npzfile['labels'] * 100
    
    if signal_type == 'neutouch':
        
        npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
        X = npzfile['signals'] / 40
        y = npzfile['labels'] * 100
    
    if signal_type == 'neuhalf':
        
        npzfile = np.load(f'preprocessed/neutouch_tool_{tool_length}.npz')
        X = npzfile['signals'][:, 0:40, :] / 40
        y = npzfile['labels'] * 100
        
    if transformation == 'default':
        
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'fft':
        
        X = np.abs(np.fft.fft(X)) / 10
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'tensor' and signal_type == 'biotac':
    
        X = torch.Tensor(np.swapaxes(X, 1, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    if transformation == 'tensor' and signal_type == 'neutouch':
        
        torch.Tensor(np.expand_dims(X, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    return X, y


def _evaluate_tool_svm(task, tool_type, freq, kernel):

    npzfile = np.load(f'../../data/kernel_features/kernel_{task}_{tool_type}_{freq}.npz')
    
    X = npzfile['signals']
    print('X shape:', X.shape)
    print('X max min:', X.max(), X.min())
    X = np.reshape(X, (X.shape[0], -1))
    y = npzfile['labels']*100
    y = y.ravel()
    print(y.max())
    
    #param_grid = { 'C': [0.01, 0.1, 1, 3, 10, 50, 100] }
    param_grid = { 'C': [10, 50, 100, 150, 200, 300] }
    
    estimator = SVR(kernel=kernel, max_iter=5000)
    evaluate = _create_sklearn_evaluator(estimator, param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)

def _evaluate_tool_lstm(task, tool_type, freq, device):
    device = 'cuda'
    from skorch import NeuralNetRegressor
    from skorch.dataset import CVSplit
    from skorch.callbacks import EarlyStopping, EpochScoring
    
    npzfile = np.load(f'../../data/kernel_features/kernel_{task}_{tool_type}_{freq}.npz')
    
    X = torch.FloatTensor( npzfile['signals'] ).to(device)
    y = torch.FloatTensor( npzfile['labels']*100 ).to(device)
    #X = npzfile['signals']
    #y = npzfile['labels']*100
    print('X shape:', X.shape)
    print('X max min:', X.max(), X.min())
    print('y shape:', y.shape)
    
    callbacks = [
        EpochScoring(scoring='neg_mean_absolute_error'),
        EarlyStopping(patience=100)
    ]
    
    estimator = NeuralNetRegressor(LSTMRegressor,
                                   module__input_dim=X.shape[-1],
                                   module__output_dim=1,
                                   optimizer=torch.optim.Adam,
                                   optimizer__weight_decay=0.5,
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


def _evaluate_tool_mlp(tool_length, signal_type, perform_fft):

    X, y = _load_tool(tool_length, signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'learning_rate_init': [0.01, 0.03, 0.1, 0.3],
        'alpha': [0.0001, 0.001]
    }
    
    estimator = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000)
    evaluate = _create_sklearn_evaluator(estimator, param_grid, 'neg_mean_absolute_error')
    
    return evaluate(X, y)


def _evaluate_tool_rnn(tool_length, signal_type):
    
    pass


def _evaluate_tool_aesvm(tool_length, kernel):

    pass


def _evaluate_tool_aemlp(tool_length):

    pass


def _evaluate_tool_neusingle_svm(tool_length, kernel, perform_fft):

    X, y = _load_tool(tool_length, 'neutouch', 'fft' if perform_fft else 'default')
    X = np.reshape(X, (X.shape[0], 80, -1))

    best_taxel = 0
    best_test_loss_mean = float('inf')
    best_test_loss_std = float('inf')
    
    with open(f'results/neutouch_singletool_{tool_length}.csv', 'w') as file:
    
        for taxel in range(1, 81):
            
            param_grid = { 'C': [1, 3, 10, 30, 100] }
    
            estimator = SVR(kernel=kernel, max_iter=5000)
            evaluate = _create_sklearn_evaluator(estimator, param_grid, 'neg_mean_absolute_error')
            test_loss_mean, test_loss_std = evaluate(X[:, taxel-1, :], y)
            file.write(f'{taxel},{test_loss_mean},{test_loss_std}\n')

            if test_loss_mean < best_test_loss_mean:

                best_taxel = taxel
                best_test_loss_mean = test_loss_mean
                best_test_loss_std = test_loss_std

            print('Result for taxel {:02d}: {:0.4f} ± {:0.4f}'.format(taxel, test_loss_mean, test_loss_std), flush=True)
    
    print(f'Best performing taxel is {best_taxel}')
    
    return best_test_loss_mean, best_test_loss_std


#    _    _                 _                        ______                      _                      _       
#   | |  | |               | |                      |  ____|                    (_)                    | |      
#   | |__| | __ _ _ __   __| | _____   _____ _ __   | |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_ ___ 
#   |  __  |/ _` | '_ \ / _` |/ _ \ \ / / _ \ '__|  |  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ __|
#   | |  | | (_| | | | | (_| | (_) \ V /  __/ |     | |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_\__ \
#   |_|  |_|\__,_|_| |_|\__,_|\___/ \_/ \___|_|     |______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|___/
#                                                               | |                                             


def _load_handover(item, signal_type, transformation):
    
    if signal_type == 'biotac':
        
        npzfile = np.load(f'preprocessed/biotac_handover_{item}.npz')
        X = npzfile['signals'] / 1000
        y = npzfile['labels'][:, 0].astype(np.compat.long)
    
    if signal_type == 'neutouch':
        
        npzfile = np.load(f'preprocessed/neutouch_handover_{item}.npz')
        X = npzfile['signals'] / 40
        y = npzfile['labels'][:, 0].astype(np.compat.long)
    
    if signal_type == 'neuhalf':
        
        npzfile = np.load(f'preprocessed/neutouch_handover_{item}.npz')
        X = npzfile['signals'][:, 0:40, :] / 40
        y = npzfile['labels'][:, 0].astype(np.compat.long)
        
    if transformation == 'default':
        
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'fft':
        
        X = np.abs(np.fft.fft(X)) / 10
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'tensor' and signal_type == 'biotac':
    
        X = torch.Tensor(np.swapaxes(X, 1, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    if transformation == 'tensor' and signal_type == 'neutouch':
        
        torch.Tensor(np.expand_dims(X, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    return X, y


def _evaluate_handover_svm(item, signal_type, perform_fft, kernel):

    X, y = _load_handover(item, signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'C': [1, 3, 10, 30, 100]
    }
    
    estimator = SVC(kernel=kernel, max_iter=5000)
    evaluate = _create_sklearn_evaluator(estimator, param_grid, 'accuracy', N=20)
    
    return evaluate(X, y)


def _evaluate_handover_mlp(item, signal_type, perform_fft):

    X, y = _load_handover(item, signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'learning_rate_init': [0.01, 0.03, 0.1, 0.3],
        'alpha': [0.0001, 0.001]
    }
    
    estimator = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000)
    evaluate = _create_sklearn_evaluator(estimator, param_grid, 'accuracy', N=20)
    
    return evaluate(X, y)


def _evaluate_handover_rnn(item):

    pass


#    ______              _    ______                      _                      _       
#   |  ____|            | |  |  ____|                    (_)                    | |      
#   | |__ ___   ___   __| |  | |__  __  ___ __   ___ _ __ _ _ __ ___   ___ _ __ | |_ ___ 
#   |  __/ _ \ / _ \ / _` |  |  __| \ \/ / '_ \ / _ \ '__| | '_ ` _ \ / _ \ '_ \| __/ __|
#   | | | (_) | (_) | (_| |  | |____ >  <| |_) |  __/ |  | | | | | | |  __/ | | | |_\__ \
#   |_|  \___/ \___/ \__,_|  |______/_/\_\ .__/ \___|_|  |_|_| |_| |_|\___|_| |_|\__|___/
#                                        | |                                             


def _load_food(signal_type, transformation):
    
    classes = [ 'empty', 'water', 'tofu', 'watermelon', 'banana', 'apple', 'pepper' ]
    
    base_signal_type = signal_type if signal_type != 'neuhalf' else 'neutouch'
    
    npzfile0 = np.load(f'preprocessed/{base_signal_type}_food_empty.npz')
    npzfile1 = np.load(f'preprocessed/{base_signal_type}_food_water.npz')
    npzfile2 = np.load(f'preprocessed/{base_signal_type}_food_tofu.npz')
    npzfile3 = np.load(f'preprocessed/{base_signal_type}_food_watermelon.npz')
    npzfile4 = np.load(f'preprocessed/{base_signal_type}_food_banana.npz')
    npzfile5 = np.load(f'preprocessed/{base_signal_type}_food_apple.npz')
    npzfile6 = np.load(f'preprocessed/{base_signal_type}_food_pepper.npz')
    
    X = np.vstack((npzfile0['signals'],
                   npzfile1['signals'],
                   npzfile2['signals'],
                   npzfile3['signals'],
                   npzfile4['signals'],
                   npzfile5['signals'],
                   npzfile6['signals']))
    
    y = np.concatenate((npzfile0['labels'] * 0,
                        npzfile1['labels'] * 1,
                        npzfile2['labels'] * 2,
                        npzfile3['labels'] * 3,
                        npzfile4['labels'] * 4,
                        npzfile5['labels'] * 5,
                        npzfile6['labels'] * 6)).astype(np.compat.long)
    
    if signal_type == 'biotac':
        
        X = X / 1000
    
    if signal_type == 'neutouch':
        
        X = X / 40
    
    if signal_type == 'neuhalf':
        
        X = X[:, 0:40, :] / 40
        
    if transformation == 'default':
        
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'fft':
        
        X = np.abs(np.fft.fft(X)) / 10
        X = np.reshape(X, (X.shape[0], -1))
    
    if transformation == 'tensor' and signal_type == 'biotac':
    
        X = torch.Tensor(np.swapaxes(X, 1, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    if transformation == 'tensor' and signal_type == 'neutouch':
        
        torch.Tensor(np.expand_dims(X, 2))
        y = torch.Tensor(np.reshape(y, (-1, 1)))
    
    return X, y


def _evaluate_food_svm(signal_type, perform_fft, kernel):

    X, y = _load_food(signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'C': [1, 3, 10, 30, 100]
    }
    
    estimator = SVC(kernel=kernel, max_iter=5000)
    evaluate = _create_sklearn_evaluator(estimator, param_grid, 'accuracy', N=20)
    
    return evaluate(X, y)


def _evaluate_food_mlp(signal_type, perform_fft):

    X, y = _load_food(signal_type, 'fft' if perform_fft else 'default')
    
    param_grid = {
        'learning_rate_init': [0.01, 0.03, 0.1, 0.3],
        'alpha': [0.0001, 0.001]
    }
    
    estimator = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000)
    evaluate = _create_sklearn_evaluator(estimator, param_grid, 'accuracy', N=20)
    
    return evaluate(X, y)


def _evaluate_food_rnn(signal_type):

    pass


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
