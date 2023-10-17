from typing import Callable
import urllib
import json
import ipykernel
import re
import os
import numpy as np
try:
    import torch
except:
    pass

import utils


def check_closest(fn):
    inputs = [
        (6, np.array([5, 3, 4])),
        (10, np.array([12, 2, 8, 9, 13, 14])),
        (-2, np.array([-5, 12, 6, 0, -14, 3]))
    ]
    assert np.isclose(fn(*inputs[0]), 5), "Jest błąd w funkcji closest!"
    assert np.isclose(fn(*inputs[1]), 9), "Jest błąd w funkcji closest!"
    assert np.isclose(fn(*inputs[2]), 0), "Jest błąd w funkcji closest!"


def check_poly(fn):
    inputs = [
        (6, np.array([5.5, 3, 4])),
        (10, np.array([12, 2, 8, 9, 13, 14])),
        (-5, np.array([6, 3, -12, 9, -15]))
    ]
    assert np.isclose(fn(*inputs[0]), 167.5), "Jest błąd w funkcji poly!"
    assert np.isclose(fn(*inputs[1]), 1539832), "Jest błąd w funkcji poly!"
    assert np.isclose(fn(*inputs[2]), -10809), "Jest błąd w funkcji poly!"


def check_multiplication_table(fn):
    inputs = [3, 5]
    assert np.all(fn(inputs[0]) == np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])), "Jest błąd w funkcji multiplication_table!"
    assert np.all(fn(inputs[1]) == np.array([
        [1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15],
        [4, 8, 12, 16, 20], [5, 10, 15, 20, 25]
    ])), "Jest błąd w funkcji multiplication_table!"


def check_1_1(mean_error, mean_squared_error, max_error, train_sets):
    train_set_1d, train_set_2d, train_set_10d = train_sets
    assert np.isclose(mean_error(train_set_1d, np.array([8])), 8.897352)
    assert np.isclose(mean_error(train_set_2d, np.array([2.5, 5.2])), 7.89366)
    assert np.isclose(mean_error(train_set_10d, np.array(np.arange(10))), 14.16922)

    assert np.isclose(mean_squared_error(train_set_1d, np.array([3])), 23.03568)
    assert np.isclose(mean_squared_error(train_set_2d, np.array([2.4, 8.9])), 124.9397)
    assert np.isclose(mean_squared_error(train_set_10d, -np.arange(10)), 519.1699)

    assert np.isclose(max_error(train_set_1d, np.array([3])), 7.89418)
    assert np.isclose(max_error(train_set_2d, np.array([2.4, 8.9])), 14.8628)
    assert np.isclose(max_error(train_set_10d, -np.linspace(0, 5, num=10)), 23.1727)


def check_1_2(minimize_me, minimize_mse, minimize_max, train_set_1d):
    assert np.isclose(minimize_mse(train_set_1d), -0.89735)
    assert np.isclose(minimize_mse(train_set_1d * 2), -1.79470584)
    assert np.isclose(minimize_me(train_set_1d), -1.62603)
    assert np.isclose(minimize_me(train_set_1d ** 2), 3.965143)
    assert np.isclose(minimize_max(train_set_1d), 0.0152038)
    assert np.isclose(minimize_max(train_set_1d / 2), 0.007601903895526174)


def check_1_3(me_grad, mse_grad, max_grad, train_sets):
    train_set_1d, train_set_2d, train_set_10d = train_sets
    assert all(np.isclose(
        me_grad(train_set_1d, np.array([0.99])),
        [0.46666667]
    ))
    assert all(np.isclose(
        me_grad(train_set_2d, np.array([0.99, 8.44])),
        [0.21458924, 0.89772834]
    ))
    assert all(np.isclose(
        me_grad(train_set_10d, np.linspace(0, 10, num=10)),
        [-0.14131273, -0.031631, 0.04742431, 0.0353542, 0.16364242, 0.23353252,
         0.30958123, 0.35552034, 0.4747464, 0.55116738]
    ))

    assert all(np.isclose(
        mse_grad(train_set_1d, np.array([1.24])),
        [4.27470585]
    ))
    assert all(np.isclose(
        mse_grad(train_set_2d, np.array([-8.44, 10.24])),
        [-14.25378235,  21.80373175]
    ))
    assert all(np.isclose(
        max_grad(train_set_1d, np.array([5.25])),
        [1.]
    ))
    assert all(np.isclose(
        max_grad(train_set_2d, np.array([-6.28, -4.45])),
        [-0.77818704, -0.62803259]
    ))

def check_02_linear_regression(lr_cls):
    from sklearn import datasets
    os.makedirs(".checker/02/", exist_ok=True)

    input_dataset = datasets.load_boston()
    lr = lr_cls()
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    # np.savez_compressed(".checker/05/lr_boston.out.npz", data=returned)
    expected = np.load(".checker/05/lr_boston.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 24.166099, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"

    input_dataset = datasets.load_diabetes()
    lr = lr_cls()
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    # np.savez_compressed(".checker/05/lr_diabetes.out.npz", data=returned)
    expected = np.load(".checker/05/lr_iris.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 26004.287402, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"

def check_02_regularized_linear_regression(lr_cls):
    from sklearn import datasets
    os.makedirs(".checker/02/", exist_ok=True)

    np.random.seed(54)
    input_dataset = datasets.load_boston()
    lr = lr_cls()
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    # np.savez_compressed(".checker/05/rlr_boston.out.npz", data=returned)
    expected = np.load(".checker/05/rlr_boston.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 42.8331406942, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"

    np.random.seed(58)
    input_dataset = datasets.load_diabetes()
    lr = lr_cls(lr=1e-2, alpha=1e-4)
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    # np.savez_compressed(".checker/05/rlr_diabetes.out.npz", data=returned)
    expected = np.load(".checker/05/rlr_diabetes.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 26111.08336411, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"
    
    
def check_4_1_mse(fn, datasets):
    results = [torch.tensor(6.5344), torch.tensor(38.6220)]
    for (data, param), loss in zip(datasets, results):
        assert torch.allclose(fn(data, param), loss), "Wrong loss returned!"
        
def check_4_1_me(fn, datasets):
    results = [torch.tensor(2.4330), torch.tensor(6.1551)]
    for (data, param), loss in zip(datasets, results):
        assert torch.allclose(fn(data, param), loss), "Wrong loss returned!"
        
def check_4_1_max(fn, datasets):
    results = [torch.tensor(5.7086), torch.tensor(8.8057)]
    for (data, param), loss in zip(datasets, results):
        assert torch.allclose(fn(data, param), loss), "Wrong loss returned!"
        
def check_4_1_lin_reg(fn, data):
    X, y, w = data
    assert torch.allclose(fn(X, w, y), torch.tensor(100908.9141)), "Wrong loss returned!"
    
def check_4_1_reg_reg(fn, data):
    X, y, w = data
    assert torch.allclose(fn(X, w, y), torch.tensor(100910.8672)), "Wrong loss returned!"    


def check_04_logistic_reg(lr_cls):
    np.random.seed(10)
    torch.manual_seed(10)
    
    # **** First dataset ****
    input_dataset = utils.get_classification_dataset_1d()
    lr = lr_cls(1)
    lr.fit(input_dataset.data, input_dataset.target, lr=1e-3, num_steps=int(1e4))
    returned = lr.predict(input_dataset.data)
    save_path = ".checker/04/lr_dataset_1d.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 0.5098415017127991, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"
    
    preds_proba = lr.predict_proba(input_dataset.data)
    save_path = ".checker/04/lr_dataset_1d_proba.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"
    
    preds = lr.predict(input_dataset.data)
    save_path = ".checker/04/lr_dataset_1d_preds.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    # **** Second dataset ****
    input_dataset = utils.get_classification_dataset_2d()
    lr = lr_cls(2)
    lr.fit(input_dataset.data, input_dataset.target, lr=1e-2, num_steps=int(1e4))
    returned = lr.predict(input_dataset.data)
    save_path = ".checker/04/lr_dataset_2d.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(loss, 0.044230662286281586, rtol=1e-03, atol=1e-06), "Wrong value of the loss function!"
    
    preds_proba = lr.predict_proba(input_dataset.data)
    save_path = ".checker/04/lr_dataset_2d_proba.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"
    
    preds = lr.predict(input_dataset.data)
    save_path = ".checker/04/lr_dataset_2d_preds.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"
    
from types import SimpleNamespace
from torch.optim import SGD
from torch.optim import Adagrad as torch_adagrad
from torch.optim import RMSprop as torch_rmsprop
from torch.optim import Adadelta as torch_adadelta
from torch.optim import Adam as torch_adam

def optim_f(w):
    x = torch.tensor([0.2, 2], dtype=torch.float)
    return torch.sum(x * w ** 2)

def optim_g(w, b):
    x = torch.tensor([0.2, 2], dtype=torch.float)
    return torch.sum(x * w + b)

opt_checker_1 = SimpleNamespace(f=optim_f, 
                                params=[torch.tensor([-6, 2], dtype=torch.float, requires_grad=True)])
opt_checker_2 = SimpleNamespace(f=optim_g, 
                               params=[torch.tensor([-6, 2], dtype=torch.float, requires_grad=True),
                                       torch.tensor([1, -1], dtype=torch.float, requires_grad=True)])


test_params = {'Momentum': {'torch_cls': SGD,
                            'torch_params': {'lr': 0.1, 'momentum': 0.9},
                            'params': {'learning_rate': 0.1, 'gamma': 0.9}},
              'Adagrad': {'torch_cls': torch_adagrad,
                            'torch_params': {'lr': 0.5, 'eps': 1e-8},
                            'params': {'learning_rate': 0.5, 'epsilon': 1e-8}},
              'RMSProp': {'torch_cls': torch_rmsprop,
                          'torch_params': {'lr': 0.5, 'alpha': 0.9, 'eps': 1e-08,},
                          'params': {'learning_rate': 0.5, 'gamma': 0.9, 'epsilon': 1e-8}},
              'Adadelta': {'torch_cls': torch_adadelta,
                          'torch_params': {'rho': 0.9, 'eps': 1e-1},
                          'params': {'gamma': 0.9, 'epsilon': 1e-1}},
              'Adam': {'torch_cls': torch_adam,
                          'torch_params': {'lr': 0.5, 'betas': (0.9, 0.999), 'eps': 1e-08},
                          'params': {'learning_rate': 0.5, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}}}

def test_optimizer(optim_cls, num_steps=10):
               
    test_dict = test_params[ optim_cls.__name__]
    
    for ns in [opt_checker_1, opt_checker_2]:
        
        torch_params = [p.clone().detach().requires_grad_(True) for p in ns.params]
        torch_opt = test_dict['torch_cls'](torch_params, **test_dict['torch_params'])
        for _ in range(num_steps):
            
            torch_opt.zero_grad()
            
            loss = ns.f(*torch_params)
            loss.backward()
            torch_opt.step()
        
        params = [p.clone().detach().requires_grad_(True) for p in ns.params]
        opt = optim_cls(params, **test_dict['params'])

        for _ in range(num_steps):
        
            opt.zero_grad()
        
            loss = ns.f(*params)
            loss.backward()
            opt.step()
            
        for p, tp in zip(params, torch_params):
            assert torch.allclose(p, tp)

            
def test_droput(dropout_cls):

    drop = dropout_cls(0.5)
    drop.train()
    x = torch.randn(10, 30)
    out = drop(x)

    for row, orig_row in zip(out, x):
        zeros_in_row = torch.where(row == 0.)[0]
        non_zeros_in_row = torch.where(row != 0.)[0]
        non_zeros_scaled = (row[non_zeros_in_row] == 2 * orig_row[non_zeros_in_row]).all()
        assert len(zeros_in_row) > 0 and len(zeros_in_row) < len(row) and non_zeros_scaled

    drop_eval = dropout_cls(0.5)
    drop_eval.eval()
    x = torch.randn(10, 30)
    out_eval = drop_eval(x)

    for row in out_eval:
        zeros_in_row = len(torch.where(row == 0.)[0]) 
        assert zeros_in_row == 0
        

def test_bn(bn_cls):

    torch.manual_seed(42)
    bn = bn_cls(num_features=100)

    opt = torch.optim.SGD(bn.parameters(), lr=0.1)

    bn.train()
    x = torch.rand(20, 100)
    out = bn(x)

    assert out.mean().abs().item() < 1e-4
    assert abs(out.var().item() - 1) < 1e-1

    assert (bn.sigma != 1).all()
    assert (bn.mu != 1).all()

    loss = 1 - out.mean()
    loss.backward()
    opt.step()

    assert (bn.beta != 0).all()
    
    n_steps = 10

    for i in range(n_steps):
        x = torch.rand(20, 100)
        out = bn(x)
        loss = 1 - out.mean()
        loss.backward()
        opt.step()


    torch.manual_seed(43)
    test_x = torch.randn(20, 100)
    bn.eval()
    test_out = bn(test_x)

    assert abs(test_out.mean() + 0.5) < 1e-1