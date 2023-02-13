import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import itertools
import pickle
import os

import lightgbm as lgb

from sklearn import datasets
from sklearn import svm
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors 


import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from gurobipy import GRB, Model, quicksum, multidict, tuplelist

# from cga2m_plus.cga2m import Constraint_GA2M
# from cga2m_plus.visualize import *

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import casadi
import pulp


import optuna
import time
import requests
import itertools
from PIL import Image

import sklearn.gaussian_process as gp
from numpy.linalg import svd, matrix_rank

from utility.constant import DOCKER, FAISS, ANNOY, NMSLIB, LOCAL, TSUBAME

try:
    import faiss
    print("近似近傍探索にfaissを使用します")
    ann_library = FAISS
    environment = DOCKER
    
except:
    try:
        from annoy import AnnoyIndex
        print("近似近傍探索にannoyを使用します")
        ann_library = ANNOY
        environment = LOCAL
    except:
        import nmslib
        print("近似近傍探索にnmslibを使用します")
        ann_library = NMSLIB
        environment = TSUBAME