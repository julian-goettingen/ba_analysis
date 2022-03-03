import sys
import numpy as np
ln = np.log
import analyze_helper as anh
import importlib
importlib.reload(anh)
from glob import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import pandas as pd
from functools import partial
import numpy.ma as npma
import scipy.optimize
import itertools as it
import collections
import matplotlib.colors
import scipy.stats
import pickle


#do_plot can be False for "never", True for "always" and None for "maybe"
def import_set(pattern,f=False, do_plot=None):
    global all_singles, imported_patterns
    if not f and pattern in imported_patterns:
        print(pattern, " already imported")
        return
    new = list((map(lambda s : anh.SingleSimul(s,fitres_cache_file, do_plot), sorted(anh.filter_successful(glob("data/"+pattern)+glob("data/analysis/"+pattern))))))
    print(len(new))
    #if len(new) == 0:
    #    raise ValueError("no simuls in set ", pattern)
    all_singles.extend(new)
    print("imported ", len(new), " dirs for ", pattern)
    imported_patterns.append(pattern)


all_singles = []
fitres_cache_file = "final_fitres.json"
