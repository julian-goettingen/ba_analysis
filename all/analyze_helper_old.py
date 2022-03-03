import numpy as np
import random
ln = np.log
import json
from enum import IntEnum
from time import sleep
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict
import scipy.optimize as opti
import scipy.stats
import functools
from dataclasses import dataclass
import typing
import itertools as it
from glob import glob
import statistics as stat
import yaml
import pandas as pd

def get_visc_from_dir(d):

    with open(d+"/core_options.json", "r") as core_opts:
        opts = json.load(core_opts)
    return float(opts["viscosity"])

def filter_successful(simuls):
    return list(filter(lambda d : os.path.isfile(d+"/results.csv") and get_visc_from_dir(d)<1, simuls))


def all_the_same(ls):
    return all(x==ls[0] for x in ls)

# xgrid,ygrid describe gridcells to calculate avg on, scatter_data (x,y,z) describes the scattered data
def local_avg(xgrid,ygrid,scatter_x, scatter_y, scatter_z):

    res = np.full((len(xgrid), len(ygrid)), fill_value=np.NaN)





@dataclass
class ArrWithStats():

    arr : np.ndarray
    mean : float
    var : float
    stddev : float
    perc10 : float
    q1 : float
    median : float
    q3 : float
    perc90 : float
    maxval : float
    minval : float

    def __init__(self, arr):
        self.arr = arr
        self.mean = np.mean(arr)
        self.stddev = np.std(arr)
        self.var = self.stddev**2
        quantiles = [0.10, 0.25, 0.5, 0.75, 0.90]
        quants = np.quantile(arr, quantiles)
        self.perc10= quants[0]
        self.q1 = quants[1]
        self.median = quants[2]
        self.q3 = quants[3]
        self.perc90 = quants[4]
        self.maxval = np.max(arr)
        self.minval = np.min(arr)

    def get(self, s):

        if s in self.__dict__:
            return self.__dict__[s]
        elif s=="iqr":
            return [self.q1, self.q3]
        elif s=="percs":
            return [self.perc10, self.prc90]
        elif s=="minmax":
            return [self.minval, self.maxval]
            
def make_multi_simul_if_fittable(single_sims, name):

    for s in single_sims:
        if s.fitres is None:
            return None
    return MultiSimul(single_sims, name)


class MultiSimul():

    def __init__(self, simuls, name):
        # simuls is a list of SingleSimuls
        self.sims = list(simuls) # ensure it is ordered and permanent
        assert(len(self.sims) > 0)

        self.name = name

        self.categ_list = np.array([s.categ.value for s in self.sims])
        if all_the_same(self.categ_list):
            self.categ = self.categ_list[0]
        else:
            # if everything is dipper and grower, use the mean
            if np.all(np.logical_or(self.categ_list==3,self.categ_list==4)):
                self.categ = np.mean(self.categ_list)
            else:
                self.categ = None

        for obs in ["mach", "Re", "Re_mesh", "g1", "g10", "g100", "urms"]:
            obs_arr = np.array([s.__dict__[obs] for s in self.sims])
            self.__dict__[obs] = ArrWithStats(obs_arr)

        for obs in ["a", "b", "rsq"]:
            obs_arr = np.array([s.fitres.__dict__[obs] for s in self.sims])
            self.__dict__[obs] = ArrWithStats(obs_arr)

        for param in ["viscosity", "forcing_magnitude", "relhel"]:
            vals = [float(s.opts[param]) for s in self.sims]
            if len(vals)==0:
                raise ValueError(f"could not find {param} to make ms {self.name} (from dicts like {self.sims[0].opts[param]}")
            if all_the_same(vals):
                self.__dict__[param] = vals[0]
            else:
                self.__dict__[param+"_arr"] = np.array(vals)
            
class Category(IntEnum):

    shrinker = 1
    stagnator = 2
    dipper = 3
    grower = 4

categ_count = defaultdict(int)

class SingleSimul():

    def __init__(self, name):
        self.dirname = name
        self.results_df = pd.read_csv(self.dirname + "/results.csv")
        with open(self.dirname+"/core_options.json", "r") as f:
            self.opts = json.load(f)

        # time averaged quantities
        self.Re = np.mean(self.results_df["Re"])
        self.Re_var = np.mean(self.results_df["Re_var"])
        self.urms = np.mean(self.results_df["urms"])
        self.urms_var = np.mean(self.results_df["urms_var"])
        self.mach = np.mean(self.results_df["mach"])
        self.mach_var= np.mean(self.results_df["mach_var"])
        self.Re_mesh = np.mean(self.results_df["Re_mesh"])
        self.Re_mesh_var = np.mean(self.results_df["Re_mesh_var"])

        # quantities as timeseries
        self.var = self.results_df["var"].to_numpy(copy=True)

        self.relvar = self.var/self.var[0]
        #self.relvar = self.var/np.min(self.var)

        #print("relvar: ", self.relvar)
        self.real_time = self.results_df["real_time"].to_numpy(copy=True)
        self.timestep = self.results_df["timestep"].to_numpy(copy=True)

        self.fitres = self.calc_fit()

        if self.fitres is not None:

            # growth-factors (from fitres)
            self.g1 = self.fitres(0.1)
            self.g10 = self.fitres(1)
            self.g100 = self.fitres(10)

        assert hasattr(self, "categ")

        self.show_maybe()
    

    def show_maybe(self):

        show_prob = 0.01
        if categ_count[self.categ] <= 3:
            show_prob = 1
        if self.min_var_index > 5 and self.categ==Category.dipper:
            show_prob = 0.1

        if random.uniform(0,1) < show_prob:
            self.show()



    def show(self):

        plt.scatter(
                self.real_time - self.real_time[0],
                self.relvar - 1,
                label="data"
                )

        s, e = self.fit_start_index, self.fit_end_index
        i = self.min_var_index

        if self.fitres is not None:
            t =  self.real_time - self.real_time[i]
            plt.plot(
                    (t + self.real_time[i] - self.real_time[0])[s:e],
                    self.fitres(t[s:e]),
                    label="exponential fit",
                    color="orange"
                    )
                
        plt.title(f"mach={self.mach} Re={self.Re} ({self.categ})")
        plt.legend()
        plt.show()
        """
        for k,v in self.__dict__.items():
            try:
                vlist = list(v)
            except:
                print(f"{k}: {v}")
            else:
                if len(vlist) < 5:
                    print(f"{k}: {vlist}")
                print(f"{k}: (len {len(vlist)}) {vlist[:3]}...{vlist[-3:]}")
        """
                    
                    

    def set_categ(self, categ):

        assert not hasattr(self, "categ")
        assert isinstance(categ, Category)
        self.categ = categ

        global categ_count
        categ_count[categ] += 1


    def calc_fit(self):

        var = self.relvar
        t = self.real_time - self.real_time[0]

        absolute_minimum = 1

        # start fitting exp from minimum, not beginning (to treat the dips)
        minvarind = max(np.argmin(var),absolute_minimum)
        #minvarind = np.argmin(var) + absolute_minimum

        #cutoff = None
        cutoff = min(len(var)-1, minvarind+20)

        #renormalization around minimum, which is the starting value for the fitting
        i = minvarind
        var /= var[i]
        t -= t[i]
        
        s = minvarind
        e = cutoff
        valid = (var[s:e] > 1)

        self.fit_start_index = s
        self.fit_end_index = e
        self.min_var_index = i




        if len(valid) < 5:
            print(f"cant fit with only {len(valid)} datapoints (dir is {self.dirname}, Re is {self.Re})")
            #plt.plot(t, var)
            #plt.show()

            if np.all(var[1:] < var[:-1]):
                # the variance is decreasing with time
                self.set_categ(Category.shrinker)
            else:
                self.set_categ(Category.stagnator)

            return None

        if minvarind == absolute_minimum:
            self.set_categ(Category.grower)
        else:
            self.set_categ(Category.dipper)

        lnt = ln(t[s:e][valid])
        lnvar = ln(var[s:e][valid] -1)
        
        assert(np.all(~np.isnan(lnt)))
        assert(np.all(~np.isnan(lnvar)))

        linres = scipy.stats.linregress(lnt,lnvar)
        a, lnb = linres.slope, linres.intercept
        a_err = linres.stderr
        b = np.exp(lnb)
        rsq = linres.rvalue**2
        fitres = Fitres(a,b,rsq,a_err,self.dirname)

        show=False
        if show:
            print(fitres.a)
            plt.plot(lnt, lnvar, label="data")
            plt.plot(lnt, lnb+ a*lnt, linestyle="--", label="fit", color="orange")
            #plt.plot(lnt, ln(fitres(t[1:e][valid])), label="log of fitres")
            plt.legend()
            plt.show()
            
            plt.plot(t[s:e], var[s:e], label="measured")
            plt.plot(t[s:e], fitres(t[s:e])+1, linestyle="--", label="fit", color="orange")
            plt.legend()
            plt.show()
            print("============")
        return fitres


@dataclass
class Fitres():
    a: float
    b: float
    rsq: float
    a_err : float
    simul_name : str
    
    def __call__(self, t):
        return self.b*t**self.a
        
