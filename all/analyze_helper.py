import numpy as np
import re
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

_pgf_backend = False

if _pgf_backend:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

    def show_plot(name):
        assert name.endswith("_log.pgf") or name.endswith("_normal.pgf")
        plt.savefig(f"/home/julian/BA_plots/growth_plots/{name}")
        plt.close()

else: # no pgf, normal showing

    import matplotlib
    matplotlib.use("module://ipykernel.pylab.backend_inline")
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    def show_plot(name):
        plt.show()


inc_timesteps = [0.05, 0.1, 0.3, 0.5, 1.0, 1.5]
timestats = []

def get_timestep_stats():

    return ArrWithStats(np.array(timestats))

def get_visc_from_dir(d):

    with open(d+"/core_options.json", "r") as core_opts:
        opts = json.load(core_opts)
    return float(opts["viscosity"])

def get_mesh_Re_from_dir(d):

    results_df = pd.read_csv(d + "/results.csv")
    mesh_Re = np.mean(results_df["Re_mesh"])
        
    return mesh_Re

def filter_successful(simuls):
    # we need a results-file and we also filter out the ones where I accidentaly set ridiculously high viscosities
    # also filter out simulations with mesh_Re too high
    return list(filter(lambda d : os.path.isfile(d+"/results.csv") and get_visc_from_dir(d)<1 and get_mesh_Re_from_dir(d) < 10, simuls))



def all_the_same(ls):
    return all(x==ls[0] for x in ls)

def return_if_equal(ls):
    if all_the_same(ls):
        return ls[0]
    raise ValueError(str(ls)+" not equal!")

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


# get the first index in x where the curvature dips
def first_curvature_dip(x):

    #stc = [1,-2,1]
    stc = [-1/12, 4/3, -5/2, 4/3, -1/12]
    #stc = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
    
    stc = np.array(stc)
    diff2 = np.convolve(stc, x, "valid")

    diff2negative = (diff2 <= 0)

    # find the first negative point where the next one is also negative (to avoid flukes)
    for i in range(len(diff2negative)-1):
        if diff2negative[i] and diff2negative[i+1]:
            return i+len(stc)//2 # add half of the stencil-length to get indices in original array x

    return len(x) # no curvature dip -> give last valid index in x + 1 to mark end



class MultiSimul():

    def __init__(self, simuls, name):

        print("MultiSimul() is outdated and prob wont work")
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

        for obs in ["mach", "Re", "Re_mesh", "g1", "g10", "g100", "urms", "avg_dt"]:
            obs_arr = np.array([s.__dict__[obs] for s in self.sims])
            self.__dict__[obs] = ArrWithStats(obs_arr)

        for obs in ["a", "b", "rsq", "a_t", "total_err", "worst_point_err", "avg_point_err"]:
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

        self.inc = {}
        for t_real in inc_timesteps:
            self.inc[t_real] = avg_inc_after_real_time(self, t_real)
            
class Category(IntEnum):

    shrinker = 1
    stagnator = 2
    dipper = 3
    grower = 4

categ_count = defaultdict(int)

worst_total_err = 0.0
worst_err_per_point = 0.0
worst_rel_err_per_point = 0.0

class SingleSimul():

    def __init__(self, name, fitres_cache_file=None, do_plot=None):
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
        assert(self.relvar[0] - 1 < 1e-7)

        #print("relvar: ", self.relvar)
        self.real_time = self.results_df["real_time"].to_numpy(copy=True)
        self.timestep = self.results_df["timestep"].to_numpy(copy=True)

        # convenience quantities
        #self.t_eta = (2*np.pi / np.sqrt(15))/(self.urms * (self.opts["forcing_kmin"]+self.opts["forcing_kmax"])/2)
        # this t_eta is wrong, calc t_eta in notebook instead
        self.avg_dt = calc_avg_dt(self)


        #self.fitres = self.calc_fit()
        self.fitres = self.calc_transient_fit(fitres_cache_file)

        if self.fitres is not None and fitres_cache_file is not None:
            self.save_fitres(fitres_cache_file)

        self.fitres_cache_file = fitres_cache_file

        assert(self.fitres is not None)
        #if self.fitres is not None:

        print("<<<timestats:")
        eps = self.opts["forcing_magnitude"]*self.urms
        kolmo_time = (self.opts["viscosity"]/eps)**(1/2)
        end_time = self.real_time[-1] - self.real_time[0]
        self.end_time = end_time
        self.kolmo_time = kolmo_time
        simul_scaled_time_length = end_time / kolmo_time
        print(f"Kolmo-time={kolmo_time}")
        print(f"end_time={end_time}")
        print(f"end/kolmo={end_time/kolmo_time}")
        global timestats
        timestats.append(simul_scaled_time_length)
        print(f"{kolmo_time < end_time}")
        print(">>>")

        # growth-factors (from fitres)
        self.g1 = self.fitres(0.1*self.kolmo_time)
        self.g10 = self.fitres(1*self.kolmo_time)
        self.g20 = self.fitres(2*self.kolmo_time)
        self.g50 = self.fitres(5*self.kolmo_time)

        self.gfin = self.relvar[-1]


        #self.scaled_a = ln(self.fitres.a) * self.t_eta
        #self.scaled_at = ln(self.fitres.a_t) * self.t_eta




        assert hasattr(self, "categ")

        self.show_maybe(do_plot)
        #self.show()

    def save_fitres(self,fitname):

        assert fitname.endswith("json")
        assert self.fitres is not None

        d = self.fitres.__dict__
        d["classname"] = str(type(self.fitres))
        with open(self.dirname+"/"+fitname, "w") as f:
            json.dump(d, f)


    def try_load_fitres(self,fitname):

        assert not hasattr(self,"fitres")
        assert fitname.endswith(".json")

        fname = self.dirname+"/"+fitname
        if not os.path.isfile(fname):
            return False
        with open(fname, "r") as f:
            d = json.load(f)
        assert d["classname"] == str(type(TransientFitres(1,1,1,1,1,1,1,1,1)))
        assert d["simul_name"] == self.dirname
        del d["classname"]
        self.fitres = TransientFitres(**d)

        return True
    

    def show_maybe(self,do_plot):

        if do_plot is False:
            show_prob = 0.0
        elif do_plot is True:
            show_prob = 1.0
        elif do_plot is None:
            show_prob = 0.05
            global worst_total_err, worst_err_per_point

            s,e = self.fit_start_index, self.fit_end_index
            total_err = self.fitres.total_err / (np.max(self.relvar[s:e])-np.min(self.relvar[s:e]))
            if worst_total_err < total_err:
                print("new worst total error")
                show_prob = 1
                worst_total_err = total_err
            if worst_err_per_point < total_err / (self.fit_end_index-self.fit_start_index):
                print("new worst err per point")
                show_prob = 1
                worst_err_per_point = total_err / (self.fit_end_index-self.fit_start_index)
        else:
            raise ValueError("do_plot must be True, False or None, was ",str(do_plot))

    
        
        if random.uniform(0,1) < show_prob:
            self.show()



    def show(self):

        s, e = self.fit_start_index, self.fit_end_index
        i = self.min_var_index
        t =  self.real_time - self.real_time[s]

        for kind in ["normal", "log"]:

            if kind=="normal":
                relvar = self.relvar,
                fitres = self.fitres(t[s:e])
            elif kind=="log":
                relvar = ln(self.relvar),
                fitres = ln(self.fitres(t[s:e])) 
            plt.scatter(
                    self.real_time - self.real_time[0], # for data plotting, do the whole thing, not just the 
                    relvar,
                    label="data"
                    )
            if self.fitres is not None:
                plt.plot(
                        #(t + self.real_time[i] - self.real_time[0])[s:e],
                        t[s:e],
                        fitres,
                        label="double-exponential fit",
                        color="orange"
                        )
                    
            plt.xlabel("time")
            ylabel = "variance"
            if kind=="log":
                ylabel+= " (logarithmic)"
            plt.ylabel(ylabel)
            plt.title(f"mach={round(self.urms ,3)} Re={int(round(self.Re))}")
            plt.legend()

            #choose plot name
            cache_file_name = "uncached" if not self.fitres_cache_file else re.sub(r"\.json", "", self.fitres_cache_file)
            dirname = self.dirname.split("/")[-1]
            name = f"{dirname}_{cache_file_name}_{kind}.pgf"
            show_plot(name)

        print(self.fitres)
        print("=====")
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

    
    def calc_transient_fit(self, fitres_cache_file=None, weighted=True, stop_at_curv_dip=False):

        var = self.relvar
        t = self.real_time - self.real_time[0]

        # no max_real_time -> just set it ridiculously high
        max_real_time = 100000

        # fit from beginning to some maximum

        # dont fit over the maximum real time
        max_time_ind = np.searchsorted(t, max_real_time)
        

        if stop_at_curv_dip:
            # choose beginning as the first dip in curvature
            dip_index = first_curvature_dip(var)
        else:
            # just fit everything
            dip_index = len(var)

        self.fit_end_index = min(dip_index, max_time_ind)
        self.fit_start_index = 0 #beginning is always relevant
        s,e = self.fit_start_index , self.fit_end_index 
        self.min_var_index = np.argmin(var)

        if fitres_cache_file is not None:
            success = self.try_load_fitres(fitres_cache_file)
            if success:
                self.set_categ(Category.dipper)
                return self.fitres

        if not weighted:
            weight = 1
        else:
            scale = 10 #datapoints are weighted from 1 to scale
            weight = scale - (var[s:e]-np.min(var[s:e]))*(scale-1)*(np.max(var[s:e])-np.min(var[s:e]))
            weight *= weight

        def lsq(arg):

            a,b,at = arg
            return np.sum(weight*(var[s:e]-(b*a**t[s:e]+(1-b)*at**t[s:e]))**2)

        bounds = [
                [1,6], #a
                [0,1], #b
                [0,1], #at
                ]
        optires = opti.differential_evolution(lsq,bounds)
        assert(optires.success)
        a,b,at = optires.x
        total_err = optires.fun
        # relative point errors
        point_errs = (var[s:e] - (b*a**t[s:e]+(1-b)*at**t[s:e]))
        point_errs_scaled = (point_errs)/(np.max(var[s:e])-np.min(var[s:e]))
        worst_point_err = np.max(np.abs(point_errs_scaled))
        avg_point_err = np.mean(np.abs(point_errs_scaled))

        res = TransientFitres(a,b,at,np.NaN,np.NaN,total_err,worst_point_err, avg_point_err ,self.dirname)

        self.set_categ(Category.dipper)

        return res



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

        lnvar = ln(var[s:e][valid])
        
        assert(np.all(~np.isnan(lnvar)))

        """
        linres = scipy.stats.linregress(lnt,lnvar)
        a, lnb = linres.slope, linres.intercept
        a_err = linres.stderr
        b = np.exp(lnb)
        rsq = linres.rvalue**2
        fitres = ExpoFitres(a,b,rsq,a_err,self.dirname)
        """
        linres = scipy.stats.linregress(t[s:e][valid], lnvar)
        lna, lnb = linres.slope, linres.intercept
        lna_err = linres.stderr
        b = np.exp(lnb)
        a = np.exp(lna)
        rsq = linres.rvalue**2
        fitres = ExpoFitres(a,b,rsq,lna_err,self.dirname)

        return fitres


@dataclass
class ExpoFitres():
    a: float
    b: float
    rsq: float
    a_err : float
    simul_name : str
    
    def __call__(self, t):
        return self.b*self.a**t

        
@dataclass
class TransientFitres():

    a: float
    b: float
    a_t: float
    rsq: float
    a_err: float
    total_err: float
    worst_point_err: float
    avg_point_err: float
    simul_name: str

    def __call__(self, t):

        return self.b*self.a**t + (1-self.b)*self.a_t**t

def calc_avg_dt(single_simul):

    s = single_simul

    dt = s.real_time[1:] - s.real_time[:-1]

    return np.mean(dt)



def increase_after_real_time(single_simul, t_real):
    # calculate increase after t_real through interpolation
    
    assert(t_real > 0)
    
    s = single_simul
    t = s.real_time - s.real_time[0]
    
    pos = np.searchsorted(t,t_real)
    if pos == len(t):
        # not enough data for this point
        return np.NaN
    
    # we have t[pos-1] < t_real <= t[pos]
    
    #relative distance to the 2 points
    lo_dist = (t_real-t[pos-1])/(t[pos]-t[pos-1])
    hi_dist = 1-lo_dist
    
    # weighted average of the 2  points as value
    res = s.relvar[pos-1]*hi_dist + s.relvar[pos]*lo_dist
    
    return res

def avg_inc_after_real_time(multi_simul, t_real):
    
    assert(t_real>0)
    
    res = np.mean(np.array([increase_after_real_time(s,t_real) for s in multi_simul.sims]))
    return res

def weight_linear(dist, rmax=20):
    
    res = (rmax - dist)/rmax
    res = res if res > 0 else 0
    return res

def dist_weighted_cartesian(xy1, xy2, xweight, yweight):
    
    x1, y1 = xy1
    x2, y2 = xy2
    return np.sqrt((xweight*(x1-x2))**2 + (yweight*(y1-y2)**2))

class LocalAvg():
    
    def __init__(self, x,y,z,weights):
        
        assert(len(x)==len(y)==len(z))
        import scipy.spatial
        
        self.xmin = np.min(x)
        self.xmax = np.max(x)
        self.x = self.scale_x(x)
        self.ymin = np.min(y)
        self.ymax = np.max(y)
        self.y = self.scale_y(y)
        stacked = np.stack([self.x,self.y],axis=1)
        #print(stacked.shape)
        #print(stacked[:3,:])
        self.tree = scipy.spatial.KDTree(stacked, copy_data=True)
        self.z = z
        self.weights = weights
        #self.r = r
        
        #print(f"have data in range x=[{min(self.scale_back_x(self.x))},{max(self.scale_back_x(self.x))}]"
        #     f"y=[{min(self.scale_back_y(self.y))},{max(self.scale_back_y(self.y))}]")
    
    def scale_x(self, x):
        # requesting data outside range is allowed bc it can be near some real values
        #if not (np.all(np.logical_and(x>=self.xmin,x<=self.xmax))):
        #    raise ValueError(f"{x} is out of range for ({self.xmin}, {self.xmax})")
        return (x - self.xmin)/(self.xmax-self.xmin)
    
    def scale_back_x(self,x):
        return x*(self.xmax-self.xmin) + self.xmin
    
    def scale_y(self, y):
        return (y - self.ymin)/(self.ymax-self.ymin)
    
    def scale_back_y(self,y):
        return y*(self.ymax-self.ymin) + self.ymin
    
    # scalar x and y
    def calc(self, x,y,r, min_weight, max_r_inc,do_scale=True):
        
        #print("calc with ",x,y,r, min_weight, max_r_inc)
        
        assert(min_weight > 0)
        
        if do_scale:
            x = self.scale_x(x)
            y = self.scale_y(y)
        
        #print("scaled: ", x,y)
        
        
        points = self.tree.query_ball_point(np.array([x,y]), r=r)
        
        res = 0
        weight_sum = 0
        for xp, yp, zp, raw_weight in zip(self.x[points], self.y[points], self.z[points], self.weights[points]):
            #print(xp, yp, zp, raw_weight, "scaled back: ", self.scale_back_x(xp), self.scale_back_y(yp))
            dist_weight = (1 - np.sqrt(((xp-x))**2 + ((yp-y))**2)/r)**2
            #print(f"x queried: {x}, xp in data: {xp}")
            #print(f"alleged distance: (must be smaller than {r})", np.sqrt(((xp-x))**2 + ((yp-y))**2))
            weight = raw_weight*dist_weight
            weight_sum += weight
            res += zp*weight
            #print("-> ", zp, weight)
        if weight_sum <= min_weight:
            if max_r_inc == 0:
                return -11
            return self.calc(x,y,r*2,min_weight,max_r_inc-1,do_scale=False)
        #print("result: ", res/weight_sum)
        return res/weight_sum
    
    def calc_on_grid(self, x, y, r, min_weight, max_r_inc):
        
        res = np.full((len(x), len(y)), fill_value=np.NaN)
        
        for i in range(len(x)):
            for j in range(len(y)):
                res[i,j] = self.calc(x[i], y[j], r, min_weight=min_weight, max_r_inc=max_r_inc)
        return res
        

# xgrid,ygrid describe gridcells to calculate avg on, scatter_data (x,y,z) describes the scattered data
def local_avg(xgrid,ygrid,scatter_x, scatter_y, scatter_z, weigths, dist_fun, weight_fun):
    
    assert(len(scatter_x)==len(scatter_y)==len(scatter_z)==len(weights))

    res = np.full((len(xgrid), len(ygrid)), fill_value=np.NaN)
    
    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            # calculate res[i,j]
            
            xyg = (xgrid[i], ygrid[j])
            dist_weights = np.full_like(weights, fill_value=np.NaN)
            for k in range(len(scatter_x)):
                xd, yd = scatter_x[k], scatter_y[k]
                dist = dist_fun((xd,yd), xyg)
                dist_weights[k] = weight_fun(dist)
            res[i,j] = np.sum(dist_weights*weights*scatter_z)/np.sum(dist_weights*weights)
                
                
    return res

def has_nan(x):
    return np.any(np.isnan(x))

import datetime
print(f"successfully loaded {__file__} at {datetime.datetime.now()}") 





