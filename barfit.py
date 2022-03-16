import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import scipy.stats as stats


########### MANUAL PARAMETERS ########
#----| moved to function args of data_deriv_barfit
#fix_at_bottom=False
#bar_thickness = 0.075
######################################

def local_fun(xpoint, xdat, ydat, fun, binsize, bar_thickness, fix_at_bottom, min_points):

    xl = xpoint-binsize/2
    xh = xpoint + (xpoint - xl) #symmetric around xpoint

    il = np.searchsorted(xdat, xl)
    ih = np.searchsorted(xdat, xh)

    if ih-il < min_points:
        return np.NaN

    res = fun(ydat[il:ih], bar_thickness, fix_at_bottom)

    return res


def iqrmidpoint(arr):
    l,h = np.quantile(arr, [0.25,0.75])
    return (l+h)/2

def trimmean1(arr):
    return stats.trim_mean(arr, 0.1)
def trimmean2(arr):
    return stats.trim_mean(arr, 0.2)
def trimmean3(arr):
    return stats.trim_mean(arr, 0.3)

def modus(arr):
    hist_vals, hist_bins = np.histogram(arr,bins=20)
    mi = np.argmax(hist_vals)
    return (hist_bins[mi]+hist_bins[mi+1])/2

def smooth_modus(arr):
    hist_vals, hist_bins = np.histogram(arr,bins=50)
    smooth_hist_vals = np.convolve([1,2,3,2,1],hist_vals,mode="same")
    mi = np.argmax(smooth_hist_vals)
    return (hist_bins[mi]+hist_bins[mi+1])/2



def modus_after_trim(arr):

    trimmed = stats.trimboth(arr, 0.2)
    return modus(trimmed)


def density_threshold(arr, bar_thickness, fix_at_bottom):

    # use d (chosen by human) as the thickness of the bar
    # place this bar to cover the most points

    d = bar_thickness
    sarr = np.sort(arr)
    possible_start_points = np.linspace(sarr[0], sarr[-1]-d,300)
    max_dpoint_num = 0
    max_dpoint_start_index = 0
    max_dpoint_end_index = 0

    if fix_at_bottom:
        i = 0
        #print(sarr.shape, possible_start_points.shape)
        #print(possible_start_points[i], possible_start_points[i]+d)
        hi = np.searchsorted(sarr, possible_start_points[i]+d, "left")
        li = np.searchsorted(sarr, possible_start_points[i], "left")
        #print(sarr[li], sarr[hi])
        n_points = hi-li
        if n_points > max_dpoint_num:
            max_dpoint_num = n_points
            max_dpoint_start_index = li
            max_dpoint_end_index = hi
            #print(f"updated range to ix=[{max_dpoint_start_index},{max_dpoint_end_index}] to include {n_points}|||\trange={sarr[max_dpoint_start_index]-sarr[max_dpoint_end_index]}")
    else:
        for i in range(len(possible_start_points)):
            #print(sarr.shape, possible_start_points.shape)
            #print(possible_start_points[i], possible_start_points[i]+d)
            hi = np.searchsorted(sarr, possible_start_points[i]+d, "left")
            li = np.searchsorted(sarr, possible_start_points[i], "left")
            #print(sarr[li], sarr[hi])
            n_points = hi-li
            if n_points > max_dpoint_num:
                max_dpoint_num = n_points
                max_dpoint_start_index = li
                max_dpoint_end_index = hi
                #print(f"updated range to ix=[{max_dpoint_start_index},{max_dpoint_end_index}] to include {n_points}|||\trange={sarr[max_dpoint_start_index]-sarr[max_dpoint_end_index]}")

    return sarr[max_dpoint_start_index: max_dpoint_end_index]

def density_threshold_mean(arr, bt, fix_at_bottom):
    return np.mean(density_threshold(arr,bt, fix_at_bottom))

def density_threshold_min(arr, bt, fix_at_bottom):
    return np.min(density_threshold(arr, bt, fix_at_bottom))

def density_threshold_max(arr, bt, fix_at_bottom):
    return np.max(density_threshold(arr, bt, fix_at_bottom))

def density_threshold_median(arr, bt, fix_at_bottom):
    return np.median(density_threshold(arr, bt, fix_at_bottom))

def density_threshold_modus(arr, bt, fix_at_bottom):
    return modus(density_threshold(arr, bt, fix_at_bottom))

def density_threshold_iqrmidpoint(arr, bt, fix_at_bottom):
    return iqrmidpoint(density_threshold(arr, bt, fix_at_bottom))

def density_threshold_minmaxmidpoint(arr, bt, fix_at_bottom):
    a = density_threshold(arr,bt, fix_at_bottom)
    return ( np.min(a) + np.max(a) ) / 2

def density_threshold_smooth_modus(arr, bt):
    return smooth_modus(density_threshold(arr, bt))

#dthreshfuncs = [name for name in globals().keys() if name.startswith("density_threshold_")]

#for func_name in ["np.median", "np.mean", "iqrmidpoint", "trimmean1", "trimmean2", "trimmean3","modus","modus_after_trim", "smooth_modus"]:
#for func_name in dthreshfuncs:
#    func = eval(func_name)
#    yplot = np.array([local_fun(xp, xdat,ydat, func) for xp in xsample])
#    lr = sklearn.linear_model.LinearRegression()
#    lr.fit(xsample.reshape(-1,1), yplot.reshape(-1,1))
#    plt.plot(xsample, yplot, label=f"{func_name} :: a={round(lr.coef_[0][0],3)}")

def data_deriv_barfit(xpoint,xdat,ydat,*,min_vals=200,min_points_in_bin=5, fix_at_bottom=False, func=density_threshold_median,area=1.0,binsize=0.2,bar_thickness=0.075,show=False):


    # min-vals unused. Slice xdat to see how many values are being used
    xsample = np.linspace(xpoint-area/2, xpoint+area/2, 50)

    ysample = np.array([local_fun(xp, xdat, ydat, func, bar_thickness, binsize,fix_at_bottom,min_points_in_bin) for xp in xsample])
    y_finite_filter = ~np.isnan(ysample)

    xsample = xsample[y_finite_filter]
    ysample = ysample[y_finite_filter]

    lr = sklearn.linear_model.LinearRegression()
    lr.fit(xsample.reshape(-1,1), ysample.reshape(-1,1))
    slope = lr.coef_[0][0]

    if show:
        plt.plot(xsample, ysample, label=f"a={slope}")
        plt.scatter(xdat,ydat, color="grey", alpha=0.5, s=3)
        plt.xlim(xpoint-area/2, xpoint+area/2)
        plt.legend()
        plt.title("x="+str(xpoint))
        plt.show()

    return slope
