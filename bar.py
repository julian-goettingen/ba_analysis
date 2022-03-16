import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import scipy.stats as stats


########### MANUAL PARAMETERS ########
fix_at_bottom=False
bar_thickness = 0.075
######################################


#xdat is sorted in ascending order, ydat is sorted so that it matches xdat
xdat = np.load("xdat.npy")
xdat = xdat.flatten()
ydat = np.load("ydat.npy")
ydat = ydat.flatten()

def local_fun(xpoint, xdat, ydat, fun):

    xl = xpoint-0.1
    xh = xpoint + (xpoint - xl)

    il = np.searchsorted(xdat, xl)
    ih = np.searchsorted(xdat, xh)

    res = fun(ydat[il:ih])

    return res

xsample = np.linspace(np.min(xdat), np.max(xdat),100)

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


def density_threshold(arr):

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

def density_threshold_mean(arr):
    return np.mean(density_threshold(arr))

def density_threshold_min(arr):
    return np.min(density_threshold(arr))

def density_threshold_max(arr):
    return np.max(density_threshold(arr))

def density_threshold_median(arr):
    return np.median(density_threshold(arr))

def density_threshold_modus(arr):
    return modus(density_threshold(arr))

def density_threshold_iqrmidpoint(arr):
    return iqrmidpoint(density_threshold(arr))

def density_threshold_minmaxmidpoint(arr):
    a = density_threshold(arr)
    return ( np.min(a) + np.max(a) ) / 2

def density_threshold_smooth_modus(arr):
    return smooth_modus(density_threshold(arr))

dthreshfuncs = [name for name in globals().keys() if name.startswith("density_threshold_")]

#for func_name in ["np.median", "np.mean", "iqrmidpoint", "trimmean1", "trimmean2", "trimmean3","modus","modus_after_trim", "smooth_modus"]:
for func_name in dthreshfuncs:
    func = eval(func_name)
    yplot = np.array([local_fun(xp, xdat,ydat, func) for xp in xsample])
    lr = sklearn.linear_model.LinearRegression()
    lr.fit(xsample.reshape(-1,1), yplot.reshape(-1,1))
    plt.plot(xsample, yplot, label=f"{func_name} :: a={round(lr.coef_[0][0],3)}")




# fit through all data
#lr = sklearn.linear_model.LinearRegression(fit_intercept=False)
#lr.fit(xdat.reshape(-1,1),ydat.reshape(-1,1))

#slope = lr.coef_[0]
#plt.plot(xdat, slope*xdat)

plt.scatter(xdat, ydat, color="grey", s=3, alpha=0.3)
#plt.ylim(-0.1,0.1)
plt.legend(bbox_to_anchor=(1.1,-0.1))
plt.tight_layout()
plt.show()
