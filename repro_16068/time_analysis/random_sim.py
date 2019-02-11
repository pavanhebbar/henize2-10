"""Python program to simulate rendom realizations and get a dist of chi^2"""

import sys
import copy
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def get_rand(low, high, num):
    """Generate an array of random numbers

    Inputs:
    low - Lower limit on the random numbers
    high - Higher limit on the random numbers
    num - Number of random numbers to be generated
    """
    return np.random.uniform(low, high, num)


def get_chi(func_parameters, func, xdata, ydata):
    """Calculate the chi^2 value for the given distribution.

    The model/expected y-values are calculated from the xdata and the input
    model function and its parameters.
    Inputs:
    func_parameters: Parameters of the function in the form of the array
    func: model function
    xdata: x values
    ydata: y values
    """
    ymodel = func(xdata, func_parameters)
    return np.sum((ydata-ymodel)**2/(1+(ydata+0.75)**0.5)**2)


def func_const(xdata, const):
    """Return the constant value of the function."""
    if isinstance(const, (list, np.ndarray)):
        const = const[0]
    return np.ones(len(xdata))*const


def func_sine(xdata, amp, period=0, x_0=0, const=0):
    """Return a sine function.

    Inputs:
    xdata: x values
    amp: Amplitude
    period: time period of sine function in s
    x0: x value when phase is zero
    const: const value
    """
    if isinstance(amp, (list, np.ndarray)):
        period = amp[1]
        x_0 = amp[2]
        const = amp[3]
        amp = amp[0]
    return amp*np.sin(2*np.pi*1.0/period*(xdata-x_0))+const


def fit_funcs(xdata, ydata):
    """Return the best fitting constant and sine models.

    Inputs:
    xdata: centres of time bins
    ydata: counts in each time bin
    """
    binsize = xdata[1]-xdata[0]
    xlen = xdata[-1]-xdata[0]
    # Fitting a constant function
    const_val, const_err = curve_fit(func_const, xdata, ydata,
                                     p0=np.mean(ydata),
                                     sigma=1+(ydata+0.75)**0.5,
                                     absolute_sigma=True)
    chi_const = get_chi(const_val, func_const, xdata, ydata)

    # Fitting a sine function
    sin_min = [0., 3*xlen, 0., 0.]
    sin_err = [0., 0., 0., 0.]
    chi_sin = get_chi(sin_min, func_sine, xdata, ydata)
    lboun = [0., binsize*2, -1.0*np.inf, 0.]
    uboun = [np.max(ydata), 2.*xlen, np.inf, np.max(ydata)]
    test_periods = xlen*1.0/np.arange(1.0, 1.0*int(len(xdata)/2.+0.5), 1.)
    test_amps = [0., 1., 2., 3., 4., 5.]
    for period in test_periods:
        for amp in test_amps:
            try:
                sin_val, sin_e = curve_fit(
                    func_sine, xdata, ydata, p0=[amp, period, np.min(xdata),
                                                 np.mean(ydata)],
                    bounds=(lboun, uboun), sigma=1+(ydata+0.75)**0.5,
                    absolute_sigma=True, maxfev=10000000)
            except ValueError:
                sin_val = copy.copy(sin_min)
                sin_err = copy.copy(sin_err)
            chi_temp = get_chi(sin_val, func_sine, xdata, ydata)
            if chi_temp < chi_sin:
                chi_sin = chi_temp
                sin_min = sin_val.copy()
                sin_err = sin_e.copy()
    sin_err = np.sqrt(np.diag(sin_err))
    return const_val, const_err, chi_const, sin_min, sin_err, chi_sin


def monte_carlo_step(opt=1, optparam=[]):
    """Perform monte_carlo simulations.

    Inputs:
    opt: option number - 1 or 2
        1 - A random uniform list of event times in created between xlim limits
        2. xdata is considered as the central bin values. y values would be the
        random created using possion statistics which are considered as the
        counts in each bin
    optparam = [xmin, xmax, len, binsize] if opt = 1
               [xvals, rate] if opt = 2
    """
    if opt == 1:
        if not len(optparam) == 4:
            raise KeyError('optparam = [xmin, xmax, len, binsize] if opt = 1')
        xdata = np.random.uniform(optparam[0], optparam[1], optparam[2])
        nbins = int((np.max(xdata)-np.min(xdata))/optparam[3]) + 1
        xvals = np.linspace(np.min(xdata), np.max(xdata), nbins)
        yvals, xvals = np.histogram(xdata, bins=xvals)
        xvals = 0.5*(xvals[:-1]+xvals[1:])
    elif opt == 2:
        if not len(optparam) == 2:
            raise KeyError('optparam = [xvals, rate] if opt = 2')
        xvals = optparam[0].copy()
        yvals = np.random.poisson(optparam[1], len(xvals))
    else:
        raise ValueError('opt can only take values 1 or 2')
    func_fitting = fit_funcs(xvals, yvals)
    return func_fitting, xvals, yvals


def monte_carlo(ntimes, opt=1, optparam=[]):
    """Perform Monte Carlo Simulations.

    Inputs:
    ntime - Number of times to perform the simulation
    opt, optparam - Parameters for monte-carlo step
    """
    const_params = np.zeros((ntimes, 3))
    sin_params = np.zeros((ntimes, 9))
    for i in range(ntimes):
        func_fits = monte_carlo_step(opt, optparam)[0]
        const_params[i, :] = func_fits[:3]
        sin_params[i, [0, 2, 4, 6]] = func_fits[3]
        sin_params[i, [1, 3, 5, 7]] = func_fits[4]
        sin_params[i, 8] = func_fits[5]
        if i % 10 == 0:
            print(i)
    return const_params, sin_params


def plotfig(name, xlabel, ylabel, xdatas, ydatas, legends=[]):
    """Plot figure"""
    legends_null = ['none']*len(xdatas)
    if legends == []:
        legends = legends_null
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i, xdata in enumerate(xdatas):
        plt.step(xdata, ydatas[i], label=legends[i], where='mid')
    if not legends == legends_null:
        plt.legend()
    plt.savefig(name)
    plt.show()


def get_stats(const_params, sin_params):
    """Return information on the constant and the sine fit params

    Inputs:
    const_params - Constant value, constant error and chi
    sin_params - [amp, amp_err, period, per_err, x_0, x_0_err, const,
                  const_err, chi]
    """
    # Comparing the chi sq values of the constant and the sine fit
    chi_bins = np.linspace(
        min(np.min(const_params[:, -1]), np.min(sin_params[:, -1])),
        max(np.max(const_params[:, -1]), np.max(sin_params[:, -1])),
        11)
    hist_chi_const, chi_bins = np.histogram(const_params[:, -1], bins=chi_bins)
    hist_chi_sin, chi_bins = np.histogram(sin_params[:, -1], bins=chi_bins)
    chi_vals = 0.5*(chi_bins[:-1] + chi_bins[1:])
    plotfig('chi_vals_2.png', r'$\chi$', r'$f$', [chi_vals, chi_vals],
            [hist_chi_const*1.0/len(sin_params),
             hist_chi_sin*1.0/len(sin_params)], ['Constant fit', 'Sine fit'])

    # Comparing the AIC values to account for the diff in dof
    aic_bins = np.linspace(
        min(np.min(const_params[:, -1]+2), np.min(sin_params[:, -1]+8)),
        max(np.max(const_params[:, -1]+2), np.max(sin_params[:, -1]+8)),
        11)
    his_aic_cons, aic_bins = np.histogram(const_params[:, -1]+2, bins=aic_bins)
    hist_aic_sin, aic_bins = np.histogram(sin_params[:, -1]+8, bins=aic_bins)
    aic_vals = 0.5*(aic_bins[:-1] + aic_bins[1:])
    plotfig('aic_vals_2.png', 'AIC value', r'$f$', [aic_vals, aic_vals],
            [his_aic_cons*1.0/len(sin_params),
             hist_aic_sin*1.0/len(sin_params)], ['Constant fit', 'Sine fit'])

    # Get the confidence in the amplitude in terms of sigma
    conf = sin_params[:, 0]*1.0/sin_params[:, 1]
    hist_conf, conf_bins = np.histogram(conf, bins=11)
    conf_vals = 0.5*(conf_bins[1:] + conf_bins[:-1])
    plotfig('amp_sin_2.png', 'Amp/Amp_err', r'$f$', [conf_vals],
            [hist_conf/len(sin_params)])


def main(opt, ntimes):
    """Run the program"""
    opt = int(opt)
    ntimes = int(ntimes)
    if opt == 1:
        xlim = [0, 162226.01292]
        length = 179
        binsize = 5000
        const_params, sin_params = monte_carlo(
                ntimes, 1, [xlim[0], xlim[1], length, binsize])
    elif opt == 2:
        xvals = np.arange(0, 162226.01292+5000, 5000)
        meancounts = 5.659421112195175
        const_params, sin_params = monte_carlo(
                ntimes, 2, [xvals, meancounts])
    np.savetxt('const_params_'+str(opt+ntimes)+'.txt', const_params)
    np.savetxt('sin_params_'+str(opt+ntimes)+'.txt', sin_params)
    get_stats(const_params, sin_params)


if __name__ == '__main__':
    print(len(sys.argv))
    main(sys.argv[1], sys.argv[2])
