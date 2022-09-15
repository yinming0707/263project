# Purpose of this source code is to plot a model against the experimental data for our pressure ODE

import numpy as np
from matplotlib import pyplot as plt
import csv
import scipy.interpolate
from scipy.optimize import curve_fit
# from Data_Vis_gradient_descent import *
from data import *
import scipy.stats as st
import random as rand


def ode_model(t, x, q, a, b, x0):
    ''' Return the derivative dx/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        x : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        x0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> ode_model(0, 1, 2, 3, 4, 5)
        22

    '''
    x0 = 25.16e+6
    if (x - x0) > .1e+6:
        return (a * q) - (b * ((x - x0) ** 2))
    else:
        return (a * q)

def ode_modelnotgood(t, x, q, a, b, x0):
    ''' Return the derivative dx/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        x : float
            Dependent variable.
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        x0 : float
            Ambient value of dependent variable.

        Returns:
        --------
        dxdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> ode_model(0, 1, 2, 3, 4, 5)
        22

    '''
    x0 = 25.16e+6
    if (x - x0) >0:
        return (a * q) - (b * ((x - x0) ** 2))
    else:
        return (a * q)

def solve_ode_qc(f, t, x0, pars, q, cap):
    ''' Solve an ODE numerically with constant q.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.
        cap : int
            how much to increase capacity by

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.
        c : array-like
            Cumulative gas leakage
    '''

    intt = t
    intx = np.zeros(len(intt))
    intx[0] = x0
    dt = intt[1] - intt[0]

    # # if you decide to sequentially increase doubled q and then decrease it in winter, use the code below
    # intt = np.round(t,1)
    # qt = np.array([q])
    # #qt = np.append(qt, q/2)
    # # by how much to increase and decrease q on average
    # qinc1 = 102472.6*cap
    # qinc2 = 290746.1*cap
    # qdec1 = 285535.6*cap
    # qdec2 = 265623.4*cap
    # mon2q = 111758.7*cap
    # mon8q = 165702*cap
    # for i in range(len(intt)-1):
    #     # current month
    #     mon = np.round(intt[i+1]%1,1)
    #     # decrease q in winter months between months 6 and 8
    #     if (mon >= 0.6) and (mon <= 0.8):
    #         if mon == 0.6:
    #             qt = np.append(qt, qdec1 - abs(qt[i]))
    #         else:
    #             qt = np.append(qt, qt[i] - qdec1)
    #     # start increasing q in winter months between months 8 and 9 (inclusive)
    #     elif (mon >= 0.8) and (mon < 1):
    #         qt = np.append(qt, qt[i] + qinc2)
    #     # increase q in non-winter months between months 0 and 3
    #     elif (mon >= 0.0) and (mon <= 0.3):
    #         if mon == 0.0:
    #             qt = np.append(qt, qinc1 + abs(qt[i]))
    #         else:
    #             qt = np.append(qt, qt[i] + qinc1)
    #     # start decreasing q as you approach winter between months 3 and 6
    #     elif (mon > 0.3) and (mon <= 0.6):
    #         qt = np.append(qt, qt[i] - qdec2)
    #     # repeat for months 2 and 8
    #     if mon == 0.2:
    #         qt[-1] += mon2q
    #     elif mon == 0.8:
    #         qt[-1] -= mon8q

    # if you decide to extrapolate q then use the sin function below
    qt = solve_sinusodial(intt, 634813.6, 216332.6, 6.28)[1] * cap
    qt[0] = q

    # # if you decide to get the average monthly mass flow then use the code below
    # qt = np.array([])
    # for i in range(len(intt)):
    #     # get the confidence interval out
    #     CI = get_average_q(np.round(intt[i]%1,3)%1)
    #     # if you want a random number between them uncomment code below
    #     avgQ = np.round(rand.uniform(CI[0],CI[1]),2)
    #     qt = np.append(qt, avgQ * cap)
    #
    #     # # if you want lower bound only uncomment code below
    #     # qt = np.append(qt, CI[0]*cap)
    #
    #     # # if you want upper bound only uncomment code below
    #     # qt = np.append(qt, CI[1]*cap)

    # # if you decide to use the last few years q then use code below
    # qt = load_massflow_data()[1][72:]*cap

    # # if you want to plot the predicted mass flow with the data, use code below
    # oqt, qv = load_massflow_data()
    # newfig, newax = plt.subplots(1,1)
    # newax.plot(oqt,qv,'-k',label='data')
    # newax.plot(intt,qt, '-r')

    for i in range(len(intt) - 1):
        # compute k1
        k1 = f(intt[i], intx[i], qt[i], *pars, x0)
        # compute euler value
        xe = intx[i] + dt * k1
        # compute k2
        k2 = f(intt[i + 1], xe, qt[i + 1], *pars, x0)
        # finally, compute improved euler value
        intx[i + 1] = intx[i] + (dt * (k1 + k2) / 2)

    return intt, intx / (10 ** 6)


def solve_ode(f, t0, t1, dt, x0, pars):
    ''' Solve an ODE numerically with non-constant q.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method.

        Function q(t) should be hard coded within this method. Create duplicates of
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''
    # setup initial value arrays and function q(t)
    intt = np.arange(t0, t1 + dt, dt)
    intx = np.zeros(len(intt))
    intx[0] = x0
    qt = interpolate_massflow(intt)

    for i in range(len(intt) - 1):
        # compute k1
        k1 = f(intt[i], intx[i], qt[i], *pars, x0)
        # compute euler value
        xe = intx[i] + dt * k1
        # compute k2
        k2 = f(intt[i + 1], xe, qt[i + 1], *pars, x0)
        # finally, compute improved euler value
        intx[i + 1] = intx[i] + ((dt / 2) * (k1 + k2))

    return intt, intx / (10 ** 6)


def solve_ode2(f, t, x0, pars):
    ''' Solve an ODE numerically, function only for curve fitting

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.
    '''

    # setup initial value arrays and function q(t)
    intt = t
    intx = np.zeros(len(intt))
    intx[0] = x0
    dt = intt[1] - intt[0]
    qt = interpolate_massflow(intt)

    for i in range(len(intt) - 1):
        # compute k1
        k1 = f(intt[i], intx[i], qt[i], *pars, x0)
        # compute euler value
        xe = intx[i] + dt * k1
        # compute k2
        k2 = f(intt[i + 1], xe, qt[i], *pars, x0)
        # finally, compute improved euler value
        intx[i + 1] = intx[i] + ((dt / 2) * (k1 + k2))

    return intt, intx / (10 ** 6)


def cumulator(p, po, thresh, t, t1, t2):
    '''

    All in Pa not MPa


    :param p: ARRAY of PRESSURES
    :param po: FLOAT AMbient pressure
    :param thresh: FLoat threshold amount (Pa)
    :param t: ARRAY of times
    :param t1: TIME TO BEGIN accumulating FLOAT
    :param t2: TIME TO END ACCUMulating

    :return: array of times, list of ACCUMULATED areas

    example:

    t, Atotal = cumulator(T2, 25.16e6, 0.1e6, t2, 2011.3, 2016)
    This will accumulate from 2011.4 to 2016.

    '''
    A = 0
    Atotal = []
    dt = t[1] - t[0]

    n1 = int(np.ceil((t1 - t[0]) / dt))
    n2 = int((t2 - t[0]) / dt)

    for i in range(n1, n2 + 1):
        if p[i] - po > thresh:
            A += (3.95e-6 * (p[i] - po) ** 2 * dt) / (10 ** 6)
            Atotal.append(A)
        else:
            Atotal.append(A)

    return t[n1:(n2 + 1)], Atotal


def load_massflow_data():
    ''' Returns time and massflow measurements from pressure data.

            Parameters:
            -----------
            none

            Returns:
            --------
            t : array-like
                Vector of times (year) at which measurements were taken.
            q : array-like
                Vector of massflow (kg) measurements during pressure data.
        '''
    # read the two text files
    mt, mfl = np.genfromtxt('gs_mass.txt', delimiter=',', skip_header=1).T

    # # replace the two months
    # yr = 2009.0
    # for i in range(len(mt)):
    #     mt[i] = yr
    #     yr += 0.1
    # mt = np.round(mt, 1)
    # print(mt)
    return mt, mfl


def load_pressure_data():
    ''' Returns time and pressure measurements from pressure data.

                Parameters:
                -----------
                none

                Returns:
                --------
                t : array-like
                    Vector of times (year) at which measurements were taken.
                q : array-like
                    Vector of pressure (MPa) measurements during pressure data.
    '''
    pt, pre = np.genfromtxt('gs_pres.txt', delimiter=',', skip_header=1).T

    return pt, pre * (10 ** 6)


def interpolate_massflow(t):
    ''' Return mass flow injection parameter q for pressure experiment.

        Parameters:
        -----------
        t : array-like
            Vector of times (year) at which to interpolate the injection source/q_source.

        Returns:
        --------
        q : array-like
            Mass Flow/q_source (kg) interpolated at times t.
    '''
    # suggested approach
    # experiment
    # use a built-in Python interpolation function

    time, qv = load_massflow_data()

    # setup interpolating function
    inter_y = scipy.interpolate.interp1d(time, qv)

    # finally, return interpolated q value
    return (inter_y(t))


def Tmodel(t, *pars):
    x0 = 25.16e+6
    pt, pr = solve_ode2(ode_model, t, x0, pars)
    return pr


def Tmodel_sinusodial(t, *pars):
    mt, mqv = solve_sinusodial(t, *pars)
    return mqv


def solve_sinusodial(t, a, s, f):
    '''computes the sin value at a particular time
    '''
    # f = 6.28
    return t, (a * np.sin(f * t) + s)


def get_average_q(month):
    ''' Computes a 95% confidence interval of mass flow (q) for a given month and finds the average value of that CI

    Parameter:
    ---------
    month : float
            the month to find the average q of
    '''
    # load mass flow data
    qt, qv = load_massflow_data()
    # find where the months are 'month'
    indmon = np.where(np.round(qt % 1, 3) == month)[0]
    # compute confidence interval
    CI = st.t.interval(alpha=0.95, df=len(qv[indmon]) - 1, loc=np.mean(qv[indmon]), scale=st.sem(qv[indmon]))
    print(CI)
    # finally return the lower bound or upper bound
    return (CI)


'''
def gd_get_opt_parameters(theta0, s0):
    #Performs gradient descent algorithm
    theta_all = [theta0]
    s_all = [s0]
    # iteration control
    N_max = 8
    N_it = 0
    # begin steepest descent iterations
    # exit when max iterations exceeded
    while N_it < N_max:
        # uncomment line below to implement line search (TASK FIVE)
        alpha = line_search(gaussian2D, theta_all[-1], s_all[-1])

        # update parameter vector
        # **uncomment and complete the command below**
        theta_next = step(theta_all[N_it], s_all[N_it], alpha)
        theta_all.append(theta_next)  # save parameter value for plotting

        # compute new direction for line search (thetas[-1]
        # **uncomment and complete the command below**
        s_next = obj_dir(gaussian2D, theta_all[N_it + 1])
        s_all.append(s_next)  # save search direction for plotting

        # compute magnitude of steepest descent direction for exit criteria
        N_it += 1
        # restart next iteration with values at end of previous iteration
        theta0 = 1. * theta_next
        s0 = 1. * s_next

    print('Optimum: ', round(theta_all[-1][0], 2), round(theta_all[-1][1], 2))
    print('Number of iterations needed: ', N_it)
'''


def predict_t(f, t0, t1, tp, dt, x0, xe, pars, sol):
    ''' Compute the pressure value in the future

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of observation
        tp : float
            Time value to predict to
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        xe : float
            Final pressure value
        pars : array-like
            List of parameters passed to ODE function f.
        sol : float
            The potential solution to the consent

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.
    '''

    # compute the time range
    intt = np.arange(t1, tp + dt, dt)

    # Depending on the value of sol different results will appear
    qlim = interpolate_massflow(np.arange(t0, t1 + dt, dt))[-1]
    # solve the ode in the original time range and the new time range
    t1, p1 = solve_ode_qc(f, intt, xe, pars, qlim, sol)

    return t1, p1

def plot_misfit():
    # laod pressure data
    otime, opres = load_pressure_data()

    f = ode_model
    # starting time
    t0 = 2009
    # finishing time
    t1 = 2019
    # time step
    dt = 1 / 12
    # ambient/initial pressure
    x0 = 25.16e+6
    # additional parameters
    g = 9.81
    A = 5.4484
    phi = 0.24
    pars = [7.5, 8e-6]
    tp = 2023

    g3,f12 = plt.subplots(1,1)
    # solve the ode numerically
    mtime, mpres = solve_ode(f, t0, t1, dt, x0, pars)
    f22=scipy.interpolate.interp1d(mtime, mpres)
    f12.plot(mtime, [0] * len(mtime), '--', label='0 line')
    f12.plot(otime[1:-1], opres[1:-1]/(10**6)-f22(otime[1:-1]), '.', label='misfit for guessed parameters.')

    f12.set_title('misfit for guessed parameters. Misfit = {0:.2f}'.format(np.sum(abs(opres[1:-1]/(10**6)-f22(otime[1:-1])))))
    f12.set_ylabel('Pressure misfit(MPa)')

    plt.show()
    # Curve Fitting Code Below
    constants = curve_fit(Tmodel, otime, opres, pars)
    a = constants[0][0]
    b = constants[0][1]
    pars2 = [a, b]
    ct, cp = solve_ode(f, t0, t1, dt, x0, pars2)

    g4, f13 = plt.subplots(1,1)
    f23 = scipy.interpolate.interp1d(ct,cp)

    f13.plot(ct, [0] * len(ct), '--')
    f13.plot(otime[1:-1], opres[1:-1]/(10**6)-f23(otime[1:-1]), '.', label = 'misfit for curve fitting parameters.')
    f13.set_title('misfit for curve fitting parameters.Misfit = {0:.2f} MPa'.format(np.sum(abs(opres[1:-1]/(10**6)-f23(otime[1:-1])))))
    f13.set_ylabel('Pressure misfit (MPa)')
    plt.tight_layout()
    plt.show()

    nott, notp = solve_ode(ode_modelnotgood, t0, t1, dt, x0, pars2)


    g6, f16 = plt.subplots(1,1)

    f16.plot(nott, notp, label = 'Initial model')
    f16.plot(otime, opres/(10**6), 'x', label = 'data')
    f16.set_title('Plot of Initial model and data')
    f16.set_ylabel('Pressure (MPa)')
    plt.show()

    g7, f160 = plt.subplots(1,1)

    f160.plot(mtime, mpres, label = 'Initial model')
    f160.plot(otime, opres/(10**6), 'x', label = 'data')
    f160.set_title('Plot of Final improved model and data')
    f160.set_ylabel('Pressure (MPa)')
    plt.show()

    g5, f15 = plt.subplots(1,1)
    f24 = scipy.interpolate.interp1d(nott,notp)

    f15.plot(nott, [0] * len(nott), '--')
    f15.plot(otime[1:-1], opres[1:-1]/(10**6)-f24(otime[1:-1]), '.')
    f15.set_title('misfit for initial model curve.Misfit = {0:.2f} MPa'.format(np.sum(abs(opres[1:-1]/(10**6)-f24(otime[1:-1])))))
    f15.set_ylabel('Pressure misfit (MPa)')
    plt.show()
    plt.show()

def plot_pressure_model():
    ''' Plot the Pressure LPM over top of the data.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to read and plot the experimental data, run and
        plot the Pressure LPM for hard coded parameters, and then either display the
        plot to the screen or save it to the disk.

    '''

    # laod pressure data
    otime, opres = load_pressure_data()

    f = ode_model
    # starting time
    t0 = 2009
    # finishing time
    t1 = 2019
    # time step
    dt = 1 / 12
    # ambient/initial pressure
    x0 = 25.16e+6
    # additional parameters
    g = 9.81
    A = 5.4484
    phi = 0.24
    pars = [7.5, 8e-6]
    tp = 2023

    g3,f12 = plt.subplots(1,1)
    # solve the ode numerically
    mtime, mpres = solve_ode(f, t0, t1, dt, x0, pars)

    # get cumulative gas leakage
    cumtime, cumgl = cumulator(opres, x0, 0.1e+6, otime, otime[0], otime[-1])

    # Curve Fitting Code Below
    constants = curve_fit(Tmodel, otime, opres, pars)
    a = constants[0][0]
    b = constants[0][1]
    pars2 = [a, b]
    ct, cp = solve_ode(f, t0, t1, dt, x0, pars2)

    # Gradient Descent Code Below
    # s0 = obj_dir(gaussian2D,pars)
    # gd_get_opt_parameters(pars, s0)

    # potential stakeholder solution predictions
    # capacity doubled
    tp1, pp1 = predict_t(f, t0, t1, tp, dt, x0, mpres[-1] * (10 ** 6), pars, 2)
    cumtp1, cumgl1 = cumulator(np.append(mpres, pp1) * (10 ** 6), x0, 0.1e+6, np.append(mtime, tp1), t0, tp)
    # capacity not doubled but increased
    tp2, pp2 = predict_t(f, t0, t1, tp, dt, x0, mpres[-1] * (10 ** 6), pars, 1.5)
    cumtp2, cumgl2 = cumulator(np.append(mpres, pp2) * (10 ** 6), x0, 0.1e+6, np.append(mtime, tp2), t0, tp)
    # consent not granted due to severe leakage, revert to no leakage
    # setting capacity as 0.5 as na example for now
    tp3, pp3 = predict_t(f, t0, t1, tp, dt, x0, mpres[-1] * (10 ** 6), pars, 0.5)
    cumtp3, cumgl3 = cumulator(np.append(mpres, pp3) * (10 ** 6), x0, 0.1e+6, np.append(mtime, tp3), t0, tp)
    # consent not granted, no change to capacity
    tp4, pp4 = predict_t(f, t0, t1, tp, dt, x0, mpres[-1] * (10 ** 6), pars, 1)
    cumtp4, cumgl4 = cumulator(np.append(mpres, pp4) * (10 ** 6), x0, 0.1e+6, np.append(mtime, tp4), t0, tp)

    # load mass flow rate, q and compute its sinusodial model
    qt, qv = load_massflow_data()
    qpars = [690727, 117325.27, 6.28]
    mt, mqv = solve_sinusodial(qt, *qpars)
    # perform curve fit on q
    qconsts = curve_fit(Tmodel_sinusodial, qt, qv, qpars)
    a = qconsts[0][0]
    s = qconsts[0][1]
    f = qconsts[0][2]
    qpars = [a, s, f]
    ct2, cqv = solve_sinusodial(qt, *qpars)

    # Uncertainty Analysis Below
    # get_familiar_with_model()
    # ap, bp, P = grid_search()
    ap = np.array([3.75, 3.9, 4.05, 4.2, 4.35, 4.5, 4.65, 4.8, 4.95,
                   5.1, 5.25, 5.4, 5.55, 5.7, 5.85, 6., 6.15, 6.3,
                   6.45, 6.6, 6.75, 6.9, 7.05, 7.2, 7.35, 7.5, 7.65,
                   7.8, 7.95, 8.1, 8.25, 8.4, 8.55, 8.7, 8.85, 9.,
                   9.15, 9.3, 9.45, 9.6, 9.75, 9.9, 10.05, 10.2, 10.35,
                   10.5, 10.65, 10.8, 10.95, 11.1, 11.25])
    bp = np.array([4.000e-06, 4.160e-06, 4.320e-06, 4.480e-06, 4.640e-06, 4.800e-06,
                   4.960e-06, 5.120e-06, 5.280e-06, 5.440e-06, 5.600e-06, 5.760e-06,
                   5.920e-06, 6.080e-06, 6.240e-06, 6.400e-06, 6.560e-06, 6.720e-06,
                   6.880e-06, 7.040e-06, 7.200e-06, 7.360e-06, 7.520e-06, 7.680e-06,
                   7.840e-06, 8.000e-06, 8.160e-06, 8.320e-06, 8.480e-06, 8.640e-06,
                   8.800e-06, 8.960e-06, 9.120e-06, 9.280e-06, 9.440e-06, 9.600e-06,
                   9.760e-06, 9.920e-06, 1.008e-05, 1.024e-05, 1.040e-05, 1.056e-05,
                   1.072e-05, 1.088e-05, 1.104e-05, 1.120e-05, 1.136e-05, 1.152e-05,
                   1.168e-05, 1.184e-05, 1.200e-05])
    P = np.genfromtxt('Pmatrix.csv', delimiter=',', skip_header=0)
    # samples = construct_samples(ap, bp, P, 100)

    # plot only mass flow rate
    f, ax = plt.subplots(1, 1)
    ax.plot(qt, qv, '-k', label='data')
    ax.plot(mt, mqv, '-r', label='best guess')
    ax.plot(ct2, cqv, '-b', label='curve fit')
    ax.set_title('Graphs of data, best guess and curve fit')
    ax.set_ylabel('Pressure(Pa)')
    # ax.plot(tp4, p1q, '-r',label='predicted normal q')
    plt.legend()
    plt.show()

    # plot all data no models or anything
    f0, ax0 = plt.subplots(1, 1)
    ax0.plot(otime, opres / (10 ** 6), '-k', label="observation")
    ax0.plot(cumtime, cumgl, '-b', label='Cumulative Gas Leakage')
    ax0.set_title("Pressure Changes Over Time in the Ahuroa Gas Storage Reservoir")
    ax0.set_ylabel("Pressure [MPa]")
    ax0.set_xlabel("Time [year]")
    plt.legend()
    plt.show()

    # plot model guess/fits
    f1, ax1 = plt.subplots(1, 1)
    ax1.plot(otime, opres / (10 ** 6), '-k', label="observation")
    ax1.plot(mtime, mpres, '-b', label="model guess")
    ax1.plot(ct, cp, '-r', label="best model fit")
    ax1.plot(otime, [25.8] * len(otime), '--r', label='Pressure Line')
    ax1.set_title("Pressure Changes Over Time in the Ahuroa Gas Storage Reservoir")
    ax1.set_ylabel("Pressure [MPa]")
    ax1.set_xlabel("Time [year]")
    plt.legend()
    plt.show()

    # plot the pressure predictions
    f2, ax2 = plt.subplots(1, 1)
    ax2.plot(otime, opres / (10 ** 6), 'ok', label="observation")
    ax2.plot(mtime, mpres, '-b', label="best model fit")
    ax2.plot(tp1, pp1, '-r', label='Double capacity')
    ax2.plot(tp2, pp2, '-g', label='1.5x capacity')
    ax2.plot(tp3, pp3, '-c', label='Capacity decreased')
    ax2.plot(tp4, pp4, '-y', label='No change')
    ax2.plot(otime, [25.8] * len(otime), '--r', label='Pressure Line')
    ax2.plot(tp1, [25.8] * len(tp1), '--r')
    ax2.set_xlim(2018, 2023)
    ax2.set_title("Predicted Pressure Changes in the Ahuroa Gas Storage Reservoir")
    ax2.set_ylabel("Pressure [MPa]")
    ax2.set_xlabel("Time [year]")
    plt.legend()
    plt.show()
    # code below plots the cumulative gas leakage predictions, there are more than one way to plot these
    # figure 3 has 4 plots in one column
    # f3, ax3 = plt.subplots(4, 1, figsize = (10,10), sharex=True, sharey=True)
    # # plot all cumulative gas leakage
    # carray = [cumgl1,cumgl2,cumgl3,cumgl4]
    # col = ["-r",'-g','-c','-y']
    # labels = ['Double capacity','1.5x capacity','Capacity decreased','No change']
    # for i, ax in enumerate(ax3.flat):
    #     ax.plot(np.append(mtime,tp1[1:]), carray[i], col[i], label=labels[i])
    #     # also plot the cumulative gas leakage till 2019
    #     #ax.plot(cumtime, cumgl, '-k', label='Cumulative Gas Leakage')
    #     ax.set_title(f"Cumulative Gas Leakage Solution {i+1}")
    # # set labels
    # plt.setp(ax3[-1], xlabel='Time [year]')
    # plt.setp(ax3[0:4], ylabel='Gas Leakage [MPa]')
    # plt.legend()

    # # figure 4 has cumulative sum from the start (2009) to the end (2023) in one plot
    # f4, ax4 = plt.subplots(1,1)
    # tpall = [tp1,tp2,tp3,tp4]
    # #carray = [cumgl1, cumgl2, cumgl3, cumgl4]
    # col = ["-r",'-g','-c','-y']
    # labels = ['Double capacity','1.5x capacity','Capacity decreased','No change']
    # for i in range(len(carray)):
    #     a = np.array([])
    #     a = np.append(cumgl,carray[i])
    #     ax4.plot(np.append(cumtime,tpall[i]), [sum(a[len(a)-len(carray[i])-1:j+1]) if j >= len(a)-len(carray[i]) else a[j]
    #                         for j in range(len(a))], col[i], label=labels[i])
    # ax4.set_title("Cumulative Gas Leakage")
    # plt.setp(ax4, xlabel='Time [year]')
    # plt.setp(ax4, ylabel='Gas Leakage [MPa]')
    # plt.legend()

    # figure 5 has just cumulative sum for the prediction time range
    f5, ax5 = plt.subplots(1, 1)
    # tpall = [tp1, tp2, tp3, tp4]
    carray = [cumgl1, cumgl2, cumgl3, cumgl4]
    col = ["-r", '-g', '-c', '-y']
    labels = ['Double capacity', '1.5x capacity', 'Capacity decreased', 'No change']
    for i in range(len(carray)):
        ax5.plot(np.append(mtime, tp1[1:]), carray[i], col[i], label=labels[i])
    ax5.set_title("Cumulative Gas Leakage")
    plt.setp(ax5, xlabel='Time [year]')
    plt.setp(ax5, ylabel='Gas Leakage [MPa]')
    ax5.set_xlim(2018, 2023)
    # ax5.set_ylim(6,9)
    plt.legend()
    plt.show()

    # uncertainty analysis figures
    # f6, ax6 = plt.subplots(1,1)
    # ax6.plot(otime, opres/(10**6), '-k', label='data')
    # ax6.plot(mtime, mpres, '-b', label='model guess')
    # for a,b in samples:
    # untime, unpres = solve_ode(ode_model, t0, t1, dt, x0, [a,b])
    # ax6.plot(untime, unpres, '-r')
    # plt.legend()
    # show the plot
    plt.show()

plot_misfit()
plot_pressure_model()
