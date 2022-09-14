# Purpose of this source code is to plot a model against the experimental data for our pressure ODE

import numpy as np
from matplotlib import pyplot as plt
import csv
import scipy.interpolate
from scipy.optimize import curve_fit
from Data_Vis_gradient_descent import *


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
    if (x - x0) > 0.1e+6:
        return (a*q)-(b*((x-x0)**2))
    else:
        return (a*q)

def solve_sinusodial(t,a,s,f):
    '''computes the sin value at a particular time
    '''
    # f = 6.28
    return t, (a*np.sin(f*t)+s)

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
    qt = solve_sinusodial(intt, 634813.6, 216332.6, 6.28)[1]*cap
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
        intx[i + 1] = intx[i] + (dt * (k1 + k2)/2)

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
        k2 = f(intt[i + 1], xe, qt[i], *pars, x0)
        # finally, compute improved euler value
        intx[i + 1] = intx[i] + ((dt / 2) * (k1 + k2))

    return intt, intx/(10**6)


def solve_ode2(f, t, x0, pars):
    ''' Solve an ODE numerically.

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
    intt = t
    intx = np.zeros(len(intt))
    intx[0] = x0
    dt = intt[1]-intt[0]
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

    return intt, intx/(10**6)


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

    return pt, pre*(10**6)


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


def Tmodel(t, pars):
    x0 = 25.16e+6
    pt, pr = solve_ode2(ode_model, t, x0, pars)
    return pr


def find_analytical_solution(t0,t1,dt,C,a,b,x0):
    """
    Computes the function value of the Lumped Parameter ODE
    doesn't work as of yet.
    """
    pass


def gd_get_opt_parameters(theta0, s0):
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
    dt = 0.1
    # ambient/initial pressure
    x0 = 25.16e+6
    # additional parameters
    g = 9.81
    A = 5.4484
    phi = 0.24
    pars = [7.5, 8e-6]

    # solve the ode numerically
    mtime, mpres = solve_ode(f, t0, t1, dt, x0, pars)

    # Curve Fitting Code Below
    constants = curve_fit(Tmodel, otime, opres, pars)
    a = constants[0][0]
    b = constants[0][1]
    print(a)
    print(b)
    pars2 = [a, b]
    ct, cp = solve_ode(f, t0, t1, dt, x0, pars2)

    # Gradient Descent Code Below
    #s0 = obj_dir(gaussian2D,pars)
    #gd_get_opt_parameters(pars, s0)

    # Stakeholder Prediction
    tp, pp = predict_t(f, t0, t1, 2023, dt, x0, mpres[-1]*10**6, pars,2)

    # plot all data/model
    f, ax1 = plt.subplots(1,1)
    ax1.plot(otime, opres/(10**6), '-k', label="observation")
    #ax1.plot(mtime, mpres, '-r', label="model guess")
    #ax1.plot(ct, cp, '-b', label="best model fit")
    ax1.plot(tp, pp, '-b', label="model guess/predict")
    ax1.set_ylabel("Pressure [MPa]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_pressure_model()