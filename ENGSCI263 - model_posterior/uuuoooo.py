# ENGSCI263: Lab Exercise 2
# lab2.py

# PURPOSE:
# IMPLEMENT a lumped parameter model and CALIBRATE it to data.

# PREPARATION:
# Review the lumped parameter model notes and use provided data from the kettle experiment.

# SUBMISSION:
# - Show your calibrated LPM to the instructor (in Week 3).

# imports
import numpy as np
from matplotlib import pyplot as plt
import csv
from scipy import interpolate
from scipy.optimize import curve_fit


def ode_model(t, x, q, a, b, xo):
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
            Derivative of depende=nt variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> ode_model(0, 1, 2, 3, 4, 5)
        22

    '''
    thresh = 0.02
    if x - xo > thresh:
        return a * q - b * (x - xo) ** 2
    else:
        return a * q


def numerical(t, x, xo, q, a, b):
    if x - xo > .02:
        return (np.exp(2 * np.sqrt(a * b * q) * t) - 1) / (np.exp(2 * np.sqrt(a * b * q) * t) + 1)
    else:
        return a * q * t


def solver(t0, t1, dt, x0, xo, q, pars):
    t = np.arange(t0, t1 + dt, dt)
    x = np.zeros(len(t))

    x[0] = x0
    for i, y in enumerate(t[0:-1]):
        x[i + 1] = numerical(y, x[i], xo,q[i], *pars)

    return t, x


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


def solve_ode(f, t0, t1, dt, x0, q, pars):
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
    t = np.arange(t0, t1 + dt, dt)
    x = np.zeros(len(t))

    x[0] = x0
    steps = len(t)
    for i in range(steps - 1):
        dydt1 = ode_model(t[i], x[i], q[i], *pars)
        ypotent = x[i] + dydt1 * dt
        dydt2 = ode_model(t[i + 1], ypotent, q[i + 1], *pars)
        x[i + 1] = x[i] + dt * 0.5 * (dydt1 + dydt2)

    return t, x


def plot_benchmark():
    ''' Compare analytical and numerical solutions.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to obtain analytical and numerical solutions,
        plot these, and either display the plot to the screen or save it to the disk.
        
    '''
    t0 = 0
    t1 = 2.3
    dt = 0.01
    t = np.arange(t0, t1 + dt, dt)
    a, b, q, po = .5, 0.08, np.sin(t), 0
    tl, legit = solver(t0, t1, dt, 0, 25,q, [a, b])

    t2, predicted = solve_ode(ode_model, t0, t1, dt, 0, q, [a, b, 25])

    ft1, axe = plt.subplots(3,1)

    axe[0].plot(t2, predicted, '.', label = 'Numerical')
    axe[0].set_title('Numerical and Analytical solution')
    axe[0].set_xlabel('Time(s)')
    axe[0].set_ylabel('Pressure(Pa)')
    axe[0].plot(tl, legit, label = 'Analytical')
    axe[0].legend()

    axe[1].plot(tl, np.abs(predicted - legit), '.')
    axe[1].set_title('Relative Error')
    axe[1].set_xlabel('Time(s)')
    axe[1].set_ylabel('Pressure(Pa)')

    new, Lost = [0.01, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019], []
    for i in new:
        t = np.arange(t0, t1 + i, i)
        q = np.sin(t)
        temp, hold = solve_ode(ode_model, t0, t1, i, 0, q, [a, b, 0])
        Lost.append(hold[-1])

    convert = lambda x: 1 / x
    axe[2].plot(list(map(convert, new))[::-1], Lost)
    axe[2].set_title('Timestep convergence')
    axe[2].set_xlabel('1/deltaTime')
    axe[2].set_ylabel('Pressure(T=0.5s)')

    ft1.tight_layout()
    plt.show()
    plt.savefig('Graphs')


def load_kettle_temperatures():
    ''' Returns time and temperature measurements from kettle experiment.

        Parameters:
        -----------
        none

        Returns:
        --------
        t : array-like
            Vector of times (seconds) at which measurements were taken.
        T : array-like
            Vector of Temperature measurements during kettle experiment.

        Notes:
        ------
        It is fine to hard code the file name inside this function.

        Forgotten how to load data from a file? Review datalab under Files/cm/
        engsci233 on the ENGSCI263 Canvas page.
    '''
    mass = np.loadtxt('gs_mass.txt', dtype=float, skiprows=1, delimiter=',')
    pressure = np.loadtxt('gs_pres.txt', dtype=float, skiprows=1, delimiter=',')

    return mass, pressure


def turner(t, *pars):
    t0 = t[0]
    t1 = t[-1]
    dt = t[1] - t[0]
    arr = np.arange(t0, t1 + dt, dt)
    q = interpolate_kettle_heatsource(arr)
    t, T = load_kettle_temperatures()
    x0 = T[:, 1][0] * (10 ** 6)

    x, y = solve_ode(ode_model, t0, t1, dt, x0, q, pars)

    return y


def interpolate_kettle_heatsource(t):
    ''' Return heat source parameter q for kettle experiment.

        Parameters:
        -----------
        t : array-like
            Vector of times at which to interpolate the heat source.

        Returns:
        --------
        q : array-like
            Heat source (Watts) interpolated at t.

        Notes:
        ------
        This doesn't *have* to be coded as an interpolation problem, although it 
        will help when it comes time to do your project if you treat it that way. 

        Linear interpolation is fine for this problem, no need to go overboard with 
        splines. 
        
        Forgotten how to interpolate in Python, review sdlab under Files/cm/
        engsci233 on the ENGSCI263 Canvas page.
    '''
    # suggested approach
    # hard code vectors tv and qv which define a piecewise heat source for your kettle 
    # experiment
    # use a built in Python interpolation function

    mass, pressure = load_kettle_temperatures()

    f = interpolate.interp1d(mass[:, 0], mass[:, 1])

    return f(t)


def plot_kettle_model():
    ''' Plot the kettle LPM over top of the data.

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
        plot the kettle LPM for hard coded parameters, and then either display the 
        plot to the screen or save it to the disk.

    '''

    t, T = load_kettle_temperatures()
    t4 = np.arange(2009, 2019, 0.2)
    q = interpolate_kettle_heatsource(t4)
    a = 7.5
    b = 3.95e-6
    constants = [a, b, 25.16e+6]
    t2, T2 = solve_ode(ode_model, 2009, 2018.8, 0.2, (T[:, 1][0] * (10 ** 6)), q, constants)

    hi, Atotal = cumulator(T2, 25.16e6, 0.1e6, t2, 2011.3, 2016.3)

    plt.plot(t2, T2 / (10 ** 6))
    plt.plot(T[:, 0], T[:, 1])
    plt.plot(hi, Atotal)
    plt.show()

plot_benchmark()
