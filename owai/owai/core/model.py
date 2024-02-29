# -*- coding: utf-8 -*-

import sympy as sym
import numpy as np

import matplotlib.pyplot as plt
import ipywidgets
from owai.core.units import units


class Model:
    """
    Generic Class for 1D plane wave models of tubes
    """
    rho = 1.225 * units.kg / units.m ** 3  # Density of fluid (air by default)
    sound_speed = 343 * units.m / units.s # Speed of sound in fluid (Air at 20°C by default)
    absorption_loss = 0 # Absorption loss -- imaginary value of K

    def u(self, f, x):
        """Return value of velocity(u) at given frequencies and x positions in the tube

        Parameters
        ----------
        f : np.ndarray * pint.Hz (or other unit)
            Frequency
        x : np.ndarray * pint.mm (or other unit)
            Location in tube to evaluate velocity

        Return
        -------
        np.ndarray(dtype=complex)
            The complex velocity in frequency space evaluated at the input frequency and positions
        """
        pass

    def p(self, f, x):
        """Return value of pressure (p) at given frequencies and x positions in the tube

        Parameters
        ----------
        f : np.ndarray * pint.Hz (or other unit)
            Frequency
        x : np.ndarray * pint.mm (or other unit)
            Location in tube to evaluate velocity

        Return
        -------
        np.ndarray(dtype=complex)
            The complex pressure in frequency space evaluated at the input frequency and positions
        """
        pass

    def z(self, f, x):
        """Return value of impedance (z=p/u) at given frequencies and x positions in the tube

        Parameters
        ----------
        f : np.ndarray * pint.Hz (or other unit)
            Frequency
        x : np.ndarray * pint.mm (or other unit)
            Location in tube to evaluate velocity

        Return
        -------
        np.ndarray(dtype=complex)
            The complex impedance in frequency space evaluated at the input frequency and positions
        """
        return self.p(f, x) / self.u(f, x)

    @property
    def z0(self):
        """The characteristic impedance = density / sound_speed
        """
        return self.rho * self.sound_speed

    def k(self, f):
        """The wave number

        Parameters
        ----------
        f : np.ndarray * pint.Hz
            The input frequency

        Returns
        -------
        np.ndarray
            The wave number k = 2πf / c (1 - iα)
        """
        return np.pi * 2 * f / self.sound_speed * (1 - 1j * self.absorption_loss)

def straight_tube_derivation():
    """Derives the equations for the coefficients of the forwards (+x) (A) and reverse (-x) B waves given
    the tube geometry, reflection coefficients, and a source at either end of the tube.

    The solution for the pressure is: P = Ae^{ikx} + Be^{-ikx}
    """
    A, B, P0, PL, R0, RL, L, k, x = sym.symbols("A, B, P0, PL, R0, RL, L, k, x")
    eqn0 = A * sym.exp(-1j*k*0) + B * sym.exp(1j*k*0) - ((1 + R0) * B * sym.exp(1j*k*0) + P0)
    eqn1 = A * sym.exp(-1j*k*L) + B * sym.exp(1j*k*L) - ((1 + RL) * A * sym.exp(-1j*k*L) + PL)
    a = sym.solve(eqn0, A)[0]
    eqn2 = eqn1.subs(A, a)
    b = sym.solve(eqn2, B)[0]
    print ("A =", a)
    print ("B =", b)


class StraightTube(Model):
    """ Generates solutions for tube of constant cross section (straight) """
    def __init__(self, L, P0, PL, R0, RL, rho=None, sound_speed=None):
        self.L = L
        self.R0 = R0
        self.RL = RL
        self.P0 = P0
        self.PL = PL
        if rho is not None:
            self.rho = rho
        if sound_speed is not None:
            self.sound_speed = sound_speed

    def u(self, f, x):
        """Return value of velocity(u) at given frequencies and x positions in the straight tube

        Parameters
        ----------
        f : np.ndarray * pint.Hz (or other unit)
            Frequency
        x : np.ndarray * pint.mm (or other unit)
            Location in tube to evaluate velocity

        Return
        -------
        np.ndarray(dtype=complex)
            The complex velocity in frequency space evaluated at the input frequency and positions
        """
        k = self.k(f)
        B = self.B(k)
        U = (self.A(k, B=B) * np.exp(-1j * k * x) - B * np.exp(1j * k * x)) / self.z0
        return U

    def p(self, f, x):
        """Return value of pressure (p) at given frequencies and x positions in the straight tube

        Parameters
        ----------
        f : np.ndarray * pint.Hz (or other unit)
            Frequency
        x : np.ndarray * pint.mm (or other unit)
            Location in tube to evaluate velocity

        Return
        -------
        np.ndarray(dtype=complex)
            The complex pressure in frequency space evaluated at the input frequency and positions
        """
        k = self.k(f)
        B = self.B(k)
        P = self.A(k, B=B) * np.exp(-1j * k * x) + B * np.exp(1j * k * x)
        return P

    def A(self, k, x=None, B=None):
        """Computes the complex forward wave amplitude coefficient

        Parameters
        ----------
        k : np.ndarray
            The wave number (see self.k)
        x : None, optional
            Dummy input to allow plotting using the same (f, x) interface (though x is ignored)
        B : np.ndarray, optional
            Pre-computed B (for optimization sake, used internally), by default None

        Returns
        -------
        np.ndarray
            Complex forward wave amplitude Ae^ikx
        """
        if B is None:
            B = self.B(k)
        return self.R0 * B + self.P0

    def B(self, k, x=None):
        """Computes the complex reverse wave amplitude coefficient

        Parameters
        ----------
        k : np.ndarray
            The wave number (see self.k)
        x : None, optional
            Dummy input to allow plotting using the same (f, x) interface (though x is ignored)

        Returns
        -------
        np.ndarray
            Complex reverse wave amplitude Ae^ikx
        """
        L, R0, RL, P0, PL = self.L, self.R0, self.RL, self.P0, self.PL
        B = (np.exp(1j * k * L) * PL + RL * P0) / (np.exp(1j * k * 2 * L) - R0 * RL )
        return B

    def widget_frequency(self, f, xs, val="z", fig_num=1, figkwargs={}, params={}):
        """Generated an ipywidgets tool to visualize the quantity in frequency space

        Parameters
        ----------
        f : np.ndarray
            Frequencies at which value will be visualized, should have units of Hz
        xs : iteratble
            Locations where function will be evaluated along the tube -- should have a unit
        val : str, optional
            Value to display, one of {"p", "u", "z", "A", "B"}, by default "z"
        fig_num : int, optional
            Figure number to use, by default 1
        figkwargs : dict, optional
            Additional arguments passed to the matplotlib figure creation function, by default {}
        params : dict, optional
            Optional parameters passed to the widgets created by this function, by default {}. These include:
                * "L": Initial value for L, length of the tube
                * "LMin": Minimum value for LMin
                * "LMax": Maximum value for LMax
                * "R0": Initial value for R0, reflection coefficient at x=0
                * "RL": Initial value for RL, reflection coefficient at x=L
                * "alpha": Initial value for alpha, the absorption coefficient (see self.k)
                * "P0": Initial value for P0, whether there is a source at x=0
                * "PL": Initial value for PL, whether there is a source at x=0

        Returns
        -------
        ipywidgets.VBox
            An ipywidgets container with the widget.
        """
        func = getattr(self, val)

        ax0 = None
        ax1 = None

        plot_out = ipywidgets.Output()
        handles = []
        # Create the initial figure
        with plot_out:
            plt.close(1)
            fig, ax = plt.subplots(2, 1, sharex=True, num=fig_num, **figkwargs)

            for x in xs:
                v = func(f, x).magnitude
                h0 = ax[0].loglog(f.magnitude, np.abs(v))[0]
                h1 = ax[1].semilogx(f.magnitude, np.rad2deg(np.angle(v)))[0]
                handles.append((h0, h1))
            ax[1].set_xlabel("Frequency (Hz)")
            ax[0].set_ylabel("%s Amplitude (uncal)" % (val.upper()))
            ax[1].set_ylabel("Phase (uncal)")
            ax[1].set_ylim([-180, 180])
            ax[1].set_xlim([f.magnitude.min(), f.magnitude.max()])
            ax[1].legend(xs)

        def update_plot(*args, **kwargs):
            self.L = L.value * Lunits
            self.R0 = R0.value
            self.RL = RL.value
            self.P0 = P0.value * 1.0
            self.PL = PL.value * 1.0
            self.absorption_loss = alpha.value
            my_max = -np.inf
            my_min = np.inf
            with plot_out:
                for h, x in zip(handles, xs):
                    v = func(f, x).magnitude
                    my_max = max(my_max, np.abs(v).max())
                    my_min = min(my_min, np.abs(v).min())
                    h[0].set_ydata(np.abs(v))
                    h[1].set_ydata(np.rad2deg(np.angle(v)))
                ax[0].set_ylim([my_min, my_max])
        Lunits = self.L.units
        L = ipywidgets.FloatSlider(
            value=params.get("L", self.L.magnitude),
            min=params.get("LMin", self.L / 10).magnitude,
            max=params.get("LMax", self.L * 2).magnitude,
            description="Length",
        )
        L.step = (L.max - L.min) / 256
        R0 = ipywidgets.FloatSlider(value=params.get("R0", self.R0), min=0, max=1, step=0.01, description=r"$R_{0}$")
        RL = ipywidgets.FloatSlider(value=params.get("RL", self.RL), min=0, max=1, step=0.01, description=r"$R_{L}$")
        alpha = ipywidgets.FloatSlider(value=params.get("alpha", self.absorption_loss), min=0, max=10, step=.025, description=r"$\alpha_{loss}$")
        P0 = ipywidgets.Checkbox(value=params.get("P0", self.P0), description=r"$P_0$")
        PL = ipywidgets.Checkbox(value=params.get("PL", self.PL), description=r"$P_L$")

        L.observe(update_plot)
        R0.observe(update_plot)
        RL.observe(update_plot)
        alpha.observe(update_plot)
        P0.observe(update_plot)
        PL.observe(update_plot)

        container = ipywidgets.VBox([ipywidgets.HBox([R0, RL, alpha, P0, PL]), L, plot_out])

        # For debugging
        self.handles = handles
        self.ax = ax
        return container
