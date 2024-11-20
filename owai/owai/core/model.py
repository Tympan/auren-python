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
    sound_speed = 343 * units.m / units.s  # Speed of sound in fluid (Air at 20°C by default)
    absorption_loss = 0  # Absorption loss -- imaginary value of K re f
    absorption_loss2 = 0  # Absorption loss -- imaginary value of K re f^3

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
        """The characteristic impedance = density / sound_speed"""
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
        omega = np.pi * 2 * f / self.sound_speed
        return omega * (1 - 1j * (self.absorption_loss + self.absorption_loss2 * omega.magnitude ** 2))

    #     return np.pi * 2 * f / self.sound_speed

    # def alpha(self, f):
    #     return self.k(f) * self.absorption_loss


def straight_tube_derivation():
    """Derives the equations for the coefficients of the forwards (+x) (A) and reverse (-x) B waves given
    the tube geometry, reflection coefficients, and a source at either end of the tube.

    The solution for the pressure is: P = Ae^{ikx} + Be^{-ikx}
    """
    A, B, P0, PL, R0, RL, L, k, x = sym.symbols("A, B, P0, PL, R0, RL, L, k, x")
    eqn0 = A * sym.exp(-1j * k * 0) + B * sym.exp(1j * k * 0) - ((1 + R0) * B * sym.exp(1j * k * 0) + P0)
    eqn1 = A * sym.exp(-1j * k * L) + B * sym.exp(1j * k * L) - ((1 + RL) * A * sym.exp(-1j * k * L) + PL)
    a = sym.solve(eqn0, A)[0]
    eqn2 = eqn1.subs(A, a)
    b = sym.solve(eqn2, B)[0]
    print("A =", a)
    print("B =", b)


def straight_tube_derivation_from_measurements():
    """Derives the equations for the coefficients of the forwards (+x) (A) and reverse (-x) B waves given
    the tube geometry, and pressure measured at two points.

    The solution for the pressure is: P = Ae^{ikx} + Be^{-ikx}
    """
    A, B, p0, p1, k, x0, x1 = sym.symbols("A, B, p0, p1, k, x0, x1")
    eqn0 = A * sym.exp(-1j * k * x0) + B * sym.exp(1j * k * x0) - p0
    eqn1 = A * sym.exp(-1j * k * x1) + B * sym.exp(1j * k * x1) - p1
    s = sym.solve([eqn0, eqn1], [A, B])
    a = sym.solve(eqn0, A)[0]
    eqn2 = eqn1.subs(A, a)
    b = sym.solve(eqn2, B)[0]
    print("A =", a)
    print("B =", b)


def straight_tube_calibration_from_measurements_direct():
    """Derives the equations for the coefficients of the forwards (+x) (A) and reverse (-x) B waves given
    the tube geometry, and pressure measured at two points.

    The solution for the pressure is: P = Ae^{ikx} + Be^{-ikx}
    """
    Aa, Ab, Ba, Bb, c0, c1, p0a, p0b, p1a, p1b, p2a, p2b, k, x0, x1, x2 = sym.symbols(
        "Aa, Ab, Ba, Bb, c0, c1, p0a, p0b p1a, p1b, p2a, p2b, k, x0, x1, x2"
    )
    eqn0a = Aa * sym.exp(-1j * k * x0) + Ba * sym.exp(1j * k * x0) - p0a * c0
    eqn1a = Aa * sym.exp(-1j * k * x1) + Ba * sym.exp(1j * k * x1) - p1a * c1
    eqn2a = Aa * sym.exp(-1j * k * x2) + Ba * sym.exp(1j * k * x2) - p2a
    eqn0b = Ab * sym.exp(-1j * k * x0) + Bb * sym.exp(1j * k * x0) - p0b * c0
    eqn1b = Ab * sym.exp(-1j * k * x1) + Bb * sym.exp(1j * k * x1) - p1b * c1
    eqn2b = Ab * sym.exp(-1j * k * x2) + Bb * sym.exp(1j * k * x2) - p2b
    s = sym.solve([eqn0a, eqn1a, eqn0b, eqn1b, eqn2a, eqn2b], [Aa, Ab, Ba, Bb, c0, c1])
    a = sym.solve(eqn0, A)[0]
    eqn2 = eqn1.subs(A, a)
    b = sym.solve(eqn2, B)[0]
    print("A =", a)
    print("B =", b)


def two_diameter_tube_from_measurements_deriviation():
    A1, A2, B1, B2, x0, x1, xp, z1, z2, k, x, p0, p1 = sym.symbols("A1, A2, B1, B2, x0, x1, xp, z1, z2, k, x, p0, p1")
    e = sym.exp
    eqn0 = A1 * e(-1j * k * x0) + B1 * e(1j * k * x0) - p0
    eqn1 = A1 * e(-1j * k * x1) + B1 * e(1j * k * x1) - p1
    # Continuity of pressure across interface
    eqn2 = A1 * e(-1j * k * xp) + B1 * e(1j * k * xp) - (A2 * e(-1j * k * xp) + B2 * e(1j * k * xp))
    # Continuity of volume flow rate across interface
    eqn3 = A1 / z1 * e(-1j * k * xp) - B1 / z1 * e(1j * k * xp) - (A2 / z2 * e(-1j * k * xp) - B2 / z2 * e(1j * k * xp))
    a1 = sym.solve(eqn0, A1)[0]
    eqn1a = eqn1.subs(A1, a1)
    b1 = sym.solve(eqn1a, B1)[0]
    a2 = sym.solve(eqn2, A2)[0]
    eqn3a = eqn3.subs(A2, a2)
    b2 = sym.solve(eqn3a, B2)[0]

    sol = sym.solve([eqn0, eqn1, eqn2, eqn3], (A1, B1, A2, B2))

    print("A1 = ", a1)
    print("B1 =", b1)
    print("A2 =", a2)
    print("B2 =", b2)


class StraightTube(Model):
    """ Generates solutions for tube of constant cross section (straight) """

    def __init__(self, L, P0, PL, R0, RL, rho=None, sound_speed=None, absorption_loss=0):
        self.L = L
        self.R0 = R0
        self.RL = RL
        self.P0 = P0
        self.PL = PL
        self.absorption_loss = absorption_loss
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
        # alpha = self.alpha(f)
        B = self.B(k)
        U = (self.A(k, B=B) * np.exp(-1j * k * x) - B * np.exp(1j * k * x)) / self.z0
        # U = (self.A(k, B=B) * np.exp(-1j * k * x) * np.exp(-alpha * x) \
        #      - B * np.exp(1j * k * x) * np.exp(-alpha * (self.L - x))) / self.z0
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
        # alpha = self.alpha(f)
        B = self.B(k)
        P = self.A(k, B=B) * np.exp(-1j * k * x) + B * np.exp(1j * k * x)
        # P = self.A(k, B=B) * np.exp(-1j * k * x) * np.exp(-alpha * x) \
        # + B * np.exp(1j * k * x) * np.exp(-alpha * (self.L - x))
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
        B = (np.exp(1j * k * L) * PL + RL * P0) / (np.exp(1j * k * 2 * L) - R0 * RL)
        return B

    def u_measured(self, f, x, x0, x1, p0, p1, A=None, B=None):
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
        if B is None:
            B = self.B_measured(k, x, x0, x1, p0, p1)
        if A is None:
            A = self.A_measured(k, x, x0, x1, p0, p1, B)
        U = (A * np.exp(-1j * k * x) - B * np.exp(1j * k * x)) / self.z0
        return U

    def p_measured(self, f, x, x0, x1, p0, p1, A=None, B=None):
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
        if B is None:
            B = self.B_measured(k, x, x0, x1, p0, p1)
        if A is None:
            A = self.A_measured(k, x, x0, x1, p0, p1, B)
        P = A * np.exp(-1j * k * x) + B * np.exp(1j * k * x)
        return P

    def A_measured(self, k, x, x0, x1, p0, p1, B=None):
        if B is None:
            B = self.B_measured(k, x, x0, x1, p0, p1)

        A = (-B * np.exp(1j * k * x0) + p0) * np.exp(1j * k * x0)
        return A

    def B_measured(self, k, x, x0, x1, p0, p1):
        B = (p0 * np.exp(1j * k * x0) - p1 * np.exp(1j * k * x1)) / (
            np.exp(2.0 * 1j * k * x0) - np.exp(2.0 * 1j * k * x1)
        )
        return B

    def widget_frequency(self, f, xs, other_plot=None, val="z", fig_num=1, figkwargs={}, params={}):
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
                * "alpha2": Initial value for alpha, the absorption coefficient (see self.k)
                * "P0": Initial value for P0, whether there is a source at x=0
                * "PL": Initial value for PL, whether there is a source at x=0
                * "xoff": Initial value for x position offset

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
                if other_plot is not None:
                    ax[0].loglog(f.magnitude, np.abs(other_plot), "k", alpha=0.5)
                    ax[1].semilogx(f.magnitude, np.rad2deg(np.angle(other_plot)), "k", alpha=0.5)
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
            self.sound_speed = (T.value * 0.606 + 331.3) * units.m / units.s
            self.R0 = R0.value
            self.RL = RL.value
            self.P0 = P0.value * 1.0
            self.PL = PL.value * 1.0
            self.absorption_loss = alpha.value
            self.absorption_loss2 = alpha2.value
            my_max = -np.inf
            my_min = np.inf
            with plot_out:
                for h, x in zip(handles, xs):
                    v = func(f, x + xoff.value * x.units).magnitude
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
        # see https://en.wikipedia.org/wiki/Speed_of_sound#Details  -- 0.606 (m/s) / °C
        T = ipywidgets.FloatSlider(value=params.get("T", 21.5), min=0, max=40, step=0.1, description=r"$Temperature$")
        R0 = ipywidgets.FloatSlider(value=params.get("R0", self.R0), min=0, max=1, step=0.01, description=r"$R_{0}$")
        RL = ipywidgets.FloatSlider(value=params.get("RL", self.RL), min=0, max=1, step=0.01, description=r"$R_{L}$")
        alpha = ipywidgets.FloatSlider(
            value=params.get("alpha", self.absorption_loss),
            min=0,
            max=0.1,
            step=0.1 / 128,
            description=r"$\alpha_{loss}$",
        )
        alpha2 = ipywidgets.FloatSlider(
            value=params.get("alpha2", self.absorption_loss2),
            min=0,
            max=0.000001 / 3,
            step=0.000001 / 3 / 128,
            description=r"$\alpha_{loss,2}$",
        )
        P0 = ipywidgets.Checkbox(value=params.get("P0", self.P0), description=r"$P_0$")
        PL = ipywidgets.Checkbox(value=params.get("PL", self.PL), description=r"$P_L$")
        xoff = ipywidgets.FloatSlider(
            value=params.get("x_off", 0), min=-2, max=2, step=0.1, description=r"x-position offset"
        )

        L.observe(update_plot)
        T.observe(update_plot)
        R0.observe(update_plot)
        RL.observe(update_plot)
        alpha.observe(update_plot)
        alpha2.observe(update_plot)
        P0.observe(update_plot)
        PL.observe(update_plot)
        xoff.observe(update_plot)

        container = ipywidgets.VBox(
            [
                ipywidgets.VBox([ipywidgets.HBox([R0, RL, alpha]), ipywidgets.HBox([P0, PL, alpha2])]),
                ipywidgets.HBox([T, L, xoff]),
                plot_out,
            ]
        )

        # For debugging
        self.handles = handles
        self.ax = ax
        return container


class TwoDiameterTube(StraightTube):
    D0 = None
    D1 = None
    xp = None
    x0 = None
    x1 = None

    @property
    def za0(self):
        area0 = 0.25 * np.pi * self.D0 ** 2
        return self.z0 / area0

    @property
    def za1(self):
        area1 = 0.25 * np.pi * self.D1 ** 2
        return self.z0 / area1

    def __init__(self, x0, x1, xp, D0, D1, rho=None, sound_speed=None, absorption_loss=0):
        """Constructor for Two-Diameter Tube model

        Parameters
        ----------
        x0 : float
            Distance from the reference x=0 location, to where the p0 pressure is measured, uses units
        x1 : float
            Distance from the reference x=0 location, to where the p1 pressure is measured, uses units
        xp : float
            Distance from the reference x=0 location, to where the diameter changes from
            D0 to D1, uses units
        D0 : float
            Diameter of the first tube, uses units
        D1 : float
            Diameter of the second tube, uses units
        rho : float, optional
            Density of medium, by default uses density of air in SI units
        sound_speed : float, optional
            Sound speed in medium, by default uses air in SI units
        absorption_loss : float, optional
            Absorption loss coefficient, by default 0
        """
        self.xp = xp
        self.x0 = x0
        self.x1 = x1
        self.D0 = D0
        self.D1 = D1
        self.absorption_loss = absorption_loss
        if rho is not None:
            self.rho = rho
        if sound_speed is not None:
            self.sound_speed = sound_speed

    def A0_measured(self, k, p0, p1, B0=None):
        return self.A_measured(k, None, self.x0, self.x1, p0, p1, B0)

    def B0_measured(self, k, p0, p1):
        return self.B_measured(k, None, self.x0, self.x1, p0, p1)

    def A1_measured(self, k, A0, B0, B1):
        A2 = A0 + B0 * np.exp(2j * k * self.xp) - B1 * np.exp(2j * k * self.xp)
        return A2

    def B1_measured(self, k, A0, B0):
        B1 = (
            0.5 * A0 * np.exp(-2j * k * self.xp)
            - 0.5 * A0 * self.za1 * np.exp(-2j * k * self.xp) / self.za0
            + 0.5 * B0
            + 0.5 * B0 * self.za1 / self.za0
        )
        return B1

    def u_measured(self, f, x, p0, p1, A0=None, B0=None, A1=None, B1=None):
        """Return value of pressure (p) at given frequencies and x positions in the straight tube

        Parameters
        ----------
        f : np.ndarray * pint.Hz (or other unit)
            Frequency
        x : np.ndarray * pint.mm (or other unit)
            Location in tube to evaluate velocity
        p0 : np.ndarray * pint.Pa (or other unit)
            Pressure measured at self.x0
        p1 : np.ndarray * pint.Pa (or other unit)
            Pressure measured at self.x1

        Return
        -------
        np.ndarray(dtype=complex)
            The complex pressure in frequency space evaluated at the input frequency and positions
        """
        k = self.k(f)
        if B0 is None:
            B0 = self.B0_measured(k, p0, p1)
        if A0 is None:
            A0 = self.A0_measured(k, p0, p1, B0)
        if x <= self.xp:
            # still in the D0 tube, proceed as usual
            return (A0 * np.exp(-1j * k * x) - B0 * np.exp(1j * k * x)) / self.z0

        # Beyond the gap, now we need A1/B1
        if B1 is None:
            B1 = self.B1_measured(k, A0, B0)
        if A1 is None:
            A1 = self.A1_measured(k, A0, B0, B1)
        return (A1 * np.exp(-1j * k * x) - B1 * np.exp(1j * k * x)) / self.z0

    def p_measured(self, f, x, p0, p1, A0=None, B0=None, A1=None, B1=None):
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
        if B0 is None:
            B0 = self.B0_measured(k, p0, p1)
        if A0 is None:
            A0 = self.A0_measured(k, p0, p1, B0)
        if x <= self.xp:
            # still in the D0 tube, proceed as usual
            return A0 * np.exp(-1j * k * x) + B0 * np.exp(1j * k * x)

        # Beyond the gap, now we need A1/B1
        if B1 is None:
            B1 = self.B1_measured(k, A0, B0)
        if A1 is None:
            A1 = self.A1_measured(k, A0, B0, B1)
        return A1 * np.exp(-1j * k * x) + B1 * np.exp(1j * k * x)
