"""
DOCUMENTATION FOR THE PROJECT MODULE
"""


def function(attrs):
    """
    Documentation example for a function

    Parameters
    -----------
    attrs : str
        We use the `numpy` docstring format

    Returns
    --------
    str
        Upper case of attrs

    Raises : TypeError
        If attr is not a string
    """

    if not isinstance(attrs, str):
        raise TypeError("attrs should be of type `str` but if of type {}".format(type(attrs)))
    return attrs.upper()


class Class(object):
    """Documentation for the class

    Attributes
    -----------
    x : float
        A number
    """

    x = None

    def __init__(self, x):
        """Constructor doc string

        Parameters
        -----------
        x : float
            The instance number
        """
        self.x = x

    @property
    def x_squared(self):
        """
        A property (similar to read-only attribute), squares A number
        """
        return self.x ** 2

    def set_x(self, x):
        """Method documentation

        Parameters
        -----------
        x : float
           A number
        """
        if not isinstance(x, float):
            raise TypeError("x should be a float, not {}".format(type(x)))
        self.x = x
