import pytest
from auren.auren import function, Class


class TestOWAIFunction(object):
    def test_function(self):
        assert function("test") == "TEST"

    def test_function_exception(self):
        with pytest.raises(TypeError, match="should be of type"):
            function(43)


class TestOWAIClass(object):
    @pytest.mark.integration
    def test_class_integration(self):
        c = Class(3.0)
        with pytest.raises(TypeError, match="should be a float"):
            c.set_x("test")

    def test_class_unit(self):
        c = Class(3)
        assert c.x_squared == 9
        c.set_x(4.0)
        assert c.x_squared == 16

        # Check to make sure Class is NOT a singleton (shared 'x' variable)
        c2 = Class(4)
        assert c2.x_squared == c.x_squared

        c2.set_x(5.0)
        assert c2.x_squared != c.x_squared
