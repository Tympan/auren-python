from hashlib import md5
import numpy as np

import owai

hash = md5


class IDMixin:
    @property
    def id(self):
        return self.__class__.__name__ + "-" + hash(self.model_dump_json().encode()).hexdigest()


class GetUnitsMixin:
    def get_unit(self, attr, units_attr="units"):
        return getattr(self, attr) * owai.units(getattr(self, units_attr))


def todB(val, ref=20e-6):
    return 20 * np.log10(np.abs(val) / ref)
