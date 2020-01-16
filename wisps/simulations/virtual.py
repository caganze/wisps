
from .core import *
from .initialize import *

class SimulatedUCD(object):
    """
    Asimulated object with all the properties properly sampled

    properties
     :fundemental parameters: age, spectral type, mass, temperature, radius, luminosity
     :survey paramaters: mags, snr, distance, selection probability
    """
    def __init__(self, **kwargs):
        self._fundemental_params={} #fundamental parameters
        self._survey_params={}

    @property
    def fundamental(self):
        return self._fundemental_params

    @property
    def surveyparams(self):
        return self.survey_params

    @fundamental.setter
    def fundamental(self, new_funda):
        self._fundemental_params=new_funda
    