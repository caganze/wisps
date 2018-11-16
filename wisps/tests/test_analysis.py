import wispshapes 
import pandas as pd
import sys
sys.path.append('./')
import wisps

file_path='./data/demo_spectrum.pkl'

def test_spectrum_object():
	sp=pd.read_pickle(file_path)
	assert(sp.cdf_snr >0.0)
