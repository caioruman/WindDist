import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np


def interpPressure(pressure, pressure_levels, data, interp='linear'):
  """
  Interpolate data to custom pressure levels

  pressure: Original pressure level

  pressure_levels: Custom pressure level

  data: Original variable to be interpolated to custom pressure level

  returns: new_val, the original variable interpolated.
  """
  from scipy import interpolate

  f = interpolate.interp1d(pressure, data, kind=interp)
  
  new_val = f(pressure_levels)

  return new_val

def calc_height(months, datai, dataf, height=[700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]):
  # Interpolate to my levels [700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  m_height = []
  for mm in months:
    for year in range(datai, dataf+1):
      df = pd.read_csv('model_height/results_model_lvl_{0}.txt'.format(year, skipinitialspace=True))
        
      m_height_m = np.squeeze(df.loc[df['Month'] == mm].values)[:]

      m_height_hpa = np.append(np.squeeze(df.loc[df['Month'] == float('{0}.{0}'.format(mm))].values)[1:], 1000)      

      m_height.append(interpPressure(m_height_hpa, height, m_height_m))
  
  return np.mean(m_height, axis=0)  

def calc_kerneldensity(df, bins1=0, bins2=40, space=80):
  hist_aux = []
  for i in range(0,8):
      kde_skl = KernelDensity(bandwidth=0.4)
      #aux = np.array(df_n['1000.0'])
      aux = np.copy(df[:,i])
      aux_grid2 = np.linspace(bins1,bins2,space)
      kde_skl.fit(aux[:, np.newaxis])
      log_pdf = kde_skl.score_samples(aux_grid2[:, np.newaxis])
      hist_aux.append(np.exp(log_pdf)*100)

  return np.array(hist_aux)

def calc_histogram(df, bins1=0, bins2=40.25, inc=1):

  hist_l = []
  bins = np.arange(bins1,bins2,inc)
  for i in range(0, df.shape[1]):    
    hist, bins = np.histogram(df[:,i], bins=bins)
    hist_l.append(hist*100/sum(hist))

  return np.array(hist_l)

def calc_wind_density(wind, pho):
  '''
    Calculating for 
  '''
  #pho = 1.225 - (1.194*10**(-4))*m_height
  wpe = pho*np.power(wind, 3)/2

  return wpe