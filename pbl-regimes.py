import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import listdir
from glob import glob
import cmocean
import tarfile
import os
import re
import shutil
from scipy import interpolate
from datetime import datetime

from common_functions import interpPressure, calc_height

'''
  - Read the soundings data
  - Read the model data (in the future)
  - Read the SHF data, and merge it.
  - Calculate the deltaT.
  - Divide the data, first by deltaT them by SHF
  - Clustering analysis in each pool.
'''

def main():

  lats = []
  lons = []
  stnames = []
  sheights = []

  stations = open('DatFiles/stations2.txt', 'r')
  for line in stations:
    aa = line.replace("\n", '').split(';')
    if (aa[0] != "#"):      
      lats.append(float(aa[3]))
      lons.append(float(aa[5]))
      stnames.append(aa[1].replace(',', '_'))
      sheights.append(float(aa[7]))

  datai = 2040
  dataf = 2069

  read_s = False

  col_names = ['300.0','400.0','500.0','600.0','700.0','800.0','850.0','900.0','925.0','950.0','975.0','1000.0']
  y_hpa = ['700.0','800.0','850.0','900.0','925.0','950.0','975.0','1000.0']

  exp = 'PanArctic_0.5d_CanHisto_NOCTEM_RUN'

  main_folder = '/pixel/project01/cruman/ModelData/{0}/CSV_RCP_old'.format(exp)

  print(main_folder)

  #percentage = open('DatFiles/percentage_seasonal_soundings_{0}.txt'.format(exp), 'w')
  #percentage.write("Station Neg Pos Neg1 Neg2 Pos1 Pos2 season type\n")

  # looping throught all the stations
  for lat, lon, name, sheight in zip(lats, lons, stnames, sheights):

    #for season, sname in zip([[12, 1, 2],[6, 7, 8],[3, 4, 5],[9, 10, 11]], ['DJF','JJA','MAM','SON']):
    for season, sname in zip([[12, 1, 2],[6, 7, 8]], ['DJF','JJA']):
      
      # Open the model data
      filepaths_n = []
      filepaths_p = []

      filepaths_tt_n = []
      filepaths_tt_p = []

      for month in season:
        print(name, month)
        
        for year in range(datai, dataf+1):

          # Open the .csv       
          filepaths_n.extend(glob('{3}/{1}/{2}_{1}{0:02d}_windpress_neg.csv'.format(month, year, name, main_folder)))
          filepaths_p.extend(glob('{3}/{1}/{2}_{1}{0:02d}_windpress_pos.csv'.format(month, year, name, main_folder)))     

          filepaths_tt_n.extend(glob('{3}/{1}/{2}_{1}{0:02d}_neg.csv'.format(month, year, name, main_folder)))
          filepaths_tt_p.extend(glob('{3}/{1}/{2}_{1}{0:02d}_pos.csv'.format(month, year, name, main_folder)))

          #print()
      y = calc_height(season, datai, dataf)

      # Creating the df from csv files     
      df_w_n = create_df(filepaths_n, col_names)
      df_w_p = create_df(filepaths_p, col_names)

      #create the SHF column
      df_w_n['SHF_w'] = -1
      df_w_p['SHF_w'] = 1

      df_tt_n = create_df(filepaths_tt_n, col_names, False)
      df_tt_p = create_df(filepaths_tt_p, col_names, False)

      df_tt_n['SHF_tt'] = -1
      df_tt_p['SHF_tt'] = 1

      # Calculating deltaT
      df_tt_n = calcDeltaT(df_tt_n)
      df_tt_p = calcDeltaT(df_tt_p)

      # Calculating wind at 80m
      df_w_n['wind_80'] = df_w_n.apply(interpWindHeight, axis=1, args=([y_hpa]))
      df_w_p['wind_80'] = df_w_p.apply(interpWindHeight, axis=1, args=([y_hpa]))
    
      #merging the dataframes
      frames = [df_tt_n, df_tt_p]
      df_tt = pd.concat(frames)
      df_tt = df_tt.rename(columns={'700.0':'700.0_tt','800.0':'800.0_tt','850.0':'850.0_tt',
                         '900.0':'900.0_tt','925.0':'925.0_tt','950.0':'950.0_tt',
                         '975.0':'975.0_tt','1000.0':'98_1000.0_tt'})
      df_tt = df_tt.reindex(sorted(df_tt.columns), axis=1)
      
      frames = [df_w_n, df_w_p]
      df_w = pd.concat(frames)
      df_w = df_w.rename(columns={'700.0':'700.0_w','800.0':'800.0_w','850.0':'850.0_w',
                         '900.0':'900.0_w','925.0':'925.0_w','950.0':'950.0_w',
                         '975.0':'975.0_w','1000.0':'98_1000.0_w','Dates':'Dates_w'})
      df_w = df_w.reindex(sorted(df_w.columns), axis=1)

      frames = [df_tt, df_w]
      df = pd.concat(frames, axis=1)

      cols_tt = ['700.0_tt', '800.0_tt', '850.0_tt', '900.0_tt', '925.0_tt', '950.0_tt', '975.0_tt', '98_1000.0_tt']
      cols_w = ['700.0_w', '800.0_w', '850.0_w', '900.0_w', '925.0_w', '950.0_w', '975.0_w', '98_1000.0_w']

      # Separate the df into deltaT and SHF, them apply cluster analysis on the wind profiles
      df_deltaT_p = df.loc[(df['deltaT'] > 0)] 
      df_deltaT_n = df.loc[(df['deltaT'] <= 0)]

      df_shf_p = df.loc[(df['SHF_w'] == 1)]
      df_shf_n = df.loc[(df['SHF_w'] == -1)]

      aux_p_w = df_shf_p[cols_w]
      aux_n_w = df_shf_n[cols_w]

      aux_p_t = df_deltaT_p[cols_w]
      aux_n_t = df_deltaT_n[cols_w]

      centroids_w_n, histo_w_n, perc_w_n, label_w_n, df_wn = kmeans_probability(aux_n_w)
      centroids_w_p, histo_w_p, perc_w_p, label_w_p, df_wp = kmeans_probability(aux_p_w)

      centroids_t_n, histo_t_n, perc_t_n, label_t_n, df_tn = kmeans_probability(aux_n_t)
      centroids_t_p, histo_t_p, perc_t_p, label_t_p, df_tp = kmeans_probability(aux_p_t)

      print(label_w_n.shape)
      print(label_w_p.shape)
      print(df.shape)

      # Add the labels to the df, and save it to a .csv to plot later.
      df_shf_n['label_wind_n'] = label_w_n
      df_shf_p['label_wind_p'] = label_w_p
      df_deltaT_n['label_tt_n'] = label_t_n
      df_deltaT_p['label_tt_p'] = label_t_p
      

      df_shf_n.to_csv('teste_shf_n_future_2040.csv')
      df_shf_p.to_csv('teste_shf_p_future_2040.csv')
      df_deltaT_n.to_csv('teste_deltaT_n_future_2040.csv')
      df_deltaT_p.to_csv('teste_deltaT_p_future_2040.csv')

      sys.exit()

      height_list = []
      height_list.append(y)
      height_list.append(y)
      height_list.append(y)
      height_list.append(y) 

      cent_t, histo_t, perc_t, shf_t = create_lists_preplot(centroids_t_n, centroids_t_p, 
                                                            histo_t_n, histo_t_p, 
                                                            perc_t_n, perc_t_p)
   
      plot_wind_seasonal(cent_t, histo_t, perc_t, shf_t, 
                         datai, dataf, name.replace(',',"_"), sname, 
                         season, height_list, 'model_tt')

      cent_w, histo_w, perc_w, shf_w = create_lists_preplot(centroids_w_n, centroids_w_p, 
                                                            histo_w_n, histo_w_p, 
                                                            perc_w_n, perc_w_p)
   
      plot_wind_seasonal(cent_w, histo_w, perc_w, shf_w, 
                         datai, dataf, name.replace(',',"_"), sname, 
                         season, height_list, 'model_w')

      sys.exit()
      #df_teste = df_w[y_hpa]
      
      #print(df.head())
      #print(df.shape)
      #print(df.columns)
      sys.exit()

      # to do
      # merge both dataframes (TT and Wind) then separate by deltaT and SHF
      # cluster analysis of each and them plot.


      

      p_neg_model = len(df_n.index)*100/(len(df_n.index) + len(df_p.index))
      p_pos_model = len(df_p.index)*100/(len(df_n.index) + len(df_p.index))

      # Open the soundings data
      # location: /pixel/project01/cruman/Data/Soundings/                 
      # Reading Soundings data
      df_height_n = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_height_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0).dropna()
      df_height_p = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_height_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0).dropna()
      df_temp_n = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_temp_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0).dropna()
      df_temp_p = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_temp_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0).dropna()
      df_wind_n = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_wind_neg.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0).dropna()
      df_wind_p = pd.read_csv('DatFiles/Soundings/{0}_{1}_{2}-{3}_wind_pos.csv'.format(name.replace(',',"_"), sname, datai, dataf), index_col=0).dropna()      
      
      df_height_n = df_height_n.drop(columns=['Date'])
      df_height_p = df_height_p.drop(columns=['Date'])

      height_mean_n = df_height_n.mean(axis=0)
      height_mean_p = df_height_p.mean(axis=0)
            
      df_wind_n = df_wind_n.drop(columns=['Date'])
      df_wind_p = df_wind_p.drop(columns=['Date'])
      df_wind_n = df_wind_n.reindex(sorted(df_wind_n.columns), axis=1)      
      df_wind_p = df_wind_p.reindex(sorted(df_wind_p.columns), axis=1)

      print(df_p.columns)
      print(df_wind_p.columns)

      # K-means for the soundings
      centroids_n, histo_n, perc_n = kmeans_probability(df_wind_n)
      centroids_p, histo_p, perc_p = kmeans_probability(df_wind_p)

      # K-means for the model data
      centroids_model_n, histo_model_n, perc_model_n = kmeans_probability(df_n)
      centroids_model_p, histo_model_p, perc_model_p = kmeans_probability(df_p)
      
      height_list = []
      height_list.append(height_mean_n.values)
      height_list.append(height_mean_n.values)
      height_list.append(height_mean_p.values)
      height_list.append(height_mean_p.values)         

      cent, histo, perc, shf = create_lists_preplot(centroids_n, centroids_p, histo_n, histo_p, perc_n, perc_p)

      plot_wind_seasonal(cent, histo, perc, shf, datai, dataf, name.replace(',',"_"), sname, season, height_list, 'soundings')

      cent, histo, perc, shf = create_lists_preplot(centroids_model_n, centroids_model_p, histo_model_n, histo_model_p, perc_model_n, perc_model_p)
   
      plot_wind_seasonal(cent, histo, perc, shf, datai, dataf, name.replace(',',"_"), sname, season, height_list, 'model')

      p_neg = len(df_wind_n.index)*100/(len(df_wind_n.index) + len(df_wind_p.index))
      p_pos = len(df_wind_p.index)*100/(len(df_wind_n.index) + len(df_wind_p.index))

      #plot_wind(centroids_p[0], histo_p[0], perc_p[0], datai, dataf, name, "positive_type1", sname)
      #plot_wind(centroids_p[1], histo_p[1], perc_p[1], datai, dataf, name, "positive_type2", sname)

      #percentage.write("{0} {1:2.2f} {2:2.2f} {3:2.2f} {4:2.2f} {5:2.2f} {6:2.2f} {7} {8}\n".format(name, p_neg_model, p_pos_model, perc_model_n[0], perc_model_n[1], perc_model_p[0], perc_model_p[1], sname, 'model'))
      #percentage.write("{0} {1:2.2f} {2:2.2f} {3:2.2f} {4:2.2f} {5:2.2f} {6:2.2f} {7} {8}\n".format(name, p_neg, p_pos, perc_n[0], perc_n[1], perc_p[0], perc_p[1], sname, 'sounding'))
      #sys.exit()

#  percentage.close()

def interpWindHeight(row, y_hpa):

  # For each row in the df, get the wind values and interpolate to 80m (or the height of choice)
  #print(row)
  dd = row['Dates']
  #print(dd)
  #print(y_hpa)
  
  month = int(dd[5:7])
  year = int(dd[0:4])

  #model_heights = getModelHeight(month, year)
  model_heights = calc_height([month], year, year, y_hpa)
  print(model_heights)
  # based on the date, check the height from the model files
  
  f = interpolate.interp1d(model_heights, row[y_hpa], kind='linear')
  #f = interpolate.interp1d("height from the model (file)", "values of temp from model", kind='linear')

  return f(80)

def calcDeltaT(df):

  values = df['925.0'] - df['1000.0']

  df['deltaT'] = values

  return df

def create_df(files, col_names, wind=True):

  df = pd.concat((pd.read_csv(f, index_col=0) for f in files), ignore_index=True)
  
  if wind:
    df[col_names] /= 1.944

  df = df.drop(columns=['300.0', '400.0', '500.0', '600.0'])
  df = df.reindex(sorted(df.columns), axis=1)

  return df

def create_lists_preplot(centroids_n, centroids_p, histo_n, histo_p, perc_n, perc_p):
  cent = []
  histo = []
  perc = []
  shf = []

  if (perc_n[0] > perc_n[1]):
    k = 0
    j = 1
  else:
    k = 1
    j = 0

  cent.append(centroids_n[k])
  cent.append(centroids_n[j])

  histo.append(histo_n[k])
  histo.append(histo_n[j])

  perc.append(perc_n[k])
  perc.append(perc_n[j])

  shf.append('SHF-')
  shf.append('SHF-')      

  if (perc_p[0] > perc_p[1]):
    k = 0
    j = 1
  else:
    k = 1
    j = 0

  cent.append(centroids_p[k])
  cent.append(centroids_p[j])

  histo.append(histo_p[k])
  histo.append(histo_p[j])

  perc.append(perc_p[k])
  perc.append(perc_p[j])

  shf.append('SHF+')
  shf.append('SHF+')

  return cent, histo, perc, shf

def plot_wind_seasonal(centroids, histo, perc, shf, datai, dataf, name, period, season, height, ntype):

  #if ntype == 'model':
  #  y = [800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  #else:
  #  y = [850.0, 875.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
    
  #y = calc_height(season, 1986, 2015, y)
  #x = np.arange(0,40,1)
  
  vmin=0
  vmax=15
  v = np.arange(vmin, vmax+1, 2)  

  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[28,16], sharex=True, sharey=True)

  for k, letter in zip(range(0,4), ['a', 'b', 'c', 'd']):

    x = np.arange(0,50,0.5)
    y = height[k]
    X, Y= np.meshgrid(x, y)

    subplt = '22{0}'.format(k+1)
    plt.subplot(subplt)

    #print(np.sum(histo[k], axis=1))
    #sys.exit()

    CS = plt.contourf(X, Y, histo[k], v, cmap='cmo.haline', extend='max')
    CS.set_clim(vmin, vmax)
    plt.gca().invert_yaxis()
    plt.plot(centroids[k], y, color='white', marker='o', lw=4, markersize=10, markeredgecolor='k')
    #if (k%2) == 1:
          
    #CB = plt.colorbar(CS, extend='both', ticks=v)
    #CB.ax.tick_params(labelsize=20)
    plt.xlim(0,30)
    plt.ylim(min(y),max(y))
    plt.xticks(np.arange(0,31,5), fontsize=20)
    plt.yticks(np.arange(0,1400,100), fontsize=20)
    plt.title('({0}) {1:2.2f} % {2}'.format(letter, perc[k], shf[k]), fontsize='20')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Height (m)')

  
  
  #cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
  cax = fig.add_axes([0.92, 0.1, 0.02, 0.8]) 
  CB = plt.colorbar(CS, cax=cax, extend='both', ticks=v)  
  CB.ax.tick_params(labelsize=20)
  #plt.tight_layout()
  plt.savefig('Images/Soundings/Soun_{0}_{1}{2}_{3}_{4}.png'.format(name, datai, dataf, period, ntype), bbox_inches='tight')
  plt.close()
  #sys.exit()

  return None  

def kmeans_probability(df, _bin=0.5):
  '''
    For now fixed at 2 clusters

    returns: Array of the centroids, the two histograms and % of each group

    receive 2 df:
     - One with the hpa Level columns to fit
     - One with the other Dates, to return the dates where each cluster is, so I can apply to the original dataframe
    Or just one df and I return the labels array and make that a column in the dataframe
  '''
  kmeans = KMeans(n_clusters=2, random_state=0).fit(df)  
        
  # Getting the location of each group.
  pred = kmeans.predict(df)
  labels = np.equal(pred, 0)

  # Converting to numpy array
  df_a = np.array(df)

  df['label'] = labels

  # Dividing between the 2 clusters
  df_0 = df_a[labels,:]
  df_1 = df_a[~labels,:]  

  # Getting the probability distribution. Bins of 0.5 m/s
  hist_0, bins_0 = calc_histogram(df_0, _bin)
  hist_1, bins_1 = calc_histogram(df_1, _bin)

  # back do a dataframe

  # Getting the probability distribution. Kernel Density  
  #hist_0 = calc_kerneldensity(df_0)
  #hist_1 = calc_kerneldensity(df_1)

  #print(np.mean(df_0, axis=0), np.mean(df_1, axis=0), kmeans.cluster_centers_)

  return kmeans.cluster_centers_, [hist_0, hist_1], [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]], labels, df

def calc_kerneldensity(df):
  hist_aux = []
  for i in range(0,df.shape[1]):
      kde_skl = KernelDensity(bandwidth=0.4)
      #aux = np.array(df_n['1000.0'])
      aux = np.copy(df[:,i])
      aux_grid2 = np.linspace(0,50,100)
      kde_skl.fit(aux[:, np.newaxis])
      log_pdf = kde_skl.score_samples(aux_grid2[:, np.newaxis])
      #print(np.exp(log_pdf)*100)
      hist_aux.append(np.exp(log_pdf)*100)

  return hist_aux

def calc_histogram(df, _bin=0.5):

  hist_l = []
  bins = np.arange(0,50.25,_bin)
  for i in range(0, df.shape[1]):    
    hist, bins = np.histogram(df[:,i], bins=bins)
    hist_l.append(hist*100/sum(hist))

  return np.array(hist_l), bins
    




if __name__ == "__main__":
  main()
