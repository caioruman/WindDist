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

from common_functions import interpPressure, calc_height

'''
  - Read the soundings data
  - Read the model data
  - Plot both on the same plot
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
      stnames.append(aa[1])
      sheights.append(float(aa[7]))

  datai = 1990
  dataf = 2010

  read_s = False

  exp = 'cPanCan011_675x540_SPN'

  #main_folder = '/pixel/project01/cruman/ModelData/PanArctic_0.5d_CanHisto_NOCTEM_RUN/CSV_RCP'
  main_folder = '/pixel/project01/cruman/ModelData/PanArctic_0.5d_ERAINT_NOCTEM_RUN/CSV_V2'
  main_folder = '/pixel/project01/cruman/ModelData/{0}/CSV'.format(exp)

  percentage = open('DatFiles/percentage_seasonal_soundings_{0}.txt'.format(exp), 'w')
  percentage.write("Station Neg Pos Neg1 Neg2 Pos1 Pos2 season type\n")

  # looping throught all the stations
  for lat, lon, name, sheight in zip(lats, lons, stnames, sheights):

    #for season, sname in zip([[12, 1, 2],[6, 7, 8],[3, 4, 5],[9, 10, 11]], ['DJF','JJA','MAM','SON']):
    for season, sname in zip([[12, 1, 2],[6, 7, 8]], ['DJF','JJA']):
      
      # Open the model data
      filepaths_n = []
      filepaths_p = []

      for month in season:
        print(name, month)
        
        for year in range(datai, dataf+1):

          # Open the .csv
          reT = re.compile(r'.*?{0}_.*?{1}{2:02d}_.*?_windpress.*?'.format(name.replace(',',"_"), year, month))
          
          #os.mkdir('{0}/outdir'.format(main_folder))
          #print('{0}/outdir/{1}/{3}_{1}{2:02d}_01_windpress_neg.csv'.format(main_folder, year, month, name.replace(',',"_")))
          #sys.exit()
          if not os.path.exists('{0}/outdir/{1}/{3}_{1}{2:02d}_01_windpress_neg.csv'.format(main_folder, year, month, name.replace(',',"_"))):
            # if already extracted the file, don't do it again
            t = tarfile.open('{0}/{1}.tar.gz'.format(main_folder, year), 'r')
            t.extractall('{0}/outdir'.format(main_folder), members=[m for m in t.getmembers() if reT.search(m.name)])          

          #print os.listdir('outdir')
          # Open the .csv
          #filepaths_n.extend(glob('CSV/*{1}*_windpress_neg.csv'.format(month, year)))        
          filepaths_n.extend(glob('{3}/outdir/{1}/*{2}_{1}{0:02d}_*_windpress_neg.csv'.format(month, year, name.replace(',',"_"), main_folder)))
          filepaths_p.extend(glob('{3}/outdir/{1}/*{2}_{1}{0:02d}_*_windpress_pos.csv'.format(month, year, name.replace(',',"_"), main_folder)))     

      # Reading the model data     
      print("testing for errors - line 88")      
      df_n = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_n), ignore_index=True)
      print("testing for errors - line 90") 
      df_p = pd.concat((pd.read_csv(f, index_col=0) for f in filepaths_p), ignore_index=True)

      #[10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
      # Delete the upper levels of the atmosphere. I need only up to 700 hPa.
      df_n = df_n.drop(columns=['300.0', '400.0', '500.0', '600.0', '700.0','Dates'])
      df_p = df_p.drop(columns=['300.0', '400.0', '500.0', '600.0','700.0','Dates'])
      df_n = df_n/1.944
      df_p = df_p/1.944
      df_n = df_n.reindex(sorted(df_n.columns), axis=1)
      df_p = df_p.reindex(sorted(df_p.columns), axis=1)

      p_neg_model = len(df_n.index)*100/(len(df_n.index) + len(df_p.index))
      p_pos_model = len(df_p.index)*100/(len(df_n.index) + len(df_p.index))

      # Open the soundings data
      # location: /pixel/project01/cruman/Data/Soundings/                 
      # Reading Soundings data
      print("testing for errors - soundings") 
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

      percentage.write("{0} {1:2.2f} {2:2.2f} {3:2.2f} {4:2.2f} {5:2.2f} {6:2.2f} {7} {8}\n".format(name, p_neg_model, p_pos_model, perc_model_n[0], perc_model_n[1], perc_model_p[0], perc_model_p[1], sname, 'model'))
      percentage.write("{0} {1:2.2f} {2:2.2f} {3:2.2f} {4:2.2f} {5:2.2f} {6:2.2f} {7} {8}\n".format(name, p_neg, p_pos, perc_n[0], perc_n[1], perc_p[0], perc_p[1], sname, 'sounding'))
      #sys.exit()

#  percentage.close()

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

  if ntype == 'model':
    y = [800.0, 850.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
  else:
    y = [850.0, 875.0, 900.0, 925.0, 950.0, 975.0, 1000.0]
    
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

def kmeans_probability(df):
  '''
    For now fixed at 2 clusters

    returns: Array of the centroids, the two histograms and % of each group
  '''
  kmeans = KMeans(n_clusters=2, random_state=0).fit(df)  
        
  # Getting the location of each group.
  pred = kmeans.predict(df)
  labels = np.equal(pred, 0)

  # Converting to numpy array
  df_a = np.array(df)

  # Dividing between the 2 clusters
  df_0 = df_a[labels,:]
  df_1 = df_a[~labels,:]  

  # Getting the probability distribution. Bins of 0.5 m/s
  hist_0, bins_0 = calc_histogram(df_0)
  hist_1, bins_1 = calc_histogram(df_1)

  # back do a dataframe

  # Getting the probability distribution. Kernel Density  
  #hist_0 = calc_kerneldensity(df_0)
  #hist_1 = calc_kerneldensity(df_1)

  #print(np.mean(df_0, axis=0), np.mean(df_1, axis=0), kmeans.cluster_centers_)

  return kmeans.cluster_centers_, [hist_0, hist_1], [df_0.shape[0]*100/df_a.shape[0], df_1.shape[0]*100/df_a.shape[0]]

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

def calc_histogram(df):

  hist_l = []
  bins = np.arange(0,50.25,0.5)
  for i in range(0, df.shape[1]):    
    hist, bins = np.histogram(df[:,i], bins=bins)
    hist_l.append(hist*100/sum(hist))

  return np.array(hist_l), bins
    




if __name__ == "__main__":
  main()
