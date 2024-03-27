import numpy as np

data1 = np.load(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\wind_93_99_pacific_warm_pool_area.npz')
print(data1.files) #['time', 'lat', 'lon', 'eastward_wind0', 'northward_wind0', 'wind_vector_divergence0', 'wind_stress0', 'surface_downward_eastward_stress0', 'surface_downward_northward_stress0', 'wind_stress_curl0', 'wind_stress_divergence0', 'wind_speed_rms0', 'eastward_wind_rms0', 'northward_wind_rms0', 'sampling_length0', 'surface_type0']

time1 = data1['time']
lat1 = data1['lat']
lon1 = data1['lon']
eastward_wind1 = data1['eastward_wind0']
northward_wind1 = data1['northward_wind0']
wind_vector_divergence1 = data1['wind_vector_divergence0']
wind_stress1 = data1['wind_stress0']
surface_downward_eastward_stress1 = data1['surface_downward_eastward_stress0']
surface_downward_northward_stress1 = data1['surface_downward_northward_stress0']
wind_stress_curl1 = data1['wind_stress_curl0']
wind_stress_divergence1 = data1['wind_stress_divergence0']
wind_speed_rms1 = data1['wind_speed_rms0']
eastward_wind_rms1 = data1['eastward_wind_rms0']
northward_wind_rms1 = data1['northward_wind_rms0']
sampling_length1 = data1['sampling_length0']
surface_type1 = data1['surface_type0']


data2 = np.load(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\wind_00_05_pacific_warm_pool_area.npz')
print(data2.files) #['time', 'lat', 'lon', 'eastward_wind0', 'northward_wind0', 'wind_vector_divergence0', 'wind_stress0', 'surface_downward_eastward_stress0', 'surface_downward_northward_stress0', 'wind_stress_curl0', 'wind_stress_divergence0', 'wind_speed_rms0', 'eastward_wind_rms0', 'northward_wind_rms0', 'sampling_length0', 'surface_type0']

time2 = data2['time']
eastward_wind2 = data2['eastward_wind0']
northward_wind2 = data2['northward_wind0']
wind_vector_divergence2 = data2['wind_vector_divergence0']
wind_stress2 = data2['wind_stress0']
surface_downward_eastward_stress2 = data2['surface_downward_eastward_stress0']
surface_downward_northward_stress2 = data2['surface_downward_northward_stress0']
wind_stress_curl2 = data2['wind_stress_curl0']
wind_stress_divergence2 = data2['wind_stress_divergence0']
wind_speed_rms2 = data2['wind_speed_rms0']
eastward_wind_rms2 = data2['eastward_wind_rms0']
northward_wind_rms2 = data2['northward_wind_rms0']
sampling_length2 = data2['sampling_length0']
surface_type2 = data2['surface_type0']


data3 = np.load(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\wind_06_10_pacific_warm_pool_area.npz')
print(data3.files) #['time', 'lat', 'lon', 'eastward_wind0', 'northward_wind0', 'wind_vector_divergence0', 'wind_stress0', 'surface_downward_eastward_stress0', 'surface_downward_northward_stress0', 'wind_stress_curl0', 'wind_stress_divergence0', 'wind_speed_rms0', 'eastward_wind_rms0', 'northward_wind_rms0', 'sampling_length0', 'surface_type0']

time3 = data3['time']
eastward_wind3 = data3['eastward_wind0']
northward_wind3 = data3['northward_wind0']
wind_vector_divergence3 = data3['wind_vector_divergence0']
wind_stress3 = data3['wind_stress0']
surface_downward_eastward_stress3 = data3['surface_downward_eastward_stress0']
surface_downward_northward_stress3 = data3['surface_downward_northward_stress0']
wind_stress_curl3 = data3['wind_stress_curl0']
wind_stress_divergence3 = data3['wind_stress_divergence0']
wind_speed_rms3 = data3['wind_speed_rms0']
eastward_wind_rms3 = data3['eastward_wind_rms0']
northward_wind_rms3 = data3['northward_wind_rms0']
sampling_length3 = data3['sampling_length0']
surface_type3 = data3['surface_type0']


data4 = np.load(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\wind_11_15_pacific_warm_pool_area.npz')
print(data4.files) #['time', 'lat', 'lon', 'eastward_wind0', 'northward_wind0', 'wind_vector_divergence0', 'wind_stress0', 'surface_downward_eastward_stress0', 'surface_downward_northward_stress0', 'wind_stress_curl0', 'wind_stress_divergence0', 'wind_speed_rms0', 'eastward_wind_rms0', 'northward_wind_rms0', 'sampling_length0', 'surface_type0']

time4 = data4['time']
eastward_wind4 = data4['eastward_wind0']
northward_wind4 = data4['northward_wind0']
wind_vector_divergence4 = data4['wind_vector_divergence0']
wind_stress4 = data4['wind_stress0']
surface_downward_eastward_stress4 = data4['surface_downward_eastward_stress0']
surface_downward_northward_stress4 = data4['surface_downward_northward_stress0']
wind_stress_curl4 = data4['wind_stress_curl0']
wind_stress_divergence4 = data4['wind_stress_divergence0']
wind_speed_rms4 = data4['wind_speed_rms0']
eastward_wind_rms4 = data4['eastward_wind_rms0']
northward_wind_rms4 = data4['northward_wind_rms0']
sampling_length4 = data4['sampling_length0']
surface_type4 = data4['surface_type0']


data5 = np.load(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\wind_16_17_pacific_warm_pool_area.npz')
print(data5.files) #['time', 'lat', 'lon', 'eastward_wind0', 'northward_wind0', 'wind_vector_divergence0', 'wind_stress0', 'surface_downward_eastward_stress0', 'surface_downward_northward_stress0', 'wind_stress_curl0', 'wind_stress_divergence0', 'wind_speed_rms0', 'eastward_wind_rms0', 'northward_wind_rms0', 'sampling_length0', 'surface_type0']

time5 = data5['time']
eastward_wind5 = data5['eastward_wind0']
northward_wind5 = data5['northward_wind0']
wind_vector_divergence5 = data5['wind_vector_divergence0']
wind_stress5 = data5['wind_stress0']
surface_downward_eastward_stress5 = data5['surface_downward_eastward_stress0']
surface_downward_northward_stress5 = data5['surface_downward_northward_stress0']
wind_stress_curl5 = data5['wind_stress_curl0']
wind_stress_divergence5 = data5['wind_stress_divergence0']
wind_speed_rms5 = data5['wind_speed_rms0']
eastward_wind_rms5 = data5['eastward_wind_rms0']
northward_wind_rms5 = data5['northward_wind_rms0']
sampling_length5 = data5['sampling_length0']
surface_type5 = data5['surface_type0']


data6 = np.load(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\wind_18_pacific_warm_pool_area.npz')
print(data6.files) #['time', 'lat', 'lon', 'eastward_wind0', 'northward_wind0', 'wind_vector_divergence0', 'wind_stress0', 'surface_downward_eastward_stress0', 'surface_downward_northward_stress0', 'wind_stress_curl0', 'wind_stress_divergence0', 'wind_speed_rms0', 'eastward_wind_rms0', 'northward_wind_rms0', 'sampling_length0', 'surface_type0']

time6 = data6['time']
eastward_wind6 = data6['eastward_wind0']
northward_wind6 = data6['northward_wind0']
wind_vector_divergence6 = data6['wind_vector_divergence0']
wind_stress6 = data6['wind_stress0']
surface_downward_eastward_stress6 = data6['surface_downward_eastward_stress0']
surface_downward_northward_stress6 = data6['surface_downward_northward_stress0']
wind_stress_curl6 = data6['wind_stress_curl0']
wind_stress_divergence6 = data6['wind_stress_divergence0']
wind_speed_rms6 = data6['wind_speed_rms0']
eastward_wind_rms6 = data6['eastward_wind_rms0']
northward_wind_rms6 = data6['northward_wind_rms0']
sampling_length6 = data6['sampling_length0']
surface_type6 = data6['surface_type0']


data7 = np.load(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\wind_19_pacific_warm_pool_area.npz')
print(data7.files) #['time', 'lat', 'lon', 'eastward_wind0', 'northward_wind0', 'wind_vector_divergence0', 'wind_stress0', 'surface_downward_eastward_stress0', 'surface_downward_northward_stress0', 'wind_stress_curl0', 'wind_stress_divergence0', 'wind_speed_rms0', 'eastward_wind_rms0', 'northward_wind_rms0', 'sampling_length0', 'surface_type0']

time7 = data7['time']
eastward_wind7 = data7['eastward_wind0']
northward_wind7 = data7['northward_wind0']
wind_vector_divergence7 = data7['wind_vector_divergence0']
wind_stress7 = data7['wind_stress0']
surface_downward_eastward_stress7 = data7['surface_downward_eastward_stress0']
surface_downward_northward_stress7 = data7['surface_downward_northward_stress0']
wind_stress_curl7 = data7['wind_stress_curl0']
wind_stress_divergence7 = data7['wind_stress_divergence0']
wind_speed_rms7 = data7['wind_speed_rms0']
eastward_wind_rms7 = data7['eastward_wind_rms0']
northward_wind_rms7 = data7['northward_wind_rms0']
sampling_length7 = data7['sampling_length0']
surface_type7 = data7['surface_type0']


time = np.concatenate((time1, time2, time3, time4, time5, time6, time7), axis=0)
eastward_wind = np.concatenate((eastward_wind1, eastward_wind2, eastward_wind3, eastward_wind4, eastward_wind5, eastward_wind6, eastward_wind7), axis=0)
northward_wind = np.concatenate((northward_wind1, northward_wind2, northward_wind3, northward_wind4, northward_wind5, northward_wind6, northward_wind7), axis=0)
wind_vector_divergence = np.concatenate((wind_vector_divergence1, wind_vector_divergence2, wind_vector_divergence3,
                                         wind_vector_divergence4, wind_vector_divergence5, wind_vector_divergence6, wind_vector_divergence7), axis=0)
wind_stress = np.concatenate((wind_stress1, wind_stress2, wind_stress3, wind_stress4, wind_stress5, wind_stress6, wind_stress7), axis=0)
surface_downward_eastward_stress = np.concatenate((surface_downward_eastward_stress1, surface_downward_eastward_stress2, surface_downward_eastward_stress3,
                                                   surface_downward_eastward_stress4, surface_downward_eastward_stress5, surface_downward_eastward_stress6, surface_downward_eastward_stress7), axis=0)
surface_downward_northward_stress = np.concatenate((surface_downward_northward_stress1, surface_downward_northward_stress2, surface_downward_northward_stress3,
                                                    surface_downward_northward_stress4, surface_downward_northward_stress5, surface_downward_northward_stress6, surface_downward_northward_stress7), axis=0)
wind_stress_curl = np.concatenate((wind_stress_curl1, wind_stress_curl2, wind_stress_curl3, wind_stress_curl4, wind_stress_curl5, wind_stress_curl6, wind_stress_curl7), axis=0)
wind_stress_divergence = np.concatenate((wind_stress_divergence1, wind_stress_divergence2, wind_stress_divergence3, wind_stress_divergence4, wind_stress_divergence5, wind_stress_divergence6, wind_stress_divergence7), axis=0)
wind_speed_rms = np.concatenate((wind_speed_rms1, wind_speed_rms2, wind_speed_rms3, wind_speed_rms4, wind_speed_rms5, wind_speed_rms6, wind_speed_rms7), axis=0)
eastward_wind_rms = np.concatenate((eastward_wind_rms1, eastward_wind_rms2, eastward_wind_rms3, eastward_wind_rms4, eastward_wind_rms5, eastward_wind_rms6, eastward_wind_rms7), axis=0)
northward_wind_rms = np.concatenate((northward_wind_rms1, northward_wind_rms2, northward_wind_rms3, northward_wind_rms4, northward_wind_rms5, northward_wind_rms6, northward_wind_rms7), axis=0)
sampling_length = np.concatenate((sampling_length1, sampling_length2, sampling_length3, sampling_length4, sampling_length5, sampling_length6, sampling_length7), axis=0)
surface_type = np.concatenate((surface_type1, surface_type2, surface_type3, surface_type4, surface_type5, surface_type6, surface_type7), axis=0)


np.savez(r'D:\heat_wave\WEIO\expand_WEIO\other_variables\wind_93_19_pacific_warm_pool_area.npz', time = time, lat = lat1, lon = lon1, eastward_wind = eastward_wind, northward_wind = northward_wind,
         wind_vector_divergence = wind_vector_divergence, wind_stress = wind_stress, surface_downward_eastward_stress = surface_downward_eastward_stress, surface_downward_northward_stress = surface_downward_northward_stress,
         wind_stress_curl = wind_stress_curl, wind_stress_divergence = wind_stress_divergence, wind_speed_rms = wind_speed_rms, eastward_wind_rms = eastward_wind_rms, northward_wind_rms = northward_wind_rms,
         sampling_length = sampling_length, surface_type = surface_type)