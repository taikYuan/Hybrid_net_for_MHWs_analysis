# 数据合并
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob
from netCDF4 import Dataset, num2date
import netCDF4 as nc
import time
import datetime

path = r'G:\data\SSH\ssh_93_19\dt_global_allsat_phy_l4_19930101_20190101.nc'
nc0 = Dataset(path)
print(nc0.variables)  # time    latitude   longitude   sea_surface_height_above_geoid   surface_geostrophic_eastward_sea_water_velocity   surface_geostrophic_northward_sea_water_velocity
# sea_surface_height_above_sea_level   surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid   surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid

time0 = nc0.variables['time'][:]  # 815232
# print((time0 / 24) - 65377)
time0 = time0.data
time0 = time0  - 15705
print(time0) # [15706.]  # 15706 15707

lon = nc0.variables['longitude']  # current shape = (192,)
# print(lon.shape)#1440
lat = nc0.variables['latitude']  # current shape = (94,)

lon00 = lon[1300:1381] #  325.125 325.375 ... 344.875 345.125
lat00 = lat[319:401] # -10.125 -9.875 ... 9.875 10.125

print(lat00)

# # print(lat00)
# # lat lon
sea_surface_height_above_geoid0 = nc0.variables['adt'][:,319:401,1300:1381]  # lat  lon   # Absolute dynamic topography
 # The absolute dynamic topography is the sea surface height above geoid; the adt is obtained as follows: adt=sla+mdt where mdt
 # is the mean dynamic topography; see the product user manual for details

surface_geostrophic_eastward_sea_water_velocity0 = nc0.variables['ugos'][:,319:401,1300:1381]  # Absolute geostrophic velocity: zonal component
surface_geostrophic_northward_sea_water_velocity0 = nc0.variables['vgos'][:,319:401,1300:1381]   # Absolute geostrophic velocity: meridian component
sea_surface_height_above_sea_level0 = nc0.variables['sla'][:,319:401,1300:1381]   # Sea level anomaly
# The sea level anomaly is the sea surface height above mean sea surface; it is referenced to the [1993, 2012] period; see the product user manual for details
surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid0 = nc0.variables['ugosa'][:,319:401,1300:1381]  # Geostrophic velocity anomalies: zonal component
surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid0 = nc0.variables['vgosa'][:,319:401,1300:1381]   # Geostrophic velocity anomalies: meridian component


# # # # # # #
sea_surface_height_above_geoid0 = sea_surface_height_above_geoid0.data
surface_geostrophic_eastward_sea_water_velocity0 = surface_geostrophic_eastward_sea_water_velocity0.data
surface_geostrophic_northward_sea_water_velocity0 = surface_geostrophic_northward_sea_water_velocity0.data
sea_surface_height_above_sea_level0 = sea_surface_height_above_sea_level0.data
surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid0 = surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid0.data
surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid0 = surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid0.data


time0 = time0
# # #
todo1 = r'G:\data\SSH\ssh_93_19\*.nc'
# # # todo1 = todo1[1:]
list_nc = glob.glob(todo1)
list_nc = sorted(list_nc[1:])
# # print(list_nc)
l = len(list_nc)
print(l)  # 36520
list_nc0 = list_nc.copy()
# print(list_nc0)
for i in list_nc0:
    nc_file = i
    nc1 = Dataset(nc_file)
    time1 = nc1.variables['time'][:]
    #     print(time1)
    time1 = time1.data
    time1 = time1 - 15705
    # print(time1)
    sea_surface_height_above_geoid1 = nc1.variables['adt'][:,319:401,1300:1381]   # lat  lon   # Absolute dynamic topography
    # The absolute dynamic topography is the sea surface height above geoid; the adt is obtained as follows: adt=sla+mdt where mdt
    # is the mean dynamic topography; see the product user manual for details

    surface_geostrophic_eastward_sea_water_velocity1 = nc1.variables['ugos'][:,319:401,1300:1381]   # Absolute geostrophic velocity: zonal component
    surface_geostrophic_northward_sea_water_velocity1 = nc1.variables['vgos'][:,319:401,1300:1381]   # Absolute geostrophic velocity: meridian component
    sea_surface_height_above_sea_level1 = nc1.variables['sla'][:,319:401,1300:1381]   # Sea level anomaly
    # The sea level anomaly is the sea surface height above mean sea surface; it is referenced to the [1993, 2012] period; see the product user manual for details
    surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid1 = nc1.variables['ugosa'][:,319:401,1300:1381]   # Geostrophic velocity anomalies: zonal component
    surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid1 = nc1.variables['vgosa'][:,319:401,1300:1381]   # Geostrophic velocity anomalies: meridian component


    sea_surface_height_above_geoid1 = sea_surface_height_above_geoid1.data
    surface_geostrophic_eastward_sea_water_velocity1 = surface_geostrophic_eastward_sea_water_velocity1.data
    surface_geostrophic_northward_sea_water_velocity1 = surface_geostrophic_northward_sea_water_velocity1.data
    sea_surface_height_above_sea_level1 = sea_surface_height_above_sea_level1.data
    surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid1 = surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid1.data
    surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid1 = surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid1.data
    # print('np.mean(eastward_wind1):{}'.format(np.mean(eastward_wind1)))
    time0 = np.concatenate((time0, time1), axis=0)
    #     print(time0.shape)
    sea_surface_height_above_geoid0 = np.concatenate((sea_surface_height_above_geoid0, sea_surface_height_above_geoid1), axis=0)  # (13271, 1, 1)
    surface_geostrophic_eastward_sea_water_velocity0 = np.concatenate((surface_geostrophic_eastward_sea_water_velocity0, surface_geostrophic_eastward_sea_water_velocity1), axis=0)  # (13271, 1, 1)
    surface_geostrophic_northward_sea_water_velocity0 = np.concatenate((surface_geostrophic_northward_sea_water_velocity0, surface_geostrophic_northward_sea_water_velocity1), axis=0)
    sea_surface_height_above_sea_level0 = np.concatenate((sea_surface_height_above_sea_level0, sea_surface_height_above_sea_level1), axis=0)  # (13271, 1, 1)
    surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid0 = np.concatenate((surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid0, surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid1), axis=0)  # (13271, 1, 1)
    surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid0 = np.concatenate((surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid0, surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid1),axis=0)
    time = list(time0)
    print(time)
    for i in range(len(time) - 1):
        if time[i + 1] - time[i] != 1:
            print('错误')
            break
# # #
    np.savez(r'D:\heat_wave\atlantic\expand_area1\ssh\ssh_93_19_expand_atlantic_area.npz', time=time0, lat=lat00, lon=lon00,
             sea_surface_height_above_geoid = sea_surface_height_above_geoid0, surface_geostrophic_eastward_sea_water_velocity = surface_geostrophic_eastward_sea_water_velocity0,
             surface_geostrophic_northward_sea_water_velocity = surface_geostrophic_northward_sea_water_velocity0,
             sea_surface_height_above_sea_level  = sea_surface_height_above_sea_level0,
             surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid  = surface_geostrophic_eastward_sea_water_velocity_assuming_sea_level_for_geoid0,
             surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid  = surface_geostrophic_northward_sea_water_velocity_assuming_sea_level_for_geoid0
             )

