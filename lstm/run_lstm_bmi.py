import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

#import data_tools
from pathlib import Path
from netCDF4 import Dataset
# This is the BMI LSTM that we will be running
import bmi_lstm


# Define primary bmi config and input data file paths 
#bmi_cfg_file=Path('./bmi_config_files/01022500_hourly_all_attributes_forcings.yml')
USE_PATH = True
run_dir = Path.cwd().parent
basin_num = '03010655'
# bmi_cfg_file  = run_dir  / 'bmi_config_files/01022500_hourly_slope_mean_precip_temp.yml'
bmi_cfg_file  = run_dir  / f'bmi_config_files/{basin_num}_hourly_aorc.yml'
# sample_data_file = run_dir + 'data/usgs-streamflow-nldas_hourly.nc'
sample_data_file = run_dir / f'data/aorc_hourly/{basin_num}_1980_to_2024_agg_rounded.csv'

# creating an instance of an LSTM model
print('Creating an instance of an BMI_LSTM model object')
model = bmi_lstm.bmi_LSTM()

# Initializing the BMI
print('Initializing the BMI')
model.initialize(bmi_cfg_file)

# Get input data that matches the LSTM test runs
print('Gathering input data')
# sample_data = Dataset(sample_data_file, 'r')
sample_data = pd.read_csv(sample_data_file)

# model._var_name_units_map["land_surface_radiation~incoming~longwave__energy_flux"] = ['DLWRF_surface','W m-2']
# model._var_name_units_map["land_surface_air__pressure"] = ['PRES_surface','Pa'],
# model._var_name_units_map["atmosphere_air_water~vapor__relative_saturation"] = ['SPFH_2maboveground','kg kg-1']
# model._var_name_units_map["atmosphere_water__liquid_equivalent_precipitation_rate"] = ['APCP_surface','mm h-1']
# model._var_name_units_map["land_surface_radiation~incoming~shortwave__energy_flux"] = ['DSWRF_surface','W m-2']
# model._var_name_units_map["land_surface_air__temperature"] = ['TMP_2maboveground','degC'],
# model._var_name_units_map["land_surface_wind__x_component_of_velocity"] = ['UGRD_10maboveground','m s-1'],
# model._var_name_units_map["land_surface_wind__y_component_of_velocity"] = ['VGRD_10maboveground','m s-1'],


# Now loop through the inputs, set the forcing values, and update the model
print('Set values & update model for number of timesteps = 100')
for i, (DLWRF, PRES, SPFH, precip, DSWRF, temp, UGRD, VGRD) in enumerate(zip(
    list(sample_data['DLWRF_surface'].values),
    list(sample_data['PRES_surface'].values),
    list(sample_data['SPFH_2maboveground'].values),
    list(sample_data['APCP_surface'].values),
    list(sample_data['DSWRF_surface'].values),
    list(sample_data['TMP_2maboveground'].values),
    list(sample_data['UGRD_10maboveground'].values),
    list(sample_data['VGRD_10maboveground'].values)
)):
    model.set_value('land_surface_radiation~incoming~longwave__energy_flux',np.atleast_1d(DLWRF))
    model.set_value('land_surface_air__pressure',np.atleast_1d(PRES))
    model.set_value('atmosphere_air_water~vapor__relative_saturation',np.atleast_1d(SPFH))
    model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate',np.atleast_1d(precip))
    model.set_value('land_surface_radiation~incoming~shortwave__energy_flux',np.atleast_1d(DSWRF))
    model.set_value('land_surface_air__temperature',np.atleast_1d(temp))
    model.set_value('land_surface_wind__x_component_of_velocity',np.atleast_1d(UGRD))
    model.set_value('land_surface_wind__y_component_of_velocity',np.atleast_1d(VGRD))

    print(f"""
    Values set in the model:
    DLWRF (Downward Long Wave Radiation Flux): {DLWRF:.2f}
    PRES (Surface Pressure): {PRES:.2f}
    SPFH (Specific Humidity): {SPFH:.2f}
    precip (Precipitation): {precip:.2f}
    DSWRF (Downward Short Wave Radiation Flux): {DSWRF:.2f}
    temp (Temperature): {temp:.2f}
    UGRD (U-component of wind): {UGRD:.2f}
    VGRD (V-component of wind): {VGRD:.2f}
    """)
    model.update()

    dest_array = np.zeros(1)
    model.get_value('land_surface_water__runoff_volume_flux', dest_array)
    runoff = dest_array[0]

    print(' Streamflow (cms) at time {} ({}) is {:.2f}'.format(model.get_current_time(), model.get_time_units(), runoff))

    if model.t > 100:
        #print('Stopping the loop')
        break

# Finalizing the BMI
print('Finalizing the BMI')
model.finalize()
