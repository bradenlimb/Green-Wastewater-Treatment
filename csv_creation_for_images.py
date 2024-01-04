#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 21:15:33 2023

@author: bradenlimb
"""

#%% Import Modules
from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
import sys
import os

import plotly.express as px
from urllib.request import urlopen
import json
from plotly.offline import plot

import geopandas as gpd

import shapefile as shp  # Requires the pyshp package
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pickle
from tqdm import tqdm

import datetime
begin_time = datetime.datetime.now()
# asdfa

#%%

# folder_use = '_2023-07-31c Final Results - No Pacific Islands'
# folder_use = '2023-08-07 Corrected Buffer Life P Level 2'
# carbon_tax = 20 # $/tonne
# conc_N = 8 # mg/L
# conc_P = 1 # mg/L


input_data = {}
input_data['Level 2'] = {
    'conc_N': 8, # mg/L
    'conc_P': 1, # mg/L
    'gray_tech_use': 'Level 2-1, A2O facility'
    }
input_data['Level 3'] = {
    'conc_N': 6, # mg/L
    'conc_P': 0.2, # mg/L
    'gray_tech_use': 'Level 4-2, MBR facility'
    }
input_data['Level 4'] = {
    'conc_N': 3, # mg/L
    'conc_P': 0.1, # mg/L
    'gray_tech_use': 'Level 4-2, MBR facility'
    }
input_data['Level 5'] = {
    'conc_N': 2, # mg/L
    'conc_P': 0.02, # mg/L
    'gray_tech_use': 'Level 5-2, MBR/RO facility'
    }

## TODO Change Level
level_use = 5
folder_use = f'2024-01-03 Level {level_use}'
conc_N = input_data[f'Level {level_use}']['conc_N'] # mg/L
conc_P = input_data[f'Level {level_use}']['conc_P'] # mg/L
gray_tech_use = input_data[f'Level {level_use}']['gray_tech_use']

carbon_tax = 20 # $/tonne

#%% Figure 1
huc_use = 'HUC6' # HUC6, HUC8, HUC10, or HUC12
# nutrient = 'Nitrogen'
# nutrient = 'Phosphorus'
# with open(f'results/{folder_use}/results_all_point_source_8concN_1concP_postGeoChanges_NorPconc_Land{huc_use}-Percent_GreenCombos_GrayLimits_ConcLimits.pkl', 'rb') as handle:
with open(f'results/{folder_use}/results_all_level-{level_use}_{huc_use}.pkl', 'rb') as handle:
    dict_in = pickle.load(handle)
df_data_in = dict_in['df_data_keep']
df_data_in.rename(columns = {'HUC 12 Code': 'HUC12'}, inplace=True)
cols_keep = [huc_use, 'Total Load (kg/yr)', 'Total Annual Flow (L/yr)', 'Mean Conc (mg/L)']
huc_list = list(df_data_in[huc_use].unique())
huc_list.remove(huc_list[0])
huc_list.sort()
df_data = pd.DataFrame(index = huc_list, columns = cols_keep)
df_data[huc_use] = df_data.index

# Define how to aggregate various fields
agg_functions = { 
                 'Total Load (kg/yr)': 'sum', 
                 'Total Annual Flow (L/yr)': 'sum',          
                 }

for nutrient in ['Nitrogen','Phosphorus']:
    # Create new DataFrame by combining rows with same id values
    df_data_in_nutrient = df_data_in.loc[df_data_in['Nutrient Type'] == nutrient]
    df_data = df_data_in_nutrient.groupby([huc_use]).aggregate(agg_functions)
    df_data['Mean Conc (mg/L)'] = df_data['Total Load (kg/yr)']*1e6/df_data['Total Annual Flow (L/yr)']
    # df_data[huc_use] = df_data.index
    
    # # Save the DataFrame to a CSV file
    df_data.to_csv(f'results/{folder_use}/nutrient_data_{huc_use}_{nutrient}.csv')

# asdf

#%% Figure 2

land_HUC = huc_use

# Import important HUC12 Data
with open('inputs/EPA EnviroAtlas/HUC12_Important_Data_v3_withAreasFixed.pkl', 'rb') as handle:
   df_importantData = pickle.load(handle)

# Remove characters from HUC12 to get other huc values
df_importantData['HUC6'] = df_importantData['HUC12'].str[:-6]
df_importantData['HUC8'] = df_importantData['HUC12'].str[:-4]
df_importantData['HUC10'] = df_importantData['HUC12'].str[:-2]

# df_importantData.loc[df_importantData['area_ha']==0,'area_ha'] = 36*258.999 # Add mean value to each huc12 if area is zero. Source: https://www.epa.gov/rps/step-2-decide-which-watershed-units-screen

# Define a lambda function to compute the weighted mean:
wm = lambda x: np.average(x, weights=df_importantData.loc[x.index, "area_ha"])

# Define how to aggregate various fields
agg_functions = { 
                 'area_ha': 'sum', 
                 'Cropland_pct': wm,       
                 'Pasture_pct': wm, 
                 'Wetlands_pct': wm, 
                 'StreamLength_km': 'sum', 
                 'StreamLength_TotalImpaired_km': 'sum', 
                 'StreamLength_NutImpaired_km': 'sum', 
                 'StreamLength_TempImpaired_km': 'sum', 
                 'Wastewater_AnnualFlow_mgy': 'sum', 
                 'TileDrain_Sum_m2': 'sum', 
                 'AgBuffer_pct': wm, 
                 'AgWetlandsPotential_pct': wm, 
                 'Fertilizer_kgN_ha_yr': wm,
                 'Elec_$perkWh': wm,
                 'NoTill_pct': wm,
                 'CoverCrop_pct': wm,
                 'N_kg/ha': wm,
                 'P_kg/ha': wm,
                 }

# Save original DF as raw DF
df_importantDataRaw = df_importantData.copy(deep=True)

# Create new DataFrame by combining rows with same id values
df_importantData = df_importantData.groupby([land_HUC]).aggregate(agg_functions)
df_importantData.reset_index(inplace=True)

df_importantData.loc[df_importantData['TileDrain_Sum_m2']>0, 'tile_drains'] = 1
df_importantData.loc[df_importantData['AgBuffer_pct']>0, 'buffer'] = 1
df_importantData[['Wetlands_pct','AgWetlandsPotential_pct']] = df_importantData[['Wetlands_pct','AgWetlandsPotential_pct']].fillna(0)
df_importantData.loc[(df_importantData['Wetlands_pct']>0) | (df_importantData['AgWetlandsPotential_pct']>0), 'wetlands'] = 1
df_importantData.loc[df_importantData['Fertilizer_kgN_ha_yr']>0, 'fertilizer'] = 1

df_importantData[['tile_drains', 'buffer', 'wetlands', 'fertilizer']] = df_importantData[['tile_drains', 'buffer', 'wetlands', 'fertilizer']].fillna(0)

# Save the DataFrame to a CSV file
df_importantData.to_csv(f'results/{folder_use}/important_data_{land_HUC}.csv')

# sadf

#%% Figure 3
# Mod 4, slide 88
# Mod 5, slie 95
# Mod 5, slie 104

# Import Green Treatment Inputs
# filename_inputs = 'inputs/model_inputs.xlsx'
# sheet_green = 'Green Water Treatment Illinois'
# # sheet_green = 'Green Water Treatment IlliTest'
# df_green = pd.read_excel(filename_inputs,
#                                 sheet_name = sheet_green,
#                                 index_col=1,
#                                 )

# # Import Results
# # xslx_path = f'results/{folder_use}/results_summaries_point_source_8concN_1concP_postGeoChanges_NorPconc_Land{land_HUC}-Percent_GreenCombos_GrayLimits_ConcLimits.xlsx'
# xslx_path = f'results/{folder_use}/results_summaries_level-{level_use}_{huc_use}.xlsx'
# df_results = pd.read_excel(xslx_path,
#                         # sheet_name='ghgAGLUC_per_gal-gCO2eq_per_gal',
#                         # index_col='Land Type'
#                         )

# Rename the Optimum Names

## Decided to just do this in Excel manually

#%% Figure 5

# Import Green Treatment Inputs
filename_inputs = 'inputs/model_inputs.xlsx'
sheet_green = 'Green Water Treatment Illinois'
# sheet_green = 'Green Water Treatment IlliTest'
sheet_green = 'Green Water Treatment Illin0'
df_green = pd.read_excel(filename_inputs,
                                sheet_name = sheet_green,
                                index_col=1,
                                )

green_options = list(df_green['Treatment Name'])[1:]

land_HUC = huc_use

# file_in = f'results/{folder_use}/results_all_point_source_8concN_1concP_postGeoChanges_NorPconc_Land{land_HUC}-Percent_GreenCombos_GrayLimits_ConcLimits.pkl'
file_in = f'results/{folder_use}/results_all_level-{level_use}_{huc_use}.pkl'

all_options = []

# Import important HUC12 Data
with open(file_in, 'rb') as handle:
   dict_resultsAll = pickle.load(handle)
   
for data_choice in ['minimal_cost','minimal_emissions']:
    df_resultsUse =    dict_resultsAll['dict_best_green_option'][data_choice]
    df_resultsUse2 =    dict_resultsAll['dict_green_treatments']['Cover Crop facility']
    
    # # Define a lambda function to compute the weighted mean:
    # wm = lambda x: np.average(x, weights=df_importantData.loc[x.index, "area_ha"])
    
    # Define how to aggregate various fields
    agg_functions = { 
        'eGrid': 'first',
        'N Total Load (kg/yr)': 'sum',
        'N Annual Flow (L/yr)': 'sum',
        'N Load Treated (kg/yr)': 'sum',
        'N Treated Cost ($/yr)': 'sum',
        'N Treated GWP (tonnes-CO2eq/yr)': 'sum',
        'P Total Load (kg/yr)': 'sum',
        'P Annual Flow (L/yr)': 'sum',
        'P Load Treated (kg/yr)': 'sum',
        'P Treated Cost ($/yr)': 'sum',
        'P Treated GWP (tonnes-CO2eq/yr)': 'sum',
        'Max Treated Cost ($/yr)': 'sum',
        'Max Treated GWP (tonnes-CO2eq/yr)': 'sum',
                     }
    
    # Save original DF as raw DF
    df_resultsUseRaw = df_resultsUse.copy(deep=True)
    
    # Create new DataFrame by combining rows with same id values
    df_resultsUse = df_resultsUse.groupby([land_HUC]).aggregate(agg_functions)
    df_resultsUse.reset_index(inplace=True)
    
    # def function ():
        
    for option_temp in green_options:
        option_temp_ab = df_green.loc[df_green['Treatment Name'] == option_temp, 'Abrev'].item()
        df_resultsUse[option_temp_ab] = 'None'
        
        
    multiHUC_N = []
    multiHUC_P = []
    # asdfasdf
    for hucUse in  tqdm(list(df_resultsUse[land_HUC])):
        df_tempN = df_resultsUseRaw.loc[(df_resultsUseRaw[land_HUC] == hucUse) & (df_resultsUseRaw['N Load Treated (kg/yr)'] > 0)]
        n_options = list(set(list(df_tempN['N Option'])))
        if len(n_options) > 1:
            multiHUC_N.append(hucUse)
            if n_options[0] in green_options:
                n_options[0] = df_green.loc[df_green['Treatment Name'] == n_options[0], 'Abrev'].item()
            df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, 'N Option'] = n_options[0]
            for option_N in n_options[0].split('_'):
                if df_green.loc[df_green['Abrev'] == option_N, 'N Removal'].item() > 0:
                    df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, option_N] = 'Nitrogen'
        elif len(n_options) == 1:
            if n_options[0] in green_options:
                n_options[0] = df_green.loc[df_green['Treatment Name'] == n_options[0], 'Abrev'].item()
            df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, 'N Option'] = n_options[0]
            for option_N in n_options[0].split('_'):
                if df_green.loc[df_green['Abrev'] == option_N, 'N Removal'].item() > 0:
                    df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, option_N] = 'Nitrogen'
        else:
            df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, 'N Option'] = 'None'
        
        
        
        df_tempP = df_resultsUseRaw.loc[(df_resultsUseRaw[land_HUC] == hucUse) & (df_resultsUseRaw['P Load Treated (kg/yr)'] > 0)]
        p_options = list(set(list(df_tempP['P Option'])))
        if len(p_options) > 1:
            multiHUC_P.append(hucUse)
            if p_options[0] in green_options:
                p_options[0] = df_green.loc[df_green['Treatment Name'] == p_options[0], 'Abrev'].item()
            df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, 'P Option'] = p_options[0]
            for option_P in p_options[0].split('_'):
                if df_green.loc[df_green['Abrev'] == option_P, 'P Removal'].item() > 0:
                    if df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, option_P].item() == 'Nitrogen':
                        df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, option_P] = 'Both'
                    else:
                        df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, option_P] = 'Phosphorus'
        elif len(p_options) == 1:
            if p_options[0] in green_options:
                p_options[0] = df_green.loc[df_green['Treatment Name'] == p_options[0], 'Abrev'].item()
            df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, 'P Option'] = p_options[0]
            for option_P in p_options[0].split('_'):
                if df_green.loc[df_green['Abrev'] == option_P, 'P Removal'].item() > 0:
                    if df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, option_P].item() == 'Nitrogen':
                        df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, option_P] = 'Both'
                    else:
                        df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, option_P] = 'Phosphorus'
        else:
            df_resultsUse.loc[df_resultsUse[land_HUC] == hucUse, 'P Option'] = 'None'
            

    # Replace underscore with plus in the columns
    df_resultsUse['N Option'] = df_resultsUse['N Option'].str.replace('_', '+')
    df_resultsUse['P Option'] = df_resultsUse['P Option'].str.replace('_', '+')
    
    all_options += list(df_resultsUse['N Option'].unique()) + list(df_resultsUse['P Option'].unique())
    
    
    
    # Save the DataFrame to a CSV file
    df_resultsUse.to_csv(f'results/{folder_use}/nutrient_choice_{data_choice}_{land_HUC}.csv')
    
    
all_options = list(set(all_options))
all_options.remove('None')
all_options.sort()

# df_alloptions = pd.DataFrame(index = all_options)

# for data_choice in ['minimal_cost','minimal_emissions']:
#     df_resultsUse =    dict_resultsAll['dict_best_green_option'][data_choice]
#     df_resultsUse2 =    dict_resultsAll['dict_green_treatments']['Cover Crop facility']
    
#     # # Define a lambda function to compute the weighted mean:
#     # wm = lambda x: np.average(x, weights=df_importantData.loc[x.index, "area_ha"])
    
#     # Define how to aggregate various fields
#     agg_functions = { 
#         'eGrid': 'first',
#         'N Total Load (kg/yr)': 'sum',
#         'N Annual Flow (L/yr)': 'sum',
#         'N Load Treated (kg/yr)': 'sum',
#         'N Treated Cost ($/yr)': 'sum',
#         'N Treated GWP (tonnes-CO2eq/yr)': 'sum',
#         'P Total Load (kg/yr)': 'sum',
#         'P Annual Flow (L/yr)': 'sum',
#         'P Load Treated (kg/yr)': 'sum',
#         'P Treated Cost ($/yr)': 'sum',
#         'P Treated GWP (tonnes-CO2eq/yr)': 'sum',
#         'Max Treated Cost ($/yr)': 'sum',
#         'Max Treated GWP (tonnes-CO2eq/yr)': 'sum',
#                      }
    
#     # Save original DF as raw DF
#     df_resultsUseRaw = df_resultsUse.copy(deep=True)
    
#     # Create new DataFrame by combining rows with same id values
#     df_resultsUse = df_resultsUse.groupby([land_HUC]).aggregate(agg_functions)
#     df_resultsUse.reset_index(inplace=True)
    
#     for option in all_options:
#         df_tempNCount = df_resultsUse.loc(df_resultsUse['N Option'])
#         df_tempPCount = 
#         df_alloptions.loc(option, f'{data_choice}_N')

# asdf

#%% eGrid Plot
huc_use_egrid = 'HUC12'

if huc_use_egrid == 'HUC8':
    egrid_filename = ''
elif huc_use_egrid == 'HUC12':
    egrid_filename = 'inputs/eGrid/HUC12_w_eGrid_v3_smalldf.pkl'

land_HUC = huc_use

with open(egrid_filename, 'rb') as handle:
    egrid_HUC = pickle.load(handle)    
    
egrid_HUC['HUC10'] = egrid_HUC['HUC12'].str[:-2]  
egrid_HUC['HUC6'] = egrid_HUC['HUC12'].str[:-6]  
# Define how to aggregate various fields
agg_functions = { 
                 'eGrid': 'first', 
                 'States': 'first',       
                 }

# Create new DataFrame by combining rows with same id values
egrid_HUC = egrid_HUC.groupby([land_HUC]).aggregate(agg_functions)
egrid_HUC.to_csv(f'results/{folder_use}/eGrid_HUC_{land_HUC}.csv')  


#%% Find Carbon Financing Percentage

land_HUC = huc_use

# carbon_tax = 20 # $/tonne

# file_in = f'results/{folder_use}/results_all_point_source_8concN_1concP_postGeoChanges_NorPconc_Land{land_HUC}-Percent_GreenCombos_GrayLimits_ConcLimits.pkl'
file_in = f'results/{folder_use}/results_all_level-{level_use}_{huc_use}.pkl'

# Import important HUC12 Data
with open(file_in, 'rb') as handle:
   dict_resultsAll = pickle.load(handle)
   
df_gray_useA2O = dict_resultsAll['dict_gray_treatments'][gray_tech_use] 
   
dict_comp = {}


for data_choice in ['minimal_cost','minimal_emissions']:
    df_green_use =  dict_resultsAll['dict_best_green_option'][data_choice]
    
    df_temp = pd.DataFrame()
    df_temp[[land_HUC,'Gray Cost ($/yr)', 'Gray Emissions (tonnes-CO2eq/yr)']] = df_gray_useA2O[[land_HUC,'Max Treated Cost ($/yr)','Max Treated GWP (tonnes-CO2eq/yr)']]
    df_temp[['Green Cost ($/yr)', 'Green Emissions (tonnes-CO2eq/yr)']] = df_green_use[['Max Treated Cost ($/yr)','Max Treated GWP (tonnes-CO2eq/yr)']]
    df_temp['Emissions Saved (tonnes-CO2eq/yr)'] = df_temp['Gray Emissions (tonnes-CO2eq/yr)'] - df_temp['Green Emissions (tonnes-CO2eq/yr)']
    
    df_temp[['N Total Load (kg/yr)', 'N Annual Flow (L/yr)', 'N Load Treated (kg/yr)', 'N Current Mean Conc (mg/L)', 'P Total Load (kg/yr)', 'P Annual Flow (L/yr)', 'P Load Treated (kg/yr)', 'P Current Mean Conc (mg/L)']] = df_green_use[['N Total Load (kg/yr)', 'N Annual Flow (L/yr)', 'N Load Treated (kg/yr)', 'N Current Mean Conc (mg/L)', 'P Total Load (kg/yr)', 'P Annual Flow (L/yr)', 'P Load Treated (kg/yr)', 'P Current Mean Conc (mg/L)']]
    
    # Custom lambda function to sum 'N Annual Flow (L/yr)' if 'N Current Mean Conc (mg/L)' > 8
    custom_sum_N = lambda x: x[df_temp['N Current Mean Conc (mg/L)'] > conc_N].sum()
    custom_sum_P = lambda x: x[df_temp['P Current Mean Conc (mg/L)'] > conc_P].sum()
    
    
    agg_functions = { 
        'Gray Cost ($/yr)': 'sum',
        'Gray Emissions (tonnes-CO2eq/yr)': 'sum',
        'Green Cost ($/yr)': 'sum',
        'Green Emissions (tonnes-CO2eq/yr)': 'sum',
        'Emissions Saved (tonnes-CO2eq/yr)': 'sum',
        'N Total Load (kg/yr)': custom_sum_N,
        'N Annual Flow (L/yr)': custom_sum_N,
        'N Load Treated (kg/yr)': custom_sum_N,
        'P Total Load (kg/yr)': custom_sum_P,
        'P Annual Flow (L/yr)': custom_sum_P,
        'P Load Treated (kg/yr)': custom_sum_P,
                     }
    
    # Save original DF as raw DF
    df_tempRaw = df_temp.copy(deep=True)
    
    # Create new DataFrame by combining rows with same id values
    df_temp = df_temp.groupby([land_HUC]).aggregate(agg_functions)
    df_temp.reset_index(inplace=True)
    
    # Calculated Needed Treated Load
    df_temp['N Load Treated Needed (kg/yr)'] = df_temp['N Total Load (kg/yr)'] - conc_N * df_temp['N Annual Flow (L/yr)'] / 1e6
    df_temp['N Treated Percent (%)'] = df_temp['N Load Treated (kg/yr)'] / df_temp['N Load Treated Needed (kg/yr)'] * 100
    
    df_temp['P Load Treated Needed (kg/yr)'] = df_temp['P Total Load (kg/yr)'] - conc_P * df_temp['P Annual Flow (L/yr)'] / 1e6
    df_temp['P Treated Percent (%)'] = df_temp['P Load Treated (kg/yr)'] / df_temp['P Load Treated Needed (kg/yr)'] * 100
    
    
    # Savings from carbon financing
    df_temp['Carbon Financing ($/yr)'] = df_temp['Emissions Saved (tonnes-CO2eq/yr)'] * carbon_tax
    df_temp['Percent Costs (%)'] = df_temp['Carbon Financing ($/yr)'] / df_temp['Green Cost ($/yr)'] * 100
    
    # Save the DataFrame to a CSV file
    df_temp.to_csv(f'results/{folder_use}/carbon_financing_{data_choice}_{land_HUC}_{carbon_tax}.csv')
    dict_comp[data_choice] = df_temp
    
    
#%% Find the number of facilities for treatment and each nutrient
df_green_use =  dict_resultsAll['dict_best_green_option']['minimal_cost']
df_treated_both = df_green_use.loc[(df_green_use['N Current Mean Conc (mg/L)'] >= conc_N) & (df_green_use['P Current Mean Conc (mg/L)'] >= conc_P)]
df_treated_N = df_green_use.loc[(df_green_use['N Current Mean Conc (mg/L)'] >= conc_N) ]
df_treated_P = df_green_use.loc[(df_green_use['P Current Mean Conc (mg/L)'] >= conc_P)]   
df_treated_neither = df_green_use.loc[(df_green_use['N Current Mean Conc (mg/L)'] < conc_N) & (df_green_use['P Current Mean Conc (mg/L)'] < conc_P)]
 

#%% Calculate Farmer Incentive Payments
farmer_incentive = 31 * 2.47105 # $/acre converted to $/ha 

different_costs = ['Buffers','Wetlands', 'Bioreactors']
df_green_use =  dict_resultsAll['dict_best_green_option']['minimal_cost']
df_green_use_FI = df_green_use.copy(deep=True)
df_green_use_FI['max_area'] = df_green_use_FI[['N Treatment Area (ha)','P Treatment Area (ha)']].max(axis=1)

for facility in tqdm(df_green_use_FI.index.tolist()):
    if (df_green_use_FI.loc[facility, 'N Option'] in different_costs) and (df_green_use_FI.loc[facility, 'P Option'] in different_costs):
        if df_green_use_FI.loc[facility, 'N Option'] == df_green_use_FI.loc[facility, 'P Option']:
            df_green_use_FI.loc[facility, 'FI_cost'] = df_green_use_FI.loc[facility, 'max_area'] * farmer_incentive * df_green.loc[df_green['Treatment Name'] == df_green_use_FI.loc[facility, 'N Option'], 'Farmer Incentive'].item()
        else:
            print(facility, df_green_use_FI.loc[facility, 'N Option'], df_green_use_FI.loc[facility, 'P Option'])
    else:
        df_green_use_FI.loc[facility, 'FI_cost'] = df_green_use_FI.loc[facility, 'max_area'] * farmer_incentive

df_green_use_FI['FI_cost_pct'] = df_green_use_FI['FI_cost'] / df_green_use_FI['Max Treated Cost ($/yr)']
print(df_green_use_FI['FI_cost'].sum())

#%% Carbon Financing V2

land_HUC = huc_use

# file_in = f'results/{folder_use}/results_facility_point_source_8concN_1concP_postGeoChanges_NorPconc_Land{land_HUC}-Percent_GreenCombos_GrayLimits_ConcLimits.xlsx'
# file_in = f'results/{folder_use}/results_facility_point_source_{int(conc_N)}concN_{int(conc_P)}concP_postGeoChanges_NorPconc_Land{land_HUC}-Percent_GreenCombos_GrayLimits_ConcLimits.xlsx'

# df_greencost = pd.read_excel(file_in,
#                                 sheet_name = '72',
#                                 index_col=0,
#                                 )

# df_grayA2O = pd.read_excel(file_in,
#                                 sheet_name = '64',
#                                 index_col=0,
#                                 )

# file_in = f'results/{folder_use}/results_all_point_source_8concN_1concP_postGeoChanges_NorPconc_Land{land_HUC}-Percent_GreenCombos_GrayLimits_ConcLimits.pkl'
file_in = f'results/{folder_use}/results_all_level-{level_use}_{huc_use}.pkl'


# Import important HUC12 Data
with open(file_in, 'rb') as handle:
   dict_resultsAll = pickle.load(handle)
   
data_choice2 = 'minimal_cost'
df_greencost =  dict_resultsAll['dict_best_green_option'][data_choice2]  
df_grayA2O = dict_resultsAll['dict_gray_treatments'][gray_tech_use] 



options_n = list(set(list(df_greencost['N Option'])))
options_n.sort()
options_p = list(set(list(df_greencost['P Option'])))
options_p.sort()

options_all = list(set(options_n + options_p))

options_all_dict = {}
max_green_cost_name = 'Combined'
options_all_abr = [max_green_cost_name]
for option in options_all:
    if option in green_options:
        options_all_dict[option] = df_green.loc[df_green['Treatment Name'] == option, 'Abrev'].item()
        options_all_abr.append(df_green.loc[df_green['Treatment Name'] == option, 'Abrev'].item())
    else:
        options_all_dict[option] = option
        options_all_abr.append(option)
        
df_greencost['cost_sum'] =  df_greencost['N Treated Cost ($/yr)'] + df_greencost['P Treated Cost ($/yr)']
df_greencost['n_cost_portion'] =  df_greencost['N Treated Cost ($/yr)']  / df_greencost['cost_sum'] * df_greencost['Max Treated Cost ($/yr)']
df_greencost['p_cost_portion'] =  df_greencost['P Treated Cost ($/yr)']  / df_greencost['cost_sum'] * df_greencost['Max Treated Cost ($/yr)']
  
df_greencost['emissions_sum'] =  df_greencost['N Treated GWP (tonnes-CO2eq/yr)'] + df_greencost['P Treated GWP (tonnes-CO2eq/yr)']
df_greencost['n_emissions_portion'] =  df_greencost['N Treated GWP (tonnes-CO2eq/yr)']  / df_greencost['emissions_sum'] * df_greencost['Max Treated GWP (tonnes-CO2eq/yr)']
df_greencost['p_emissions_portion'] =  df_greencost['P Treated GWP (tonnes-CO2eq/yr)']  / df_greencost['emissions_sum'] * df_greencost['Max Treated GWP (tonnes-CO2eq/yr)']

df_grayA2O['cost_sum'] =  df_grayA2O['N Treated Cost ($/yr)'] + df_grayA2O['P Treated Cost ($/yr)']
df_grayA2O['n_cost_portion'] =  df_grayA2O['N Treated Cost ($/yr)']  / df_grayA2O['cost_sum'] * df_grayA2O['Max Treated Cost ($/yr)']
df_grayA2O['p_cost_portion'] =  df_grayA2O['P Treated Cost ($/yr)']  / df_grayA2O['cost_sum'] * df_grayA2O['Max Treated Cost ($/yr)']
  
df_grayA2O['emissions_sum'] =  df_grayA2O['N Treated GWP (tonnes-CO2eq/yr)'] + df_grayA2O['P Treated GWP (tonnes-CO2eq/yr)']
df_grayA2O['n_emissions_portion'] =  df_grayA2O['N Treated GWP (tonnes-CO2eq/yr)']  / df_grayA2O['emissions_sum'] * df_grayA2O['Max Treated GWP (tonnes-CO2eq/yr)']
df_grayA2O['p_emissions_portion'] =  df_grayA2O['P Treated GWP (tonnes-CO2eq/yr)']  / df_grayA2O['emissions_sum'] * df_grayA2O['Max Treated GWP (tonnes-CO2eq/yr)']

  
        
df_financing_Results = pd.DataFrame(index = options_all_abr, 
                                    columns = ['n_count','n_treated','n_cost_green','n_cost_gray','n_gwp_green','n_gwp_gray','n_cost_cf', 'n_cost_w_cf','p_count','p_treated','p_cost_green','p_cost_gray','p_gwp_green','p_gwp_gray','p_cost_cf', 'p_cost_w_cf',])


df_financing_Results.loc[max_green_cost_name,'n_count'] = len(df_greencost)
df_financing_Results.loc[max_green_cost_name,'n_treated'] = df_greencost['N Load Treated (kg/yr)'].sum()
df_financing_Results.loc[max_green_cost_name,'n_cost_green'] = df_greencost['n_cost_portion'].sum()
df_financing_Results.loc[max_green_cost_name,'n_cost_gray'] = df_grayA2O['n_cost_portion'].sum()
df_financing_Results.loc[max_green_cost_name,'n_gwp_green'] = df_greencost['n_emissions_portion'].sum()
df_financing_Results.loc[max_green_cost_name,'n_gwp_gray'] = df_grayA2O['n_emissions_portion'].sum()

df_financing_Results.loc[max_green_cost_name,'p_count'] = len(df_greencost)
df_financing_Results.loc[max_green_cost_name,'p_treated'] = df_greencost['P Load Treated (kg/yr)'].sum()
df_financing_Results.loc[max_green_cost_name,'p_cost_green'] = df_greencost['p_cost_portion'].sum()
df_financing_Results.loc[max_green_cost_name,'p_cost_gray'] = df_grayA2O['p_cost_portion'].sum()
df_financing_Results.loc[max_green_cost_name,'p_gwp_green'] = df_greencost['p_emissions_portion'].sum()
df_financing_Results.loc[max_green_cost_name,'p_gwp_gray'] = df_grayA2O['p_emissions_portion'].sum()


for option in options_all:
    index_n = df_greencost.loc[df_greencost['N Option'] == option].index.tolist()
    index_p = df_greencost.loc[df_greencost['P Option'] == option].index.tolist()
    
    option_out = options_all_dict[option]
    
    df_financing_Results.loc[option_out,'n_count'] = len(index_n)
    df_financing_Results.loc[option_out,'n_treated'] = df_greencost.loc[index_n,'N Load Treated (kg/yr)'].sum()
    df_financing_Results.loc[option_out,'n_cost_green'] = df_greencost.loc[index_n,'n_cost_portion'].sum()
    df_financing_Results.loc[option_out,'n_cost_gray'] = df_grayA2O.loc[index_n,'n_cost_portion'].sum()
    df_financing_Results.loc[option_out,'n_gwp_green'] = df_greencost.loc[index_n,'n_emissions_portion'].sum()
    df_financing_Results.loc[option_out,'n_gwp_gray'] = df_grayA2O.loc[index_n,'n_emissions_portion'].sum()
    
    df_financing_Results.loc[option_out,'p_count'] = len(index_p)
    df_financing_Results.loc[option_out,'p_treated'] = df_greencost.loc[index_n,'P Load Treated (kg/yr)'].sum()
    df_financing_Results.loc[option_out,'p_cost_green'] = df_greencost.loc[index_p,'p_cost_portion'].sum()
    df_financing_Results.loc[option_out,'p_cost_gray'] = df_grayA2O.loc[index_p,'p_cost_portion'].sum()
    df_financing_Results.loc[option_out,'p_gwp_green'] = df_greencost.loc[index_p,'p_emissions_portion'].sum()
    df_financing_Results.loc[option_out,'p_gwp_gray'] = df_grayA2O.loc[index_p,'p_emissions_portion'].sum()
    
df_financing_Results['n_cost_cf'] = (df_financing_Results['n_gwp_gray'] - df_financing_Results['n_gwp_green']) * carbon_tax
df_financing_Results['n_cost_w_cf'] = df_financing_Results['n_cost_green'] - df_financing_Results['n_cost_cf']

df_financing_Results['p_cost_cf'] = (df_financing_Results['p_gwp_gray'] - df_financing_Results['p_gwp_green']) * carbon_tax
df_financing_Results['p_cost_w_cf'] = df_financing_Results['p_cost_green'] - df_financing_Results['p_cost_cf']

df_financing_Results['n_treated_pct'] = df_financing_Results['n_treated'] / df_financing_Results.loc[max_green_cost_name,'n_treated'] * 100
df_financing_Results['p_treated_pct'] = df_financing_Results['p_treated'] / df_financing_Results.loc[max_green_cost_name,'p_treated'] * 100


df_financing_Results.to_csv(f'results/{folder_use}/carbon_financing_v2_{data_choice2}_{land_HUC}.csv')
