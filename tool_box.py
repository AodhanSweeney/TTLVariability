import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy import stats
import statsmodels.api as sm
import pingouin as pg


def starter():
    # Project is stored here
    base_path = '/home/disk/p/aodhan/cf_physical_parameters_correlations'

    # Temperature Data
    temp_map_prof_files = glob.glob('/home/bdc2/aodhan/ROwetPrfMonthlyGriddedMapsCDAAC/*.npy')
    temp_map_prof_files = np.sort(temp_map_prof_files)
    temp_profs = np.array([np.load(temp_map_prof_files[yr]) for yr in range(len(temp_map_prof_files))])

    cpt_files = np.sort(glob.glob('/usb/cmm_gws/monthlyDryCPT/*'))
    cptcpz_cals = np.array([np.load(file_item) for file_item in cpt_files])
    cpts = cptcpz_cals[:,:,:,:,0]
    cpzs = cptcpz_cals[:,:,:,:,1]

    # Water Vapor Anomaly Data
    wv_raw = np.load('/home/disk/p/aodhan/cf_physical_parameters_correlations/fileCreators/h2o_calendar_raw.npy')

    # TTL Cirrus Total Cloud Fraction Data (dealt with separately from other cloud fraction indices)
    total_cf_map_files_old_old = glob.glob(base_path + '/aerosol_cloud_distinctions/cfmaps/TTLtotalcfMonthly_strataerosolremoved_*.npy')
    total_cf_map_files_old = []
    for file in total_cf_map_files_old_old:
        if len(file) == len(base_path + '/aerosol_cloud_distinctions/cfmaps/TTLtotalcfMonthly_strataerosolremoved_2006_06.npy'):
            total_cf_map_files_old.append(file)
    total_cf_map_files_ = np.sort(total_cf_map_files_old)
    total_cf = np.array([np.load(total_cf_map_files_[yr])[0] for yr in range(len(total_cf_map_files_))])
    empty_tcf_map = np.empty(np.shape(total_cf[:5]))
    empty_tcf_map[:] = np.NaN
    total_cf = np.concatenate((empty_tcf_map, total_cf), axis=0)
    total_cf = np.reshape(total_cf, (16,12,24,144))
    total_cf[10,1] = np.nanmean([total_cf[10,0], total_cf[10,2]], axis=0)
    
    # TTL Cirrus Profile Cloud Fraction Data (dealt with separately from other cloud fraction indices)
    base_path = '/home/disk/p/aodhan/cf_physical_parameters_correlations'
    profile_cf_map_files_old_old = glob.glob(base_path + '/aerosol_cloud_distinctions/cfmaps/TTLcfMonthlyProfiles_strataerosolremoved_*.npy')
    profile_cf_map_files_old = []
    for file in profile_cf_map_files_old_old:
        if len(file) == len(base_path + '/aerosol_cloud_distinctions/cfmaps/TTLcfMonthlyProfiles_strataerosolremoved_2006_06.npy'):
            profile_cf_map_files_old.append(file)
    profile_cf_map_files_ = np.sort(profile_cf_map_files_old)
    profile_cf = np.array([np.load(profile_cf_map_files_[yr])[0] for yr in range(len(profile_cf_map_files_))])
    empty_prof_map = np.empty(np.shape(profile_cf[:5]))
    empty_prof_map[:] = np.NaN
    profile_cf = np.concatenate((empty_prof_map, profile_cf), axis=0)
    profile_cf = np.reshape(profile_cf, (16,12,24,144,101))
    profile_cf[10,1] = np.nanmean([profile_cf[10,0], profile_cf[10,2]], axis=0)
    
    # Total Cloud Fraction above 14.5 km (not just TTL cirrus)
    allTCF_above_14p5_files = glob.glob('/home/bdc2/aodhan/CFmaps/MonthlyNoStratAerosolsTropRelative/TCFabove14p5MonthlyStratAerosolRemoved_*.npy')
    allTCF_above_14p5_files = np.sort(allTCF_above_14p5_files)
    allTCFMonthlyMaps = np.array([np.load(allTCF_above_14p5_files[yr])[0] for yr in range(len(allTCF_above_14p5_files))])
    EMPTYallTCFMonthlyMaps = np.empty(np.shape(allTCFMonthlyMaps[:5]))
    EMPTYallTCFMonthlyMaps[:] = np.NaN
    allTCFMonthlyMaps_2006_2020 = np.concatenate((EMPTYallTCFMonthlyMaps, allTCFMonthlyMaps), axis=0)
    allTCFCalendarMaps = np.reshape(allTCFMonthlyMaps_2006_2020, (16,12,24,144))

    # Large Scale Dynamics
    lsd_base_path = '/home/disk/p/aodhan/large_scale_dynamics/'
    mjo_index = np.load(lsd_base_path + 'mjo_vpm_pc2_index.npy')
    enso_index = np.load(lsd_base_path + 'Monthly_ERSSTv5_Niño_3p4_1979_2020.npy')
    bdc_eddy = np.load(lsd_base_path + 'bdc_25_poleward_06_20_100hpa_meridional_heatflux_weighted.npy')
    bdc_mf = np.load(base_path + '/fileCreators/MF_15NS_100hPa_regout_qbo2molead_enso4molead_1979_2020_newQBO.npy')
    qbo_index = np.load(base_path + '/fileCreators/u50hPa_10NS_1979_2021.npy')

    return(cpts, cpzs, total_cf, temp_profs, profile_cf, allTCFCalendarMaps, mjo_index, enso_index, qbo_index, bdc_mf, bdc_eddy, wv_raw)

def cpt_finder(temp_prof):
    cold_point_temps = np.nanmin(temp_prof, axis=4)
    cold_point_heights = (np.ma.masked_invalid(temp_prof).argmin(axis=4))/10 + 5.
    return(cold_point_temps, cold_point_heights)
    
def anomaly_finder(data_calendar):
    """ 
    Data should be indexed as (years, months, lats, lons)
    """
    seasonal_average = np.nanmean(data_calendar, axis=0)
    anomalies_calendar = data_calendar - seasonal_average
    return anomalies_calendar

def cf_profile_finder(cf_prof_maps):
    cf_prof_maps_ = np.sort(cf_prof_maps)
    profile_cf = np.array([np.load(cf_prof_maps_[yr])[0] for yr in range(len(cf_prof_maps_))])
    empty_prof_map = np.empty(np.shape(profile_cf[:5]))
    empty_prof_map[:] = np.NaN
    profile_cf = np.concatenate((empty_prof_map, profile_cf), axis=0)
    profile_cf = np.reshape(profile_cf, (15,12,24,144,221))
    profile_cf[10,1] = np.nanmean([profile_cf[10,0], profile_cf[10,2]], axis=0)
    profile_cf_anoms = profile_cf - np.nanmean(profile_cf, axis=0)
    profile_cf_anoms_zm = np.nanmean(np.reshape(profile_cf_anoms, (15,12,24*144,221)), axis=2)
    profile_cf_anoms_ts = np.reshape(profile_cf_anoms_zm, (180,221))[5:]
    return(profile_cf_anoms_ts)

def vert_temp_gradient(data_calendar):
    data_calendar = np.array(data_calendar)
    t_1 = data_calendar[:,:,:,:,1:]
    t_2 = data_calendar[:,:,:,:,:-1]
    dt_dz = (t_1 - t_2)/0.1 # K/km
    dt_dz_1 = dt_dz[:,:,:,:,1:]
    dt_dz_2 = dt_dz[:,:,:,:,:-1]
    dt_dz_new = np.nanmean([dt_dz_1, dt_dz_2], axis=0)
    empty = np.empty(np.shape(data_calendar[:,:,:,:,0]))
    empty[:] = np.NaN
    dt_dz_new = np.insert(dt_dz_new, 98, empty, axis=4)
    dt_dz_new = np.insert(dt_dz_new, 0, empty, axis=4)
    return(dt_dz_new)

def tropical_average(data_calendar):
    """
    This function assumes latitudes are along axis 2
    """
    range_30ns = [-28.75, 24, 0, 24]
    range_20ns = [-18.75, 16, 4, 20]
    range_15ns = [-13.75, 12, 6, 18]
    range_10ns = [-8.75, 8, 8, 16]
    
    bounds = range_15ns
    
    weights = np.cos(np.deg2rad(np.linspace(bounds[0], bounds[0]*-1, bounds[1])))
    if np.ndim(data_calendar) == 4:
        data_calendar_15ns = data_calendar[:,:,bounds[2]:bounds[3],:]
        data_calendar_weighted = data_calendar_15ns*weights[np.newaxis, np.newaxis, :, np.newaxis]
        shape = np.shape(data_calendar_weighted)
        tropical_reshaped = np.reshape(data_calendar_weighted, (shape[0], shape[1], shape[2]*shape[3]))
        tropical_mean = np.nansum(tropical_reshaped, axis=2)/(sum(weights)*144)
    elif np.ndim(data_calendar) == 5:
        data_calendar_15ns = data_calendar[:,:,bounds[2]:bounds[3],:,:]
        data_calendar_weighted = data_calendar_15ns*weights[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        shape = np.shape(data_calendar_weighted)
        tropical_reshaped = np.reshape(data_calendar_weighted, (shape[0], shape[1], shape[2]*shape[3], shape[4]))
        tropical_mean = np.nansum(tropical_reshaped, axis=2)/(sum(weights)*144)
    return(tropical_mean)

def regress_out(things_to_regress_out_of_ts, ts):
    X = pd.DataFrame(things_to_regress_out_of_ts)
    Y = ts
    X = sm.add_constant(X) # adding a constant

    lm = pg.linear_regression(X, Y, add_intercept=True, relimp=True)
    lm.round(3)
    
    model = lm.coef[0]
    for x in range(0, len(things_to_regress_out_of_ts.columns)):
        coef_idx = x + 1
        col = things_to_regress_out_of_ts.columns[x]
        model = model + things_to_regress_out_of_ts[col]*lm.coef[coef_idx]
    regressed_out_ts = Y - model
    return(regressed_out_ts)