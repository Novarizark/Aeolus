#/usr/bin/env python3
"""
    Build Observation Station Objects

    Class       
    ---------------
                obv
"""

import datetime
import pandas as pd 
import numpy as np
from scipy import interpolate

import sys
sys.path.append('../')
from utils import utils

import copy

print_prefix='lib.obv>>'

class obv:

    '''
    Construct Observation Station Objs

    Attributes
    -----------
    idx:        int, scalar
        in-situ station ID

    lat:        float, scalar
        in situ station lat

    lon:        float, scalar
        in situ station lon

    height:     float, scalar
        in situ station elevation above terrain

    t:          datetime obj
        in situ station observation time
    
    u0, v0:     float, scalar
        uv observation in station height
    
    prof_pvalue: float, scalar
        exponent in wind power law within near surf layer

    u_prof, v_prof: float, 1d, bottom_top
        uv profile at station on model layers

    Methods
    -----------

    '''
    
    def __init__(self, df_row, wind_prof_df, cfg, fields_hdl):
        """ construct obv obj """
        single_obv_examiner(df_row)
        
        (self.idx, self.lat, self.lon, self.z) = (df_row.Index, df_row.lat, df_row.lon, df_row.height)
        
        self.sta_name=df_row.attr1
        self.t=datetime.datetime.strptime(str(df_row.yyyymmddhhMM),'%Y%m%d%H%M')
        (self.u0, self.v0)=utils.wswd2uv(df_row.wind_speed, df_row.wind_dir)
        self.t0=df_row.temp
        
        # which layer is the obv located in
        self.iz=utils.get_closest_idx(fields_hdl.z.values,self.z)
        
        (self.rough_len, self.stab_lvl)=(cfg['CORE']['roughness_length'], cfg['CORE']['stability_level'])
        
        
        # get wind profile power law exponent value by linear search in the dataframe
        self.set_pvalue(wind_prof_df)
        
        # get wind and temp profile
        self.get_in_situ_prof(fields_hdl, cfg)

    def set_pvalue(self, wind_prof_df):
        """ search wind profile pvalue in table """

        x_org=wind_prof_df['roughness_length'].values
        y_org=wind_prof_df[self.stab_lvl].values
        interp=interpolate.interp1d(x_org,y_org)
        self.prof_pvalue=interp(self.rough_len)
                
    def get_in_situ_prof(self, fields_hdl, cfg):
        """ get in-situ wind profile in the observation station """
        
        # get temp profile used lapse rate
        self.t_lapse=float(cfg['CORE']['lapse_t'])
        
        # get the wind profile from template file
        # BUG FIX: Need to copy the field var as field will be changed in aeolus.cast
        (idx,idy)=utils.get_closest_idxy(fields_hdl.XLAT.values,fields_hdl.XLONG.values,self.lat,self.lon)
        self.u_prof=copy.copy(fields_hdl.U.values[:,idx,idy])
        self.v_prof=copy.copy(fields_hdl.V.values[:,idx,idy])
        self.t_prof=copy.copy(fields_hdl.T.values[:,idx,idy])
        self.p_prof=copy.copy(fields_hdl.p.values[:,idx,idy])

        # power law for all layers at first atempt 
        self.u_prof=[utils.wind_prof(self.u0, self.z, z, self.prof_pvalue) for z in fields_hdl.z.values]
        self.v_prof=[utils.wind_prof(self.v0, self.z, z, self.prof_pvalue) for z in fields_hdl.z.values]
        self.t_prof=[utils.temp_prof(self.t0, self.z, p, z, self.t_lapse) for z,p in zip(fields_hdl.z.values, self.p_prof)]



def single_obv_examiner(df_row):
    """ Examine single observational record """
    row_footprint='Parsing error in record:: yyyymmddhhMM: '+str(df_row.yyyymmddhhMM)+' lat: '+str(df_row.lat)+' lon: '+str(df_row.lon)
    row_footprint=row_footprint+' height: '+str(df_row.height)
    
    # test lat lon range
    if df_row.lat>90.0 or df_row.lat<-90 or df_row.lon>360 or df_row.lon<0 or df_row.height<-1000 or df_row.height>20000:
        utils.write_log(print_prefix+row_footprint, lvl=40)
        utils.throw_error(print_prefix, 'invalid lat, lon, or height range!')

    # test wind speed range
    if df_row.wind_speed>40.0 or df_row.wind_speed<0:
        utils.write_log(print_prefix+row_footprint, lvl=40)
        utils.throw_error(print_prefix, 'weird wind speed detected! wind_speed='+str(df_row.wind_speed))

    # test temp range
    if df_row.temp>70.0 or df_row.temp<-50:
        utils.write_log(print_prefix+row_footprint, lvl=40)
        utils.throw_error(print_prefix, 'weird temperature detected! temp='+str(df_row.temp))

def obv_examiner(obv_df, cfg):
    """ Examine the input observational data """
    # test the input length
    if len(obv_df)< 3:
        utils.throw_error(print_prefix,'At least 3 observational records are needed!')
    
    # test the necessary elements
    na_mtx=obv_df.iloc[:,0:7].isna()
    if na_mtx.any().any():
        print(obv_df)
        utils.throw_error(print_prefix, 'Missing value is not allowed through "yyyymmddhhMM":"temp_2m"!\n'+
                'please check in the above obv table!')

    interp_strt_t=datetime.datetime.strptime(cfg['CORE']['interp_strt_t'],'%Y%m%d%H%M')
    interp_end_t=interp_strt_t+datetime.timedelta(hours=int(cfg['CORE']['interp_t_length']))
    
    obv_tlist_int=obv_df.yyyymmddhhMM.values
    try:
        obv_tlist=[datetime.datetime.strptime(str(itm),'%Y%m%d%H%M') for itm in obv_tlist_int]
    except:
        utils.throw_error(print_prefix, 'datetime formate error, please use "%Y%m%d%H%M"')

    # delta time between obv times to interp strt time in min
    delta_strt_lst=[abs((interp_strt_t-tfrm).total_seconds())/60.0 for tfrm in obv_tlist]
    # delta time between obv times to interp end time in min
    delta_end_lst=[abs((interp_end_t-tfrm).total_seconds())/60.0 for tfrm in obv_tlist]

    if min(delta_strt_lst)>1440 or min(delta_end_lst)>1440:
        utils.write_log(print_prefix+'Invalid time overlap between obv input and config file! See below:',lvl=40)
        utils.write_log(print_prefix+'config start:'+interp_strt_t.strftime('%Y-%m-%d_%H:%M')+', config end:'+interp_end_t.strftime('%Y-%m-%d_%H:%M')+'; obv start:'+obv_tlist[0].strftime('%Y-%m-%d %H:%M')+', obv end:'+obv_tlist[-1].strftime('%Y-%m-%d %H:%M'),lvl=40)
        utils.throw_error(print_prefix,'Minimum delta time between obv timelist and config > 24 hr, please check input data')
  
    if min(delta_strt_lst)>180 or min(delta_end_lst)>180:
        utils.write_log(print_prefix+'Minimum delta time between obv timelist and interpolating time > 180 min ',lvl=30)


def set_upper_wind(fields_hdl, obv_lst):
    """ Set geostrophic and Ekman layer wind according to sounding in observation data, if any. """
    
    zlays=fields_hdl.z.values
    nz=zlays.shape[0]
    pbl_top=fields_hdl.pbl_top
    idz1=fields_hdl.near_surf_z_idx
    idz2=fields_hdl.geo_z_idx
    
    # x_org for interpolation
    x_org=[zlays[idz1], zlays[idz2]]
    umean,vmean=fields_hdl.U.mean(axis=(1,2)), fields_hdl.V.mean(axis=(1,2))
    geo_u, geo_v=umean.values[idz2], vmean.values[idz2]

    geo_u_arr=np.array([obv.u0 for obv in obv_lst if obv.z >=pbl_top])
    geo_v_arr=np.array([obv.v0 for obv in obv_lst if obv.z >=pbl_top])

    if len(geo_u_arr)>0:
        geo_u, geo_v=geo_u_arr.mean(), geo_v_arr.mean()
    
    for obv in obv_lst:
        if obv.z<=pbl_top:
            # Ekman layer, interpolate to geostrophic wind
            u_org=[obv.u_prof[idz1], geo_u]
            v_org=[obv.v_prof[idz1], geo_v]
            interp_u=interpolate.interp1d(x_org,u_org)
            interp_v=interpolate.interp1d(x_org,v_org)

            obv.u_prof[idz1:idz2]=interp_u(zlays[idz1:idz2])
            obv.v_prof[idz1:idz2]=interp_v(zlays[idz1:idz2])

            # free atm
            for iz in range(idz2,nz):
                obv.u_prof[iz], obv.v_prof[iz]=geo_u, geo_v


if __name__ == "__main__":
    pass
