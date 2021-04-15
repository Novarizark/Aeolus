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

def obv_examiner(obv_df):
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

def set_upper_wind(fields_hdl, obv_lst):
    """ Set geostrophic and Ekman layer wind according to sounding in observation data, if any. """
    
    zlays=fields_hdl.z.values
    nz=zlays.shape[0]
    pbl_top=fields_hdl.pbl_top
    idz1=fields_hdl.near_surf_z_idx
    idz2=fields_hdl.geo_z_idx
    x_org=[zlays[idz1], zlays[idz2]]
    
    geo_u_arr=np.array([obv.u0 for obv in obv_lst if obv.z >=pbl_top])
    geo_v_arr=np.array([obv.v0 for obv in obv_lst if obv.z >=pbl_top])

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
