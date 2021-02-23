#/usr/bin/env python3
"""Build Observation Station Objects"""

import datetime
import pandas as pd 
import numpy as np
from scipy import interpolate

import sys
sys.path.append('../')
from utils import utils


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
        
        (self.rough_len, self.stab_lvl)=(cfg['CORE']['roughness_length'], cfg['CORE']['stability_level'])
        
        # get wind profile power law exponent value by leaner search in the dataframe
        self.set_pvalue(wind_prof_df)
        
        self.get_in_situ_prof(fields_hdl)

    def set_pvalue(self, wind_prof_df):
        """ search wind profile pvalue in table """

        x_org=wind_prof_df['roughness_length'].values
        y_org=wind_prof_df[self.stab_lvl].values
        interp=interpolate.interp1d(x_org,y_org)
        self.prof_pvalue=interp(self.rough_len)
                
    def get_in_situ_prof(self, fields_hdl):
        """ get in-situ wind profile in the observation station """
        
        # get the wind profile from template file
        (idx,idy)=utils.get_closest_idxy(fields_hdl.XLAT_U.values,fields_hdl.XLONG_U.values,self.lat,self.lon)
        self.u_prof=fields_hdl.U.values[:,idx,idy]
        (idx,idy)=utils.get_closest_idxy(fields_hdl.XLAT_V.values,fields_hdl.XLONG_V.values,self.lat,self.lon)
        self.v_prof=fields_hdl.V.values[:,idx,idy]

        # power law within near surf layer 
        idz=fields_hdl.near_surf_z_idx+1
        self.u_prof[0:idz]=[utils.wind_prof(self.u0, self.z, z, self.prof_pvalue) for z in fields_hdl.z[0:idz].values]
        self.v_prof[0:idz]=[utils.wind_prof(self.u0, self.z, z, self.prof_pvalue) for z in fields_hdl.z[0:idz].values]
        
        # Ekman layer, interpolate to geostrophic wind
        idz2=fields_hdl.geo_z_idx+1
        
        x_org=[fields_hdl.z[idz-1], fields_hdl.z[idz2]]
        u_org=[self.u_prof[idz-1], self.u_prof[idz2]]
        v_org=[self.v_prof[idz-1], self.v_prof[idz2]]
        interp_u=interpolate.interp1d(x_org,u_org)
        interp_v=interpolate.interp1d(x_org,v_org)

        self.u_prof[idz:idz2]=geo_u=interp_u(fields_hdl.z.values[idz:idz2])
        self.v_prof[idz:idz2]=geo_u=interp_v(fields_hdl.z.values[idz:idz2])

if __name__ == "__main__":
    pass
