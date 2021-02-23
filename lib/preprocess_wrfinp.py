#/usr/bin/env python
"""Preprocessing the WRF input file"""

import datetime
import xarray as xr
import gc
import netCDF4 as nc4
import wrf  
from copy import copy
from scipy import interpolate
import sys
sys.path.append('../')
from utils import utils

print_prefix='lib.preprocess_wrfinp>>'

class wrf_mesh:

    '''
    Construct grid info and UVW mesh template
    
    Attributes
    -----------
    dx:         float, scalar
        discritized spacing in x-direction (m)

    dy:         float, scalar
        discritized spacing in y-direction (m)

    U:          float, 3d
        zonal wind (m/s)

    V:          float, 3d
        meridional wind (m/s)

    W:          float, 3d
        vertical wind (m/s)

    ter:        float, 2d
        terrain height

    z:          float, 1d, bottom_top
        model layer height above terrain

    ztop:       float, scalar
        model top layer elevation above sea level

    dnw:        float, 1d, bottom_top
        delta eta values on vertical velocity levels 

    geo_z_idx:  int, scalar
        z index where geostraphic wind prevails (init height of free atm)

    near_surf_z_idx: int, scalar
        z index of init height of Ekman layer

    Methods
    '''
    
    def __init__(self, config):
        """ construct input wrf file names """
        
        print(print_prefix+'Init wrf_mesh obj...')
        print(print_prefix+'Read template file...')
        wrf_hdl=nc4.Dataset(config['INPUT']['input_root']+config['INPUT']['input_wrf'])
        # collect global attr
        self.dx=wrf_hdl.DX
        self.dy=wrf_hdl.DY
        
        # template UVW
        self.U = wrf.getvar(wrf_hdl, 'U')
        self.V = wrf.getvar(wrf_hdl, 'V')
        self.W = wrf.getvar(wrf_hdl, 'W')

        z3d=wrf.getvar(wrf_hdl,'z') # model layer elevation above sea level
        self.dnw=wrf.getvar(wrf_hdl,'DNW') # d_eta value on model layer
        self.ter=wrf.getvar(wrf_hdl,'ter') # terrain height  

        # model layer elevation above terrain
        self.z=z3d.mean(['south_north','west_east'])
        self.ztop=self.z[-1]
        self.z=self.z-self.ter.mean(['south_north','west_east'])
        # eta value on model layer
        
        # get index of z for near surface layer and free atm 
        temp_z=float(config['CORE']['geo_wind_lv'])
        self.geo_z_idx=utils.get_closest_idx(self.z.values,temp_z)
        temp_z=float(config['CORE']['near_surf_lv'])
        self.near_surf_z_idx=utils.get_closest_idx(self.z.values,temp_z)

        (self.n_sn, self.n_we)=self.ter.shape # on mass grid

        # lats lons on mass and staggered grids
        self.XLAT=wrf.getvar(wrf_hdl,'XLAT')
        self.XLONG=wrf.getvar(wrf_hdl,'XLONG')
        self.XLAT_U=wrf.getvar(wrf_hdl,'XLAT_U')
        self.XLONG_U=wrf.getvar(wrf_hdl,'XLONG_U')
        self.XLAT_V=wrf.getvar(wrf_hdl,'XLAT_V')
        self.XLONG_V=wrf.getvar(wrf_hdl,'XLONG_V')
        
        # DEPRECATED
        if config['CORE']['interp_mode'] == 'accurate':
            print(print_prefix+'Construct layer heights on staggered U/V mesh...')
            self.z_on_u=self.interp_u_stag(self.z)
            self.z_on_v=self.interp_v_stag(self.z)
        
        gc.collect()

    def interp_u_stag(self, var):
        """ DEPRECATED METHOD
        Linear/Nearest interpolate var from mass grid onto staggered U grid 
        """
        
        template=copy(self.U)

        x_org=self.XLAT.values.flatten()
        y_org=self.XLONG.values.flatten()
        
        for ii in range(0, self.nz):
            z_org=self.z.values[ii,:,:].flatten()
            interp=interpolate.NearestNDInterpolator(list(zip(x_org, y_org)), z_org)
            #interp=interpolate.LinearNDInterpolator(list(zip(x_org, y_org)), z_org)
            template.values[ii,:,:] = interp(self.XLAT_U.values, self.XLONG_U.values)
        return template 
 
    def interp_v_stag(self, var):
        """ DEPRECATED METHOD
        Linear/Nearest interpolate var from mass grid onto staggered U grid 
        """

        template=copy(self.V)

        x_org=self.XLAT.values.flatten()
        y_org=self.XLONG.values.flatten()
        
        for ii in range(0, self.nz):
            z_org=self.z.values[ii,:,:].flatten()
            interp=interpolate.NearestNDInterpolator(list(zip(x_org, y_org)), z_org)
            #interp=interpolate.LinearNDInterpolator(list(zip(x_org, y_org)), z_org)
            template.values[ii,:,:] = interp(self.XLAT_V.values, self.XLONG_V.values)
        return template 
       
        
if __name__ == "__main__":
    pass
