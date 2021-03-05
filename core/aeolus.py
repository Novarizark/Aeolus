#/usr/bin/env python
"""
Core Component: Aeolus Interpolator
    Classes: 
    -----------
        aeolus: core class, domain interpolator

    Functions:
    -----------
        select_obv(obv_lster,clock): Select observation list for this timeframe
        diag_vert_vel(aeolus, fields_hdl): Diagnose vertical velocity according to divergence in eta coordinate
        inv_dis_wgt_2d(ws_lst, dis_mtx): Inverse Distance Weighting (IDW) Interpolation
        output_fields(cfg, aeolus): Output diagnostic fields into WRF template file

"""

import xarray as xr
import numpy as np
import copy

import sys, os
sys.path.append('../')
from utils import utils

print_prefix='core.aeolus>>'

class aeolus:

    '''
    Aeolus interpolator, interpolate in-situ obvs onto wrf mesh 
    
    Attributes
    -----------
    dis_mtx_u(n_sn, n_we_stag, n_obv), float, distance between obv and grid point, matrix on staggered u grid
    dis_mtx_v(n_sn_stag, n_we, n_obv), float, distance between obv and grid point, matrix on staggered v grid
    U(bottom_top,n_sn, n_we_stag), float, interpolated U
    V(bottom_top,n_sn_stag, n_we), float, interpolated V
    W(bottom_top,n_sn_stag, n_we), float, diagnostic W

    Methods
    -----------
    cast(), cast interpolation on WRF mesh
    adjust_mass(), adjust result according to continuity equation (mass conservation)
    '''
    
    def __init__(self, fields_hdl):
        """ construct aeolus interpolator """
        self.U=fields_hdl.U
        self.V=fields_hdl.V
        self.W=fields_hdl.W
   
    def cast(self, obv_lst, fields_hdl, clock):
        """ cast interpolation on WRF mesh """
        print(print_prefix+'Interpolate UV...')
        cast_lst=select_obv(obv_lst, clock)
        n_obv=len(cast_lst)
        
        # construct distance matrix on staggered u grid
        self.dis_mtx_u=np.zeros((fields_hdl.n_sn, fields_hdl.n_we+1, n_obv))
        
        # construct distance matrix on staggered v grid
        self.dis_mtx_v=np.zeros((fields_hdl.n_sn+1, fields_hdl.n_we, n_obv))
        
        for idx, obv in enumerate(cast_lst):
            # advection distance (km) adjustment according to delta t
            adv_dis=abs((obv.t-clock.curr_time).total_seconds())*utils.wind_speed(obv.u0, obv.v0)/1000.0
            self.dis_mtx_u[:,:,idx]=adv_dis+utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT_U, fields_hdl.XLONG_U)
            self.dis_mtx_v[:,:,idx]=adv_dis+utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT_V, fields_hdl.XLONG_V)
        
        usort_idx=np.argsort(self.dis_mtx_u)
        vsort_idx=np.argsort(self.dis_mtx_v)

        # sorted distance matrix and take the nearest 3 to construct the calculating matrix
        dis_mtx_u_near=np.take_along_axis(self.dis_mtx_u, usort_idx, axis=-1)[:,:,0:3]
        dis_mtx_v_near=np.take_along_axis(self.dis_mtx_v, vsort_idx, axis=-1)[:,:,0:3]
       
        # get uv profile (n_obv, nlvl)
        u_profs=np.asarray([obv.u_prof for obv in cast_lst])
        v_profs=np.asarray([obv.v_prof for obv in cast_lst])
        nz=u_profs.shape[1]

        # construct for calculation
        u_prof_mtx=np.broadcast_to(u_profs, (fields_hdl.n_sn, fields_hdl.n_we+1, n_obv, nz))
        v_prof_mtx=np.broadcast_to(v_profs, (fields_hdl.n_sn+1, fields_hdl.n_we, n_obv, nz))
        
        
        for idz in range(0, nz):
            u_prof_near=np.take_along_axis(u_prof_mtx[:,:,:,idz], usort_idx, axis=-1)[:,:,0:3]
            self.U.values[idz,:,:]=inv_dis_wgt_2d(u_prof_near,dis_mtx_u_near)
              
        for idz in range(0, nz):
            v_prof_near=np.take_along_axis(v_prof_mtx[:,:,:,idz], vsort_idx, axis=-1)[:,:,0:3]
            self.V.values[idz,:,:]=inv_dis_wgt_2d(v_prof_near, dis_mtx_v_near)

        print(print_prefix+'First-guess W...')
        self.W.values=diag_vert_vel(self, fields_hdl)
        
        print(print_prefix+"Adjust results by mass conservation...")
        self.adjust_mass(fields_hdl)
        
    def adjust_mass(self, fields_hdl):
        """ adjust result according to continuity equation (mass conservation) """
        pass

def diag_vert_vel(aeolus, fields_hdl):
    """ Diagnose vertical velocity according to divergence in eta coordinate"""
    dx, dnw, ter, ztop=fields_hdl.dx, fields_hdl.dnw, fields_hdl.ter, fields_hdl.ztop
    n_sn, n_we=fields_hdl.n_sn, fields_hdl.n_we
    n_z=dnw.shape[0]
    U=aeolus.U.values
    V=aeolus.V.values
    W=aeolus.W.values
    # bottom boundary condition for vertical velocity
    W[0,:,:]=0.0

    ter_xstag=np.zeros(U.shape[1:])
    ter_xstag[:,0:n_we]=ter
    ter_xstag[:,n_we]=ter[:,n_we-1]
    ter_ystag=np.zeros(V.shape[1:])
    ter_ystag[0:n_sn,:]=ter
    ter_ystag[n_sn,:]=ter[n_sn-1,:]
    
    div=utils.div_2d(U, V, dx, dx)

    # eta coordinate adjustment terms, see (4.9') in Magnusson.pdf
    for ii in range(0,n_z):
        adj_term1=(1.0/(ztop-ter))*U[ii,:,0:n_we]*((ter_xstag[:,1:]-ter_xstag[:,0:n_we])/dx)
        adj_term2=(1.0/(ztop-ter))*V[ii,0:n_sn,:]*((ter_ystag[1:,:]-ter_ystag[0:n_sn,:])/dx)
        W[ii+1,:,:]=(adj_term1+adj_term2-div[ii,:,:])*dnw[ii]+W[ii,:,:]

    return W

def select_obv(obv_lster,clock_obj):
    """ Select observation list for this timeframe """
    valid_lster=[]
    clock=clock_obj.curr_time
    effect_win=clock_obj.effect_win*60 # shift to seconds
    # get the absolute delta time between each obv and current clock
    dt_lst=[abs((obv.t-clock).total_seconds()) for obv in obv_lster]
    # sort the dt_lst and return the index of sorted list from min to max
    dt_rank=sorted(range(len(dt_lst)), key=lambda k: dt_lst[k])
    
    for idx in dt_rank:
        if dt_lst[idx]<effect_win:
            valid_lster.append(obv_lster[idx])
    len_valid=len(valid_lster)
    # make sure at least 3 stations will be selected
    if len_valid < 3:
        for i in dt_rank[len_valid:3]:
            valid_lster.append(obv_lster[i])
    return valid_lster 


def inv_dis_wgt_2d(ws_lst, dis_mtx):
    """ Inverse Distance Weighting (IDW) Interpolation """
    
    (n_sn,n_we)=dis_mtx[:,:,0].shape
    ws_2d=np.zeros((n_sn,n_we))
    
    wgt_mtx=np.power(dis_mtx, -2)
    ws_2d=np.sum(wgt_mtx*ws_lst, axis=2)/np.sum(wgt_mtx, axis=2)
    
    return ws_2d



def output_fields(cfg, aeolus, clock):
    """ Output diagnostic fields into WRF template file """
    print(print_prefix+'Output...')
    
    init_timestamp=clock.init_time.strftime('%Y-%m-%d_%H:%M:%S')
    curr_time=clock.curr_time

    time_stamp=curr_time.strftime('%Y-%m-%d_%H:%M:%S')
    template_fn=cfg['INPUT']['input_root']+cfg['INPUT']['input_wrf']
    out_fn=cfg['OUTPUT']['output_root']+cfg['INPUT']['input_wrf']+'_'+time_stamp
    
    os.system('cp '+template_fn+' '+out_fn)
    ds=xr.open_dataset(out_fn)
    ds.attrs['START_DATE']=init_timestamp
    ds.attrs['SIMULATION_START_DATE']=init_timestamp
    ds['Times'][0:19]=time_stamp[0:19]
    ds['U'].values[0,:,:,:]=aeolus.U.values
    ds['V'].values[0,:,:,:]=aeolus.V.values
    ds['W'].values[0,:,:,:]=aeolus.W.values
    ds.to_netcdf(out_fn, mode='a')

if __name__ == "__main__":
    pass
