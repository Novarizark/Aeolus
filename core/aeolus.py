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

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

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
    
    def __init__(self, fields_hdl, cfg_hdl):
        """ construct aeolus interpolator """
        self.U=fields_hdl.U
        self.V=fields_hdl.V
        self.W=fields_hdl.W
        
        self.U10=fields_hdl.U10
        self.V10=fields_hdl.V10
        self.T2=fields_hdl.T2
       
        self.alpha_x=float(cfg_hdl['CORE']['alpha_x'])
        self.alpha_y=float(cfg_hdl['CORE']['alpha_y'])
        self.alpha_z=float(cfg_hdl['CORE']['alpha_z'])

    def cast(self, obv_lst, fields_hdl, clock):
        """ cast interpolation on WRF mesh """
        print(print_prefix+'Interpolate UV...')
        cast_lst=select_obv(obv_lst, clock)
        n_obv=len(cast_lst)
        
        # construct distance matrix on staggered u grid
        self.dis_mtx_u=np.zeros((fields_hdl.n_sn, fields_hdl.n_we+1, n_obv))
        
        # construct distance matrix on staggered v grid
        self.dis_mtx_v=np.zeros((fields_hdl.n_sn+1, fields_hdl.n_we, n_obv))
        
        # construct distance matrix on mass grid
        self.dis_mtx=np.zeros((fields_hdl.n_sn, fields_hdl.n_we, n_obv))

        for idx, obv in enumerate(cast_lst):
            # advection distance (km) adjustment according to delta t
            adv_dis=abs((obv.t-clock.curr_time).total_seconds())*utils.wind_speed(obv.u0, obv.v0)/1000.0
            self.dis_mtx_u[:,:,idx]=adv_dis+utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT_U, fields_hdl.XLONG_U)
            self.dis_mtx_v[:,:,idx]=adv_dis+utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT_V, fields_hdl.XLONG_V)
            self.dis_mtx[:,:,idx]=adv_dis+utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT, fields_hdl.XLONG)
        
        usort_idx=np.argsort(self.dis_mtx_u)
        vsort_idx=np.argsort(self.dis_mtx_v)
        msort_idx=np.argsort(self.dis_mtx)

        # sorted distance matrix and take the nearest 3 to construct the calculating matrix
        dis_mtx_u_near=np.take_along_axis(self.dis_mtx_u, usort_idx, axis=-1)[:,:,0:3]
        dis_mtx_v_near=np.take_along_axis(self.dis_mtx_v, vsort_idx, axis=-1)[:,:,0:3]
        dis_mtx_near=np.take_along_axis(self.dis_mtx, msort_idx, axis=-1)[:,:,0:3]

        # get uv profile (n_obv, nlvl)
        u_profs=np.asarray([obv.u_prof for obv in cast_lst])
        v_profs=np.asarray([obv.v_prof for obv in cast_lst])
        nz=u_profs.shape[1]

        # construct for calculation
        u_prof_mtx=np.broadcast_to(u_profs, (fields_hdl.n_sn, fields_hdl.n_we+1, n_obv, nz))
        v_prof_mtx=np.broadcast_to(v_profs, (fields_hdl.n_sn+1, fields_hdl.n_we, n_obv, nz))
        
        # cast vertical profile interpolation 
        for idz in range(0, nz):
            u_prof_near=np.take_along_axis(u_prof_mtx[:,:,:,idz], usort_idx, axis=-1)[:,:,0:3]
            self.U.values[idz,:,:]=inv_dis_wgt_2d(u_prof_near,dis_mtx_u_near)
              
        for idz in range(0, nz):
            v_prof_near=np.take_along_axis(v_prof_mtx[:,:,:,idz], vsort_idx, axis=-1)[:,:,0:3]
            self.V.values[idz,:,:]=inv_dis_wgt_2d(v_prof_near, dis_mtx_v_near)

        # cast 10-m uv/ 2-m temp interpolation
            self.U10.values=cast_2d([obv.u10 for obv in cast_lst], fields_hdl,dis_mtx_near, msort_idx)            
            self.V10.values=cast_2d([obv.v10 for obv in cast_lst], fields_hdl,dis_mtx_near, msort_idx)            
            self.T2.values=cast_2d([obv.t2 for obv in cast_lst], fields_hdl,dis_mtx_near, msort_idx)            
            
        
        print(print_prefix+'First-guess W...')
        self.W.values,self.zx,self.zy=diag_vert_vel(self, fields_hdl)
        
        print(print_prefix+"Adjust results by mass conservation...")
        self.adjust_mass(fields_hdl)
        
    def adjust_mass(self, fields_hdl):
        """ adjust result according to continuity equation (mass conservation) """
        # prepare variables
        dx, ter, ztop=fields_hdl.dx,  fields_hdl.ter, fields_hdl.ztop
        n_sn, n_we=fields_hdl.n_sn, fields_hdl.n_we
        n_z=fields_hdl.geo_z_idx+1 # only adjust mass within the boundary layer
        dnw=fields_hdl.dnw[0:n_z+1] # delt eta value on each layer
        zx, zy=self.zx, self.zy
        
        # first guass u, v, w
        U0, V0, W0=self.U.values, self.V.values, self.W.values
        alx, aly, alz=self.alpha_x, self.alpha_y, self.alpha_z
        
        #zx xstag to right 
        zx_xstag_r=np.zeros(U0.shape[1:])
        zx_xstag_r[:,0:n_we]=zx
        zx_xstag_r[:,n_we]=zx[:,n_we-1]
        
        zy_ystag_u=np.zeros(V0.shape[1:])
        zy_ystag_u[0:n_sn,:]=zy
        zy_ystag_u[n_sn,:]=zy[n_sn-1,:]


        zx_xstag_l=np.zeros(U0.shape[1:])
        zx_xstag_l[:,1:]=zx
        zx_xstag_l[:,0]=zx[:,0]
        
        zy_ystag_d=np.zeros(V0.shape[1:])
        zy_ystag_d[1:,:]=zy
        zy_ystag_d[0,:]=zy[0,:]
 

        # write the terms according to (4.16) in Magnusson.pdf
        ax2 = 1.0/(alx*alx)
        ay2 = 1.0/(aly*aly)
        az2 = 1.0/(alz*alz)

        Bx=ax2*(2+(zx_xstag_r[:,1:]-zx_xstag_r[:,0:n_we])/(2*dx)) # (n_sn, n_we)
        By=ay2*(2+(zy_ystag_u[1:,:]-zy_ystag_u[0:n_sn,:])/(2*dx)) # (n_sn, n_we)
        Bz=2*az2/(dnw[1:]*(dnw[0:n_z]+dnw[1:])) # (n_z)
        
        O1=-(ax2/(dx*dx)+ay2/(dx*dx)+az2/(dnw[1:]*dnw[0:n_z])) # (n_z)
        O2=-(ax2*zx*zx+ay2*zy*zy) #(n_sn,n_we)
        O1_mtx=np.broadcast_to(O1, (n_sn, n_we, n_z))
        O2_mtx=np.broadcast_to(O2, (n_z, n_sn, n_we))
        O=np.zeros((n_z,n_sn,n_we)) #(n_sn, n_we, n_z)
        for iz in range(0,n_z):
            O[iz,:,:]=O2_mtx[iz,:,:]+O1_mtx[:,:,iz]

        Ax=ax2*(2+(zx_xstag_l[:,1:]-zx_xstag_l[:,0:n_we])/(2*dx)) # (n_sn, n_we)
        Ay=ay2*(2+(zy_ystag_d[1:,:]-zy_ystag_d[0:n_sn,:])/(2*dx)) # (n_sn, n_we)
        Bz=2*az2/(dnw[0:n_z]*(dnw[0:n_z]+dnw[1:])) # (n_z)

        # Right terms. Magnusson.pdf (4.17)
        R=0
        # all necessary data collected! Let's do the magic!
        


def cast_2d(var_list, fields_hdl, dis_mtx_near, sort_idx):
    """ For cast 10-m uv/ 2-m temp """
    n_obv=len(var_list)
    var_array=np.asarray(var_list)
    var_mtx=np.broadcast_to(var_array, (fields_hdl.n_sn, fields_hdl.n_we, n_obv))
    var_near=np.take_along_axis(var_mtx, sort_idx, axis=-1)[:,:,0:3]
    return inv_dis_wgt_2d(var_near, dis_mtx_near)


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
    # eta coordinate terrain gradient term, see (4.12a, 4.12b) in Magnusson.pdf
    zx=(1.0/(ztop-ter))*((ter_xstag[:,1:]-ter_xstag[:,0:n_we])/dx)
    zy=(1.0/(ztop-ter))*((ter_ystag[1:,:]-ter_ystag[0:n_sn,:])/dx)

    
    # eta coordinate adjustment terms, see (4.9') in Magnusson.pdf
    for ii in range(0,n_z):
        adj_term1=zx*U[ii,:,0:n_we]
        adj_term2=zy*V[ii,0:n_sn,:]        
        W[ii+1,:,:]=(adj_term1+adj_term2-div[ii,:,:])*dnw[ii]+W[ii,:,:]

    return W, zx, zy

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
    
    ds['U10'].values[0,:,:]=aeolus.U10.values
    ds['V10'].values[0,:,:]=aeolus.V10.values
    ds['T2'].values[0,:,:]=aeolus.T2.values
    
    ds.to_netcdf(out_fn, mode='a')

if __name__ == "__main__":
    pass
