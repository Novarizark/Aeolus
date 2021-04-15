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
from scipy.sparse.linalg import spsolve, eigsh
from scipy import interpolate

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
        self.T=fields_hdl.T
        
        self.U10=fields_hdl.U10
        self.V10=fields_hdl.V10
        self.T2=fields_hdl.T2
       
        self.alpha_x=float(cfg_hdl['CORE']['alpha_x'])
        self.alpha_y=float(cfg_hdl['CORE']['alpha_y'])
        self.alpha_z=float(cfg_hdl['CORE']['alpha_z'])

    def cast(self, obv_lst, fields_hdl, clock):
        """ cast interpolation on WRF mesh """
        # e-folding rate in cross-layer interpolation 
        efold_r=fields_hdl.efold_r
        # convective propagation distance
        conv_t=fields_hdl.conv_t
        
        print(print_prefix+'Interpolate UV...')
        cast_lst=select_obv(obv_lst, clock)
        n_obv=len(cast_lst)
       
        n_sn, n_we = fields_hdl.n_sn, fields_hdl.n_we
        nz = fields_hdl.z.shape[0]
        # construct distance matrix on staggered u grid
        self.dis_mtx_u=np.zeros((n_sn, n_we+1, n_obv))
        
        # construct distance matrix on staggered v grid
        self.dis_mtx_v=np.zeros((n_sn+1, n_we, n_obv))
        
        # construct distance matrix on mass grid
        self.dis_mtx_t=np.zeros((n_sn, n_we, n_obv))

        # get z level position (n_obv)
        zpos=np.asarray([obv.iz for obv in cast_lst])
        
        # calculate distance matrix
        for idx, obv in enumerate(cast_lst):
            # advection distance (km) adjustment according to delta t
            adv_dis=abs((obv.t-clock.curr_time).total_seconds())*utils.wind_speed(obv.u0, obv.v0)/1000.0
            self.dis_mtx_u[:,:,idx]=adv_dis+utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT_U, fields_hdl.XLONG_U)
            self.dis_mtx_v[:,:,idx]=adv_dis+utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT_V, fields_hdl.XLONG_V)
            self.dis_mtx_t[:,:,idx]=adv_dis+utils.great_cir_dis_2d(obv.lat, obv.lon, fields_hdl.XLAT, fields_hdl.XLONG)
        
        # get uvt profile (n_obv, nlvl)
        u_profs, v_profs, t_profs=np.asarray([obv.u_prof for obv in cast_lst]), np.asarray([obv.v_prof for obv in cast_lst]), np.asarray([obv.t_prof for obv in cast_lst])


        # construct for calculation
        u_prof_mtx=np.broadcast_to(u_profs, (n_sn, n_we+1, n_obv, nz))
        v_prof_mtx=np.broadcast_to(v_profs, (n_sn+1, n_we, n_obv, nz))
        t_prof_mtx=np.broadcast_to(t_profs, (n_sn, n_we, n_obv, nz))

        # cast vertical profile interpolation 
        for idz in range(0, nz):
            # get dis to idz
            zdis=abs(zpos-idz)
            zdis_umtx=np.broadcast_to(zdis, (n_sn, n_we+1, n_obv))
            zdis_vmtx=np.broadcast_to(zdis, (n_sn+1, n_we, n_obv))
            zdis_tmtx=np.broadcast_to(zdis, (n_sn, n_we, n_obv))

            # penalize according to vertical distance
            dis_mtx_u=(self.dis_mtx_u+conv_t*zdis_umtx)*np.exp(zdis_umtx*efold_r)
            dis_mtx_v=(self.dis_mtx_v+conv_t*zdis_vmtx)*np.exp(zdis_vmtx*efold_r)
            dis_mtx_t=(self.dis_mtx_t+conv_t*zdis_tmtx)*np.exp(zdis_tmtx*efold_r)
            
            #print(dis_mtx[50,50,:])
            # sort_idx (n_sn, n_we, n_obv)
            usort_idx, vsort_idx, tsort_idx=np.argsort(dis_mtx_u), np.argsort(dis_mtx_v), np.argsort(dis_mtx_t)

            # sorted distance matrix and take the nearest 3 to construct the calculating matrix
            dis_mtx_u_near=np.take_along_axis(dis_mtx_u, usort_idx, axis=-1)[:,:,0:3]
            dis_mtx_v_near=np.take_along_axis(dis_mtx_v, vsort_idx, axis=-1)[:,:,0:3]
            dis_mtx_t_near=np.take_along_axis(dis_mtx_t, tsort_idx, axis=-1)[:,:,0:3]

            u_prof_near=np.take_along_axis(u_prof_mtx[:,:,:,idz], usort_idx, axis=-1)[:,:,0:3]
            self.U.values[idz,:,:]=inv_dis_wgt_2d(u_prof_near,dis_mtx_u_near)
              
            v_prof_near=np.take_along_axis(v_prof_mtx[:,:,:,idz], vsort_idx, axis=-1)[:,:,0:3]
            self.V.values[idz,:,:]=inv_dis_wgt_2d(v_prof_near, dis_mtx_v_near)
            
            t_prof_near=np.take_along_axis(t_prof_mtx[:,:,:,idz], tsort_idx, axis=-1)[:,:,0:3]
            self.T.values[idz,:,:]=inv_dis_wgt_2d(t_prof_near, dis_mtx_t_near)

        print(print_prefix+'First-guess W...')
        self.W.values,self.zx,self.zy=diag_vert_vel(self, fields_hdl)
        
        print(print_prefix+"Adjust results by mass conservation...")
        solve_nz=fields_hdl.solve_nz # try 3 layer scale at first
        while solve_nz > 0:
            solve_nz=self.adjust_mass(fields_hdl, solve_nz)

        if solve_nz <0:
            print('simple adjust...')
            self.simple_terrain_adjust(fields_hdl)
        
        # cast 10-m uv 
        self.U10.values[:,0:n_we]= utils.wind_prof_2d(self.U.values[0,:,0:n_we], fields_hdl.z[0], 10.0, fields_hdl.pval)
        self.V10.values[0:n_sn,:]= utils.wind_prof_2d(self.V.values[0,0:n_sn,:], fields_hdl.z[0], 10.0, fields_hdl.pval)
        
        # cast 2-m temp interpolation
        self.T2.values=self.T.values[0,:,:]+300.0
        
        
        
    def simple_terrain_adjust(self, fields_hdl):
        """adjust result according to simple terrain gradient"""
        dx, ter, pbl_top=fields_hdl.dx,  fields_hdl.ter, fields_hdl.pbl_top
        n_sn, n_we=fields_hdl.n_sn, fields_hdl.n_we
        n_z=fields_hdl.geo_z_idx+1 # only adjust mass within the boundary layer
        zx, zy=self.zx, self.zy
        zx, zy=zx*(pbl_top-ter), zy*(pbl_top-ter)
        U0, V0=self.U.values[0:n_z,:,:], self.V.values[0:n_z,:,:]
        
        sigma=[1.0, 1.0] 
        # get slope in radius
        x_slop=np.arctan(2*zx)
        y_slop=np.arctan(2*zy)
        for iz in range(0,n_z):
            U0[iz,:,0:n_we]=U0[iz,:,0:n_we]/np.cos(x_slop)
            V0[iz,0:n_sn,:]=V0[iz,0:n_sn,:]/np.cos(y_slop)
           # U0[iz,:,:] = sp.ndimage.filters.gaussian_filter(U0[iz,:,:], sigma, mode='constant')
           # V0[iz,:,:] = sp.ndimage.filters.gaussian_filter(V0[iz,:,:], sigma, mode='constant')


    def adjust_mass(self, fields_hdl, n_z):
        """ adjust result according to continuity equation (mass conservation) """
        n_sn, n_we=fields_hdl.n_sn, fields_hdl.n_we

        # only adjust mass within 6 layers by solving Ax=b
        if n_z>6:
            return -1        
        # scaling factor to adjust deltaU and deltaV
        scal_f = 2.5 

        dx, ter=fields_hdl.dx,  fields_hdl.ter[0:n_sn,0:n_we].values
        z3d=fields_hdl.abz3d[0:n_z+2,0:n_sn,0:n_we].values
        ztop=fields_hdl.abz3d[n_z+2,0:n_sn,0:n_we].values
        ter3d=np.broadcast_to(ter, (n_z+2, n_sn, n_we))
        ztop3d=np.broadcast_to(ztop, (n_z+2, n_sn, n_we))
        eta=ztop3d*(z3d-ter3d)/(ztop3d-ter3d)
        dnw=eta[1:,:,:]-eta[0:n_z+1,:,:]
        ter_xstag=utils.pad_var2d(ter, 'tail', dim=1)
        ter_ystag=utils.pad_var2d(ter, 'tail', dim=0)
        # eta coordinate terrain gradient term (within PBL), see (4.12a, 4.12b) in Magnusson.pdf
        zx=(1.0/(ztop-ter))*((ter_xstag[:,1:]-ter)/dx)
        zy=(1.0/(ztop-ter))*((ter_ystag[1:,:]-ter)/dx)

        U0, V0, W0=self.U.values[0:n_z,0:n_sn,0:n_we+1], self.V.values[0:n_z,0:n_sn+1,0:n_we], self.W.values[0:n_z,0:n_sn,0:n_we]
        alx, aly, alz=self.alpha_x, self.alpha_y, self.alpha_z
       
        #zx xstag to east/west and north/south 
        zx_xstag_r=utils.pad_var2d(zx,'tail',dim=1)
        zx_xstag_l=utils.pad_var2d(zx,'head',dim=1)
        zy_ystag_u=utils.pad_var2d(zy,'tail',dim=0)
        zy_ystag_d=utils.pad_var2d(zy,'head',dim=0)
    
        # leftward (westward) pad U0
        U0_l=utils.pad_var3d(U0,'head',dim=2)
        
        # downward (southward) pad V0
        V0_d=utils.pad_var3d(V0,'head', dim=1)
                          
        # upward and downward pad W0
        W0_ex=np.zeros((n_z+2,n_sn,n_we))
        W0_ex[1:n_z+1,:,:]=W0
        W0_ex[0,:,:]=W0[0,:,:]
        W0_ex[-1,:,:]=W0[-1,:,:]

        # write the terms according to (4.16) in Magnusson.pdf
        ax2, ay2, az2 = 1.0/(alx*alx), 1.0/(aly*aly), 1.0/(alz*alz)

        Bx=ax2*(2+dx*(zx_xstag_r[:,1:]-zx))/(2*dx*dx) # (n_sn, n_we)
        By=ay2*(2+dx*(zy_ystag_u[1:,:]-zy))/(2*dx*dx) # (n_sn, n_we)
        Bz=2*az2/(dnw[1:,:,:]*(dnw[0:n_z,:,:]+dnw[1:,:,:])) # (n_z, n_sn, n_we)
        
        O1=-(ax2/(dx*dx)+ay2/(dx*dx)+az2/(dnw[1:,:,:]*dnw[0:n_z,:,:])) # (n_z, n_sn, n_we)
        O2=-(ax2*zx*zx+ay2*zy*zy) #(n_sn,n_we)
        O2_mtx=np.broadcast_to(O2, (n_z, n_sn, n_we))
        O=O1+O2

        Ax=ax2*(2+dx*(zx_xstag_l[:,1:]-zx))/(2*dx*dx) # (n_sn, n_we)
        Ay=ay2*(2+dx*(zy_ystag_d[1:,:]-zy))/(2*dx*dx) # (n_sn, n_we)
        Az=2*az2/(dnw[0:n_z,:,:]*(dnw[0:n_z,:,:]+dnw[1:,:,:])) # (n_z, n_sn, n_we)

        # Right terms. Magnusson.pdf (4.17)
        R1=(U0[:,:,1:]-U0_l[:,:,:n_we])/(2*dx) #(n_z, n_sn, n_we)
        R2=(V0[:,1:,:]-V0_d[:,:n_sn,:])/(2*dx) #(n_z, n_sn, n_we)
        
        R3_c1=np.power(dnw[0:n_z,:,:],2)*W0_ex[2:n_z+2,:,:]
        R3_c2=(np.power(dnw[0:n_z,:,:],2)-np.power(dnw[1:,:,:],2))*W0_ex[1:n_z+1,:,:]
        R3_c3=np.power(dnw[1:,:,:],2)*W0_ex[:n_z,:,:]        
        
        R3_de=dnw[1:,:,:]*dnw[0:n_z,:,:]*(dnw[0:n_z,:,:]+dnw[1:,:,:])
        R3=(R3_c1+R3_c2+R3_c3)/R3_de        
        R=np.zeros((n_z,n_sn,n_we))
        
        
        zx_3d, zy_3d=np.broadcast_to(zx, (n_z, n_sn, n_we)), np.broadcast_to(zy, (n_z, n_sn, n_we))
        
        R=zx_3d*U0[:,:,:n_we]+zy_3d*V0[:,:n_sn,:]
        R=-(R1+R2+R3-R)
        # all necessary data collected! Let's do the magic!
        
        # set boundary condition for b vector
        R[0,:,:], R[-1,:,:] = 0, 0
        R[:,0,:], R[:,-1,:] = 0, 0
        R[:,:,0], R[:,:,-1] = 0, 0

        # set b vector
        bsize=n_z*n_sn*n_we
        b=np.reshape(R,(bsize), order='C') # last axis change fastest
        # construct A
        A=lil_matrix((bsize,bsize))

        # set diag for A
        A.setdiag(1)
        
        # set lower boundary for A
        for iy in range(1, n_sn-1):
            for ix in range(1, n_we-1):
                ib=ix+iy*n_we
                ib2=ib+1
                A[ib,ib2]=-1
        
        # fill A 
        for iz in range(1,n_z-1):
            for iy in range(1, n_sn-1):
                for ix in range(1, n_we-1):
                    ib=ix+iy*n_we+iz*n_sn*n_we
                    ib2=ib
                    ib2_ip1=(ix+1)+iy*n_we+iz*n_sn*n_we
                    ib2_jp1=ix+(iy+1)*n_we+iz*n_sn*n_we
                    ib2_kp1=ix+iy*n_we+(iz+1)*n_sn*n_we
                    ib2_is1=(ix-1)+iy*n_we+iz*n_sn*n_we
                    ib2_js1=ix+(iy-1)*n_we+iz*n_sn*n_we
                    ib2_ks1=ix+iy*n_we+(iz-1)*n_sn*n_we
                    A[ib,ib2],A[ib,ib2_ip1],A[ib,ib2_jp1],A[ib,ib2_kp1]=O[iz, iy, ix],Bx[iy, ix],By[iy, ix],Bz[iz, iy, ix] 
                    A[ib,ib2_is1],A[ib,ib2_js1],A[ib,ib2_ks1]=Ax[iy, ix],Ay[iy, ix],Az[iz, iy, ix]
        
        # print filling ratio
        fill_r=100*A.getnnz()/(bsize*bsize)
        print(print_prefix+'A matrix filling ratio:%7.6f%%' % fill_r)
        A=A.tocsr()

        # let's solve 
        l=spsolve(A,b)
        l3d=np.reshape(l,(n_z,n_sn,n_we), order='C')
        
        # eastward pad lambda 
        l3d_e=utils.pad_var3d(l3d, 'tail', dim=2)

        # northward pad lambda
        l3d_n=utils.pad_var3d(l3d, 'tail', dim=1)

        # upward pad lambda
        l3d_u=np.zeros((n_z+1,n_sn,n_we))
        l3d_u[:n_z,:,:]=l3d
        l3d_u[-1,:,:]=l3d[-1,:,:]

        # yield du dv dw
        du=ax2*(((l3d_e[:,:,1:]-l3d)/dx)+zx_3d*l3d)
        dv=ay2*(((l3d_n[:,1:,:]-l3d)/dx)+zy_3d*l3d)
        dw=az2*(((l3d_u[1:,:,:]-l3d)/dnw[0:n_z,:,:]))
        du[[0,-1],:,:],dv[[0,-1],:,:],dw[[0,-1],:,:]=du[[1,-2],:,:],dv[[1,-2],:,:],dw[[1,-2],:,:] # set lower boundary
        
        scal_u0,scal_v0,scal_w0=np.max(abs(U0),axis=(1,2)),np.max(abs(V0),axis=(1,2)),np.max(abs(W0),axis=(1,2))
        scal_du,scal_dv,scal_dw=np.max(abs(du),axis=(1,2)),np.max(abs(dv),axis=(1,2)),np.max(abs(dw),axis=(1,2))
        
        print(print_prefix+'deltaU vector:')
        print(scal_du)
        
        if scal_du.mean()>5.0: # unreal solution
            print(print_prefix+'Solving linear system with '+str(n_z)+' vertical layers failed, try another time.')
            return n_z+1
            
        print(print_prefix+'Solving linear system with '+str(n_z)+' vertical layers successful!')
        U0[:,:,1:],V0[:,1:,:]=U0[:,:,1:]+scal_f*du,V0[:,1:,:]+scal_f*dv
        U0[:,:,0],V0[:,0,:]=U0[:,:,1],V0[:,0,:]
        W0=W0+dw
        self.interp_residual(n_z, fields_hdl, du[-1,:,:], dv[-1,:,:])
        
        return 0

    def interp_residual(self, n_z, fields_hdl, du0, dv0):
        """ Interpolate residual layers """
        U0, V0=self.U.values, self.V.values
        interp_topz=fields_hdl.near_surf_z_idx+3

        layer_h=fields_hdl.z[interp_topz]-fields_hdl.z[n_z-1]
        strt_h=fields_hdl.z[n_z-1]
        
        for iz in range(n_z, interp_topz):
            dh=fields_hdl.z[iz]-strt_h
            U0[iz,:,1:]=U0[iz,:,1:]+du0*(1-dh/layer_h).values
            U0[iz,:,0]=U0[iz,:,0]
            V0[iz,1:,:]=V0[iz,1:,:]+dv0*(1-dh/layer_h).values
            V0[iz,0,:]=V0[iz,0,:]


def cast_2d(var_list, fields_hdl, dis_mtx_near, sort_idx):
    """ For cast 2-m temp """
    n_obv=len(var_list)
    var_array=np.asarray(var_list)
    var_mtx=np.broadcast_to(var_array, (fields_hdl.n_sn, fields_hdl.n_we, n_obv))
    var_near=np.take_along_axis(var_mtx, sort_idx, axis=-1)[:,:,0:3]
    return inv_dis_wgt_2d(var_near, dis_mtx_near)


def diag_vert_vel(aeolus, fields_hdl):
    """ Diagnose vertical velocity according to divergence in eta coordinate thru all layers"""
    dx, dnw, ter, pbl_top=fields_hdl.dx, fields_hdl.dnw, fields_hdl.ter, fields_hdl.pbl_top
    n_sn, n_we=fields_hdl.n_sn, fields_hdl.n_we
    n_z=fields_hdl.geo_z_idx+1 # only diagnose vertical velocity within PBL 
   # n_z=dnw.shape[0]
    U=aeolus.U.values
    V=aeolus.V.values
    W=aeolus.W.values
    # bottom boundary condition for vertical velocity
    W[0,:,:]=0.0

    ter_xstag=utils.pad_var2d(ter, 'tail', dim=1)
    ter_ystag=utils.pad_var2d(ter, 'tail', dim=0)

    
    div=utils.div_2d(U, V, dx, dx)

    # eta coordinate terrain gradient term, see (4.12a, 4.12b) in Magnusson.pdf
    zx=(1.0/(pbl_top-ter))*((ter_xstag[:,1:]-ter)/dx)
    zy=(1.0/(pbl_top-ter))*((ter_ystag[1:,:]-ter)/dx)

    
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
    template_fn='./db/'+cfg['INPUT']['input_wrf']
    out_fn=cfg['OUTPUT']['output_root']+cfg['INPUT']['input_wrf']+'_'+time_stamp
    
    os.system('cp '+template_fn+' '+out_fn)
    ds=xr.open_dataset(template_fn)
    ds.attrs['START_DATE']=init_timestamp
    ds.attrs['SIMULATION_START_DATE']=init_timestamp
    ds['Times'][0:19]=time_stamp[0:19]
    ds['U'].values[0,:,:,:]=aeolus.U.values
    ds['V'].values[0,:,:,:]=aeolus.V.values
    ds['W'].values[0,:,:,:]=aeolus.W.values
    ds['T'].values[0,:,:,:]=aeolus.T.values
    
    ds['U10'].values[0,:,:]=aeolus.U10.values
    ds['V10'].values[0,:,:]=aeolus.V10.values
    ds['T2'].values[0,:,:]=aeolus.T2.values
    
    ds.to_netcdf(out_fn, mode='a')
    ds.close()
if __name__ == "__main__":
    pass
