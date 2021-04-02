#!/home/metctm1/array/soft/anaconda3/bin/python
'''
Date: Feb 04, 2021
Aeolus is a wind diagnostic model to interpolate in-situ wind observations
onto WRF output template file.

This is the main script to drive the model

Revision:
Feb 19, 2021 --- MVP v0.01 completed
Feb 24, 2021 --- MVP v0.02 Unit test finished
Mar 05, 2021 --- MVP v0.03 New version 
    --- Adevection Distance Adjustement
    --- Wind 16 directions and degree directions
    --- Interpolation of Nearest 3 stations.
Mar 08, 2021 --- MVP v0.04 New version
    --- Observation Examiner
    --- U10 V10 T2 bug fix
Mar 30, 2021 --- MVP v0.90
    --- Mass adjustment done! Dramatic leep! 
Apr 02, 2021
    --- Takes in sounding data
    --- Vertical convection time scale adjustment

Zhenning LI
'''

import numpy as np
import pandas as pd
import os

import lib 
import core
from multiprocessing import Pool, sharedctypes

def main_run():
    
    print('*************************AEOLUS START*************************')
   
    # wall-clock ticks
    time_mgr=lib.time_manager.time_manager()

    print('Read Config...')
    cfg_basic_hdl=lib.cfgparser.read_cfg('./conf/config.basic.ini')
    cfg_hdl=lib.cfgparser.read_cfg('./conf/config.ini')
    cfg_hdl=lib.cfgparser.merge_cfg(cfg_basic_hdl, cfg_hdl)
    cfg_hdl['CORE']['ntasks']='1'

    print('Read Input Observations...')
    obv_df=pd.read_csv(cfg_hdl['INPUT']['input_root']+cfg_hdl['INPUT']['input_obv'],header=0,
            names=['yyyymmddhhMM','lat','lon','height','wind_speed','wind_dir','temp_2m','rh','pres','attr1','attr2'])
    # make sure the list is sorted by datetime and long enough
    obv_df=obv_df.sort_values(by='yyyymmddhhMM') 
    
    print('Input Quality Control...')
    lib.obv_constructor.obv_examiner(obv_df)
    
    print('Read Wind Profile Exponents...')
    wind_prof_df=pd.read_csv('./db/power_coef_wind.csv')
    time_mgr.toc('INPUT MODULE')

    print('Construct WRFOUT Handler...')
    fields_hdl=lib.preprocess_wrfinp.wrf_mesh(cfg_hdl)
    
    print('Construct Observation Satation Objs...')
    obv_lst=[] 
    for row in obv_df.itertuples():
        obv_lst.append(lib.obv_constructor.obv(row, wind_prof_df, cfg_hdl, fields_hdl))
    
    # get area mean pvalue 
    fields_hdl.get_area_pvalue([obv.prof_pvalue for obv in obv_lst])
    # setup Ekman layer and geostrophic wind in obv wind profile
    lib.obv_constructor.set_upper_wind(fields_hdl, obv_lst)

    print('Construct Model Clocks and Interpolating Estimators.....')
    clock_cfg=lib.model_clock.clock_cfg_parser(cfg_hdl)
    ntasks=clock_cfg['nclock']
    clock_lst=[]
    estimator_lst=[]
    
    for i in range(0, ntasks):
        clock_lst.append(lib.model_clock.model_clock(clock_cfg, i))
        estimator_lst.append(core.aeolus.aeolus(fields_hdl, cfg_hdl))

    time_mgr.toc('CONSTRUCTION MODULE')

    if ntasks == 1:
        print('Aeolus Interpolating Estimator Casting...')
        clock=clock_lst[0]
        estimator=estimator_lst[0]
        while not(clock.done):
           
            estimator.cast(obv_lst, fields_hdl, clock)
            time_mgr.toc('CAST MODULE')

            print('Output Diagnostic UVW Fields...')
            core.aeolus.output_fields(cfg_hdl, estimator, clock)
            time_mgr.toc('OUTPUT MODULE')
            
            clock.advance()
    else:
        print('Multiprocessing initiated. Master process %s.' % os.getpid())
        # let's do the multiprocessing magic!
        # start process pool
        results=[]
        process_pool = Pool(processes=ntasks)
        for itsk in range(ntasks):  
            results.append(process_pool.apply_async(run_mtsk,args=(itsk, obv_lst, clock_lst[itsk], 
                estimator_lst[itsk],fields_hdl,cfg_hdl,)))

        process_pool.close()
        process_pool.join()
        time_mgr.toc('MULTI CAST MODULE')

    time_mgr.dump()

    print('*********************AEOLUS ACCOMPLISHED*********************')



def run_mtsk(itsk, obv_lst, clock, estimator, fields_hdl, cfg_hdl):
    """
    Aeolus cast function for multiple processors
    """
    print('TASK[%02d]: Aeolus Interpolating Estimator Casting...' % itsk)
    while not(clock.done):
           
        estimator.cast(obv_lst, fields_hdl, clock)

        print('TASK[%02d]: Output Diagnostic UVW Fields...' % itsk )
        core.aeolus.output_fields(cfg_hdl, estimator, clock)
        
        clock.advance()
    print('TASK[%02d]: Aeolus Subprocessing Finished!' % itsk)
    
    return 0

if __name__=='__main__':
    main_run()
