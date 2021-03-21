#/usr/bin/env python
"""Model clock obj to control the interpolation frame."""

import datetime
import copy

print_prefix='lib.clock>>'

def clock_cfg_parser(cfg):
    
    ntasks=int(cfg['CORE']['ntasks'])

    # special config dict for clocks
    clck_cfg={}

    clck_cfg['init_time']=datetime.datetime.strptime(cfg['CORE']['interp_strt_t'], '%Y%m%d%H%M')
    clck_cfg['interp_len']=int(float(cfg['CORE']['interp_t_length'])*60) # turn to minutes
    clck_cfg['effect_win']=int(cfg['CORE']['effect_win'])   
    clck_cfg['curr_time_lst']=[]
    clck_cfg['end_time_lst']=[]

    interval=int(cfg['CORE']['interp_interval'])
    nframes=clck_cfg['interp_len']//interval+1
    dt=datetime.timedelta(minutes=interval)
    clck_cfg['dt']=dt

    # deal with multiprocessing
    if ntasks > nframes:
        ntasks=nframes

    len_per_task=nframes//ntasks
    for itask in range(0,ntasks):
       clck_cfg['curr_time_lst'].append(clck_cfg['init_time'] + len_per_task*(itask)*dt)
       clck_cfg['end_time_lst'].append(clck_cfg['init_time'] + len_per_task*(itask+1)*dt-dt)
    clck_cfg['end_time_lst'][-1]=clck_cfg['init_time']+datetime.timedelta(minutes=clck_cfg['interp_len'])
    
    clck_cfg['nclock']=len(clck_cfg['curr_time_lst'])
    
    return clck_cfg

class model_clock:
    '''
    Model clock obj to control the interpolation frame.
    
    Attributes
    -----------

    Methods
    -----------
    advance(), advance the model clock to the next timeframe 

    '''

    def __init__(self, cfg, idx=0):
        """construct time manager object"""
        self.idx=idx
        self.init_time=cfg['init_time']
        self.curr_time=copy.copy(cfg['curr_time_lst'][idx])
        self.end_time=cfg['end_time_lst'][idx]
        
        self.dt=cfg['dt']
        self.effect_win=cfg['effect_win'] 
        self.done=False
        print(print_prefix+'Model clock '+str(idx)+' initiated at '+self.curr_time.strftime("%Y-%m-%d %H:%M:%S"))    


    def advance(self):
        """advance the model clock by time interval"""
        self.curr_time=self.curr_time+self.dt
        
        if self.curr_time <= self.end_time:
            print(print_prefix+"Model clock "+str(self.idx)+" advanced to "+self.curr_time.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            print(print_prefix+"Model clock "+str(self.idx)+" finished!")
            self.done=True
