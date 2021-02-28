#/usr/bin/env python
"""Model clock obj to control the interpolation frame."""

import datetime

print_prefix='lib.clock>>'
class model_clock:
    '''
    Model clock obj to control the interpolation frame.
    
    Attributes
    -----------

    Methods
    -----------
    advance(), advance the model clock to the next timeframe 

    '''

    def __init__(self, cfg):
        """construct time manager object"""
        self.init_time=datetime.datetime.strptime(cfg['CORE']['interp_strt_t'], '%Y%m%d%H%M')
        self.curr_time=self.init_time
        
        self.interp_len=int(cfg['CORE']['interp_t_length'])*60
        self.interp_interval=int(cfg['CORE']['interp_interval'])
        self.end_time=self.init_time+datetime.timedelta(minutes=self.interp_len)
        self.assim_win=int(cfg['CORE']['assim_win'])
        self.done=False
        print(print_prefix+"Model clock initiated at "+self.curr_time.strftime("%Y-%m-%d %H:%M:%S"))    


    def advance(self):
        """advance the model clock by time interval"""
        self.curr_time=self.curr_time+datetime.timedelta(minutes=self.interp_interval)
        
        if self.curr_time < self.end_time:
            print(print_prefix+"Model clock advanced to "+self.curr_time.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            print(print_prefix+"Model clock finished at "+self.end_time.strftime('%Y-%m-%d %H:%M:%S')) 
            self.done=True

    def collect_obv(self, obv_lst):
        """collect effective obvs within the assimilation window or extrapolating"""
        return obv_lst
