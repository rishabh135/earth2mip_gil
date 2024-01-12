import numpy as np

from earth2mip.inference_medium_range import score_deterministic


import datetime
from earth2mip.networks import get_model
from earth2mip.initial_conditions import cds
from earth2mip.inference_ensemble import run_basic_inference




time_loop  = get_model("e2mip://fcnv2_sm", device="cuda:0")


scores = score_deterministic(time_loop,
    data_source=data_source,
    n=10,
    initial_times=[datetime.datetime(2018, 1, 1)],
    # fill in zeros for time-mean, will typically be grabbed from data.
    time_mean=np.zeros((7, 721, 1440))
    
    
    
data_source = cds.DataSource(time_loop.in_channel_names)
ds = run_basic_inference(time_loop, n=10, data_source=data_source, time=datetime.datetime(2018, 1, 1))
ds.chunk()