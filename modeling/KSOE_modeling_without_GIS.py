import pandas as pd
import scipy.stats as st
import numpy as np
import simpy
import time
import functools



start_running = time.time()

# input data
data_all = pd.read_excel('../data/Layout_Activity_4_12.xlsx')
print(data_all)
print(0)