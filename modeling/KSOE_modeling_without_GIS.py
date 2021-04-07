import pandas as pd
import scipy.stats as st
import numpy as np
import simpy
import time
import datetime
import functools

start_running = time.time()

# input data
data_all = pd.read_excel('../data/Layout_Activity_4_12.xlsx')



test_date = "20140206"
convert_date = datetime.datetime.strptime(test_date, '%Y%m%d').date()

print(convert_date)