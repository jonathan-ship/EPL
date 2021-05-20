import pandas as pd
import numpy as np
from datetime import datetime
import time

start = time.time()

activity_data = pd.read_excel('../data/Layout_Activity.xlsx')

activity_data = activity_data[(activity_data['공정공종'] != 'C91') & (activity_data['공정공종'] != 'CX3') & \
                              (activity_data['공정공종'] != 'F91') & (activity_data['공정공종'] != 'FX3') & \
                              (activity_data['공정공종'] != 'G4A') & (activity_data['공정공종'] != 'G4B') & \
                              (activity_data['공정공종'] != 'GX3') & (activity_data['공정공종'] != 'HX3') & \
                              (activity_data['공정공종'] != 'K4A') & (activity_data['공정공종'] != 'K4B') & \
                              (activity_data['공정공종'] != 'L4B') & (activity_data['공정공종'] != 'L4A') & \
                              (activity_data['공정공종'] != 'LX3') & (activity_data['공정공종'] != 'JX3')]
activity_data = activity_data[(activity_data['시작일'] > 0) & (activity_data['종료일'] > 0)]
print("공정 골라내기 완료")
activity_data['시작일'] = pd.to_datetime(activity_data['시작일'], format='%Y%m%d')
activity_data['종료일'] = pd.to_datetime(activity_data['종료일'], format='%Y%m%d')
initial_date = activity_data['시작일'].min()

activity_data['시작일'] = activity_data['시작일'].apply(lambda x: (x - initial_date).days)
activity_data['종료일'] = activity_data['종료일'].apply(lambda x: (x - initial_date).days)
print("timestamp으로 바꾸기 완료")

activity_data['호선'] = activity_data['호선'].apply(lambda x: str(x))
activity_data['블록'] = activity_data['블록'].apply(lambda x: str(x) if not (type(x) == int and x < 1000) else str(x) + 'E0')
activity_data['block code'] = activity_data['호선'] + '_' + activity_data['블록']
activity_data['process_head'] = activity_data['공정공종'].apply(lambda x: x[0])  # 공정공종 첫 알파벳
block_list = list(activity_data.drop_duplicates(['block code'])['block code'])
print("블록 코드 구현 완료")

new_activity = pd.DataFrame(columns=['series_block_code','series', 'block_code', 'process_code', 'start_date', 'finish_date', 'location', 'loc_indicator'])
process_head = ['A', 'B', 'C', 'F', 'G', 'H', 'J', 'M', 'K', 'L', 'N']
block_group = activity_data.groupby(activity_data['block code'])  ## block code에 따른 grouping
series = []
only_block_code = []
block = []
process_code = []
start_date = []
finish_date = []
location = []
location_indicator = []

for block_code in block_list:
    block_data = block_group.get_group(block_code)  # block code 별로 grouping 한 결과
    for i in range(len(process_head)):
        head = process_head[i]  # 공정공종 첫 알파벳
        grouped_by_process = block_data[block_data['process_head'] == head]
        if len(grouped_by_process):
            early_date = initial_date + pd.offsets.Day(np.min(grouped_by_process['시작일']))
            early_date = datetime.strftime(early_date, '%Y%m%d')
            latest_date = initial_date + pd.offsets.Day(np.max(grouped_by_process['종료일']))
            latest_date = datetime.strftime(latest_date, '%Y%m%d')
            location_list = list(grouped_by_process.drop_duplicates(['작업부서'])['작업부서'])
            indicator = True if len(location_list) < 2 else False
            series.append(block_code[:5])
            only_block_code.append(block_code[-5:])
            block.append(block_code)
            process_code.append(head)
            start_date.append(early_date)
            finish_date.append(latest_date)
            location.append(location_list[0])
            location_indicator.append(indicator)
print("데이터 전처리 완료")

new_activity['series'] = series
new_activity['block_code'] = only_block_code
new_activity['series_block_code'] = block
new_activity['process_code'] = process_code
new_activity['start_date'] = start_date
new_activity['finish_date'] = finish_date
new_activity['location'] = location
new_activity['loc_indicator'] = location_indicator
new_activity.to_excel('../data/new_activity_ALL.xlsx')
print('Finish at', time.time() - start)
