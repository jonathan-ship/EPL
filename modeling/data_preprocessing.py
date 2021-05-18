import pandas as pd
import numpy as np

activity_data = pd.read_excel('../data/Layout_Activity_A0001.xlsx')
bom_data = pd.read_excel('../data/Layout_BOM_A0001.xlsx')
activity_data['시작일'] = pd.to_datetime(activity_data['시작일'], format='%Y%m%d')
activity_data['종료일'] = pd.to_datetime(activity_data['종료일'], format='%Y%m%d')
initial_date = activity_data['시작일'].min()

activity_data['시작일'] = activity_data['시작일'].apply(lambda x: (x - initial_date).days)
activity_data['종료일'] = activity_data['종료일'].apply(lambda x: (x - initial_date).days)

activity_data['호선'] = activity_data['호선'].apply(lambda x: str(x))
activity_data['블록'] = activity_data['블록'].apply(lambda x: str(x) if not x == 324 else str(x) + 'E0')
activity_data['block code'] = activity_data['호선'] + '_' + activity_data['블록']
activity_data['process_head'] = activity_data['공정공종'].apply(lambda x: x[0])  # 공정공종 첫 알파벳
block_list = list(activity_data.drop_duplicates(['block code'])['block code'])

bom_data['호선'] = bom_data['호선'].apply(lambda x: str(x))
bom_data['블록'] = bom_data['블록'].apply(lambda x: str(x))
bom_data['상위블록'] = bom_data['상위블록'].apply(lambda x: str(x))
bom_data['block code'] = bom_data['호선'] + '_' + bom_data['블록']

new_activity = pd.DataFrame(columns=['block_code', 'process_code', 'start_date', 'finish_date', 'location', 'loc_indicator'])
process_head = ['A', 'B', 'C', 'F', 'G', 'H', 'J', 'M', 'K', 'L', 'N']
block_group = activity_data.groupby(activity_data['block code'])  ## block code에 따른 grouping
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
            latest_date = initial_date + pd.offsets.Day(np.max(grouped_by_process['종료일']))
            location_list = list(grouped_by_process.drop_duplicates(['작업부서'])['작업부서'])
            indicator = True if len(location_list) < 2 else False
            block.append(block_code)
            process_code.append(head)
            start_date.append(early_date)
            finish_date.append(latest_date)
            location.append(location_list[0])
            location_indicator.append(indicator)

new_activity['block_code'] = block
new_activity['process_code'] = process_code
new_activity['start_date'] = start_date
new_activity['finish_date'] = finish_date
new_activity['location'] = location
new_activity['loc_indicator'] = location_indicator
new_activity.to_excel('../data/new_activity.xlsx')