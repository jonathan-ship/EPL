import simpy, random, copy, math
import pandas as pd
import numpy as np
from collections import OrderedDict
from datetime import timedelta


def record_capacity(capacity_dict, process_unit, block, type='used', inout=None):
    if process_unit == 'm2':  # 면적
        if inout == 'in':
            capacity_dict[process_unit][type] += block.area
        else:
            capacity_dict[process_unit][type] -= block.area
    elif process_unit == 'EA':  # 개수
        if inout == 'in':
            capacity_dict[process_unit][type] += 1
        else:
            capacity_dict[process_unit][type] -= 1
    else:  # ton수
        if inout == 'in':
            capacity_dict[process_unit][type] += block.weight
        else:
            capacity_dict[process_unit][type] -= block.weight

    return capacity_dict


def convert_process(block_dict, block, step, converting_dict, inout, network_distance, process_dict):
    present_process = block.data[(step, 'process')]
    if (present_process not in ['선행도장부', '선행의장부', '기장부', '의장1부', '의장3부', '의장2부', '도장1부', '도장2부', '발판지원부']) and (
            present_process not in converting_dict.keys()):
        return present_process
    elif present_process in ['건조1부', '건조2부', '건조3부']:
        return "{0}도크".format(block.dock)
    elif present_process == "Sink":
        return "Sink"
    elif (present_process not in ['선행도장부', '선행의장부', '기장부', '의장1부', '의장3부', '의장2부', '도장1부', '도장2부', '발판지원부']) and (
            present_process in converting_dict.keys()):
        return converting_dict[present_process]
    else:  # source_location 이 필요한 경우
        # determine source location
        source_location = None
        if (block.location is not None) and (block.location in process_dict.keys()):
            source_location = block.location
        elif block.child is not None:  # find from child of grand child block
            child_location_list = list()
            for child in block.child:
                if block_dict[child].location is not None:
                    child_location_list.append([block_dict[child].location, block_dict[child].weight])
                else:
                    if block_dict[child].child is not None:
                        for grandchild in block_dict[child].child:
                            if block_dict[grandchild].location is not None:
                                child_location_list.append(
                                    [block_dict[grandchild].location, block_dict[grandchild].weight])

            if len(child_location_list) == 0:
                source_location = "선각공장" if block.yard == 1 else "2야드 중조공장"
            else:
                child_location_list = sorted(child_location_list, key=lambda x: x[1], reverse=True)
                source_location = child_location_list[0][0]
        else:
            source_location = "선각공장" if block.yard == 1 else "2야드 중조공장"

        if present_process in ['도장1부', '도장2부', '발판지원부']:
            return source_location
        else:
            process_convert_by_dict = converting_dict[present_process]
            distance = []
            pre_choice = process_convert_by_dict[:]
            compared_process = inout[source_location][1]
            for process in process_convert_by_dict:
                process_temp = inout[process][0]
                if (process_temp in network_distance.keys()) and (compared_process in network_distance.keys()) and \
                        (network_distance[process_temp][compared_process] is not None) and \
                        (network_distance[process_temp][compared_process] < 100000):
                    temp_process = process_dict[process]
                    if temp_process.capacity_unit == 'm2':
                        if_to_be_capacity = temp_process.capacity - temp_process.capacity_dict['m2'][
                            'preemptive'] - block.area
                    elif temp_process.capacity_unit == 'EA':
                        if_to_be_capacity = temp_process.capacity - temp_process.capacity_dict['EA']['preemptive'] - 1
                    else:
                        if_to_be_capacity = temp_process.capacity - temp_process.capacity_dict['ton'][
                            'preemptive'] - block.weight

                    if if_to_be_capacity > 0:
                        distance.append(network_distance[process_temp][compared_process])
                    else:
                        pre_choice.remove(process)
                else:
                    pre_choice.remove(process)
            if len(distance) > 0:
                process_idx = distance.index(min(distance))
                process = pre_choice[process_idx]
            else:
                if present_process == '선행도장부':
                    process = 'Painting'
                else:
                    process = 'Shelter'

            block.data[(step, 'process')] = process
            block_dict[block.name].data[(step, 'process')] = process
            process_dict[process].capacity_dict = record_capacity(process_dict[process].capacity_dict,
                                                                  process_dict[process].capacity_unit,
                                                                  block_dict[block.name], type='preemptive',
                                                                  inout='in')
            return process


class Block:
    def __init__(self, name, area, size, weight, data, yard, dock, child=None):
        self.name = name
        self.area = area
        self.size = size  # 임시
        self.weight = weight
        columns = pd.MultiIndex.from_product([[i for i in range(8)], ['start_time', 'process_time', 'process', 'work']])
        self.data = pd.Series(data, index=columns)
        self.dock = dock
        self.child = child

        self.yard = yard
        self.location = None
        self.step = 0
        self.its_start_time = 0.0


class Part:
    def __init__(self, name, env, process_data, processes, monitor, block, blocks=None, resource=None,
                 network=None, child=None, parent=None, stocks=None, Inout=None, convert_to_process=None,
                 dock=None, size=None, stock_lag=2):
        self.name = name
        self.env = env
        self.block = block  # class로 모델링한 블록
        columns = pd.MultiIndex.from_product([[i for i in range(8)], ['start_time', 'process_time', 'process', 'work']])
        self.blocks = blocks
        self.data = pd.Series(process_data, index=columns)
        self.processes = processes
        self.monitor = monitor
        self.resource = resource
        self.network_distance = network
        self.child = child
        self.parent = parent
        self.stocks = stocks
        self.inout = Inout
        self.convert_to_process = convert_to_process
        ## 1, 2, 3 도크에서 작업되면 1야드, 8, 9도크에서 작업되면 2야드
        self.yard = 1 if dock in [1, 2, 3, 4, 5] else 2
        self.block.yard = self.yard
        self.dock = dock
        self.size = size
        self.stock_lag = stock_lag

        self.in_child = {'Parent': list(), 'Stock': list()}
        self.part_store = simpy.Store(env, capacity=1)
        self.assembly_idx = False
        if self.child is None:
            self.part_store.put(self.block)
        self.num_clone = 0
        self.moving_distance_w_TP = []
        self.moving_time = []

        self.tp_flag = False
        self.assemble_flag = False if self.child is not None else True
        self.in_stock = False
        self.where_stock = None
        self.finish = False
        self.process_not_determined = False

        self.part_delay = []
        self.resource_delay = env.event()
        self.assemble_delay = {}
        env.process(self._sourcing)

    @property
    def _sourcing(self):
        step = 0
        process = None
        previous_tp_flag = False
        while not self.finish:
            tp = None
            waiting = None
            part = None
            # 이번 step에서 가야 할 공정
            # part 호출
            start_time = self.data[(step, 'start_time')]
            if self.data[(step, 'process')] != 'Sink':
                lag = start_time - self.env.now
                if lag > 0:
                    yield self.env.timeout(lag)
            convert_process_by_function = convert_process(self.blocks, self.block, step, self.convert_to_process,
                                                          self.inout, self.network_distance, self.processes)
            if convert_process_by_function not in self.processes.keys():
                print(0)
            process = self.processes[convert_process_by_function]
            if process.name == 'Sink':
                self.finish = True
                self.tp_flag = False
            if step == 0:
                if (self.process_not_determined is True) and (len(self.in_child['Parent']) >= 1):
                    for child in self.in_child['Parent']:
                        child_block = child[0]
                        child_tp_flag = child[1]
                        used_resource = "Transporter" if child_tp_flag is True else None
                        self.monitor.record(self.env.now, None, None, part_id=child_block.name, event="Child to Parent",
                                            resource=used_resource, load=child_block.weight, from_process=self.blocks[child_block.name].location,
                                            to_process=process.name,
                                            distance=self.network_distance[self.inout[self.blocks[child_block.name].location][1]][self.inout[process.name][0]])
                elif len(self.in_child['Stock']) >= 1:
                    for child in self.in_child['Stock']:
                        child_name = child[0]
                        child_stock = child[1]
                        temp = yield self.stocks[child_stock].stock_yard.get(lambda x: x[1].name == child_name)
                        child_part = temp[1]
                        self.stocks[child_stock].capacity_dict = record_capacity(
                            self.stocks[child_stock].capacity_dict, self.stocks[child_stock].capacity_unit,
                            self.blocks[child_part.name], type='used', inout='out')
                        self.stocks[child_stock].capacity_dict = record_capacity(
                            self.stocks[child_stock].capacity_dict, self.stocks[child_stock].capacity_unit,
                            self.blocks[child_part.name], type='preemptive', inout='out')
                        self.monitor.record(self.env.now, child_stock, None, part_id=child_part.name,
                                            event="Stock Out", process_type="Stock",
                                            load=self.stocks[child_stock].capacity_dict[
                                                self.stocks[child_stock].capacity_unit]['used'],
                                            unit=self.stocks[child_stock].capacity_unit, memo="for parent block")

                        child_last_work = child_part.data[(child_part.step, 'work')]
                        tp_flag = False
                        if ('F' in child_last_work) or ('G' in child_last_work) or ('H' in child_last_work) or (
                                'J' in child_last_work) or ('M' in child_last_work) or ('K' in child_last_work) or (
                                'L' in child_last_work) or ('N' in child_last_work):
                            tp_flag = True
                        used_resource = "Transporter" if tp_flag is True else None
                        self.monitor.record(self.env.now, None, None, part_id=child_part.name, event="Child to Parent",
                                            resource=used_resource, load=child_part.weight, from_process=child_stock,
                                            to_process=process.name,
                                            distance=self.network_distance[self.inout[child_stock][1]][self.inout[process.name][0]])

                    if len(self.in_child['Stock']) + len(self.in_child['Parent']) == len(self.child):
                        if len(self.assemble_delay) > 0:
                            end_delay = list(self.assemble_delay.keys())
                            self.assemble_delay[end_delay[0]].succeed()
                            self.monitor.record(self.env.now, None, None, part_id=self.name, event="process delay finish")
            ## 이전 tp flag 도 따로 저장 -> Stock에서 불러올 때 tp 사용하는 기준은 이전 단계 기준
            work = self.data[(step, 'work')] if process.name != 'Sink' else 'A'

            # transporter 사용 여부
            if ('F' in work) or ('G' in work) or ('H' in work) or ('J' in work) or ('M' in work) or ('K' in work) or \
                    ('L' in work) or ('N' in work):
                if (process.name != 'Sink') or (step > 0):
                    self.tp_flag = True

            if step == 0:  # 첫 단계이면
                part = yield self.part_store.get()
                self.monitor.record(self.env.now, None, None, part_id=self.name, event="Block Created")
                self.blocks['Created Part'] += 1
                print("{0} created at {1}, Creating Part is finished {2}/{3}".format(self.name, self.env.now, self.blocks['Created Part'], len(self.blocks)-1))
            elif self.in_stock:  # 적치장에 있는 경우 -> 시간에 맞춰서 꺼내줘야 함
                temp = yield self.stocks[self.where_stock].stock_yard.get(lambda x: x[1].name == self.name)
                part = temp[1]
                self.stocks[self.where_stock].capacity_dict = record_capacity(
                    self.stocks[self.where_stock].capacity_dict, self.stocks[self.where_stock].capacity_unit,
                    self.blocks[self.name], type='used', inout='out')
                self.stocks[self.where_stock].capacity_dict = record_capacity(
                    self.stocks[self.where_stock].capacity_dict, self.stocks[self.where_stock].capacity_unit,
                    self.blocks[self.name], type='preemptive', inout='out')
                previous_process = self.where_stock
                self.monitor.record(self.env.now, self.where_stock, None, part_id=self.name,
                                    event="Stock out", process_type="Stock",
                                    load=self.stocks[self.where_stock].capacity_dict[
                                        self.stocks[self.where_stock].capacity_unit]['used'],
                                    unit=self.stocks[self.where_stock].capacity_unit, memo="for next process")

                self.where_stock = None
            else:  # get part from previous process
                part = yield self.part_store.get()

            # Go to Next Priocess
            if process.name == 'Sink':
                process.put(part)
                self.finish = True

            else:
                self.blocks[self.name].location = process.name
                self.blocks[self.name].step = step
                process.buffer_to_machine.put(part)
                used_resource = "Transporter" if self.tp_flag is True else None
                if self.in_stock is True:  # 적치장에서 공장으로 갈 때
                    self.monitor.record(self.env.now, None, None, part_id=self.name, event="Stock to Process",
                                        resource=used_resource, load=self.block.weight, unit="ton",
                                        from_process=previous_process, to_process=process.name,
                                        distance=self.network_distance[self.inout[previous_process][1]][self.inout[process.name][0]])
                    self.in_stock = False
                else:  # 공장에서 공장으로 갈 때
                    if step >= 1:
                        self.monitor.record(self.env.now, None, None, part_id=self.name, event="Process to Process",
                                            resource=used_resource, load=self.block.weight, unit="ton",
                                            from_process=previous_process, to_process=process.name,
                                            distance=self.network_distance[self.inout[previous_process][1]][self.inout[process.name][0]])

                step += 1
                previous_process = process.name

    def return_to_part(self, part):
        tp_flag = False
        next_start_time = self.data[(part.step + 1, 'start_time')]

        work = self.data[(part.step, 'work')] if self.data[(part.step, 'process')] != 'Sink' else 'A'
        # transporter 사용 여부
        if ('F' in work) or ('G' in work) or ('H' in work) or ('J' in work) or ('M' in work) or ('K' in work) or \
                ('L' in work) or ('N' in work):
            tp_flag = True

        if (next_start_time is not None) and (next_start_time <= 100000) and (
                next_start_time - self.env.now >= self.stock_lag):  # 적치장 가야 함
            next_process_name = convert_process(self.blocks, self.block, part.step + 1, self.convert_to_process, self.inout,
                                                     self.network_distance, self.processes)
            next_process = self.inout[next_process_name][0]

            # 다음 적치장 탐색
            next_stock = self._find_stock(next_process)
            used_resource = "Transporter" if tp_flag is True else None
            self.monitor.record(self.env.now, None, None, part_id=self.name, event="Process to Stockyard",
                                resource=used_resource, load=self.block.weight, unit="ton",
                                from_process=part.location, to_process=next_stock,
                                distance=self.network_distance[self.inout[part.location][1]][self.inout[next_stock][0]])
            self.stocks[next_stock].put(part, next_start_time)

            self.in_stock = True
            self.where_stock = next_stock

        else:  # 적치장을 갈 필요도 없고, clone이 아닌 경우에만 return
            self.part_store.put(part)
            self.tp_flag = False
            self.where_stock = None
            if len(self.part_delay) > 0:
                self.part_delay.pop(0).succeed()

    def _find_stock(self, next_process):
        stock_list = list(self.stocks.keys())
        stock_list.remove('Stock')

        idx = 0
        for stock_name in stock_list:
            if self.block.area > self.stocks[stock_name].capacity:
               idx += 1

        if idx == len(stock_list):
            self.monitor.record(self.env.now, None, None, part_id=self.name, memo="No stock can contain its area")

        next_stock = None
        shortest_path = 1e8

        for idx in range(len(stock_list)):
            temp_stock = self.stocks[stock_list[idx]]
            if temp_stock.capacity_unit == 'm2':
                if_to_be_capacity = temp_stock.capacity - temp_stock.capacity_dict['m2'][
                    'preemptive'] - self.block.area
            elif temp_stock.capacity_unit == 'EA':
                if_to_be_capacity = temp_stock.capacity - temp_stock.capacity_dict['EA']['preemptive'] - 1
            else:
                if_to_be_capacity = temp_stock.capacity - temp_stock.capacity_dict['ton'][
                    'preemptive'] - self.block.weight
            if if_to_be_capacity > 0:
                if next_stock is None:  # No stock to compare
                    next_stock = stock_list[idx]
                    shortest_path = self.network_distance[next_process][self.inout[stock_list[idx]][0]]
                else:
                    compared_path = self.network_distance[next_process][self.inout[stock_list[idx]][0]]
                    if shortest_path > compared_path:  # replaced
                        next_stock = stock_list[idx]
                        shortest_path = compared_path
                    elif shortest_path == compared_path:
                        shortest_stock_len = len(self.stocks[next_stock].stock_yard.items)
                        compared_stock_len = len(self.stocks[stock_list[idx]].stock_yard.items)
                        next_stock = stock_list[idx] if compared_stock_len < shortest_stock_len else next_stock

        next_stock = next_stock if next_stock is not None else 'Stock'
        self.stocks[next_stock].capacity_dict = record_capacity(self.stocks[next_stock].capacity_dict,
                                                                self.stocks[next_stock].capacity_unit,
                                                                self.blocks[self.name], type='preemptive',
                                                                inout='in')

        return next_stock


class Sink:
    def __init__(self, env, processes, parts, monitor, network, stocks, inout, converting, stock_lag=2):
        self.env = env
        self.name = 'Sink'
        self.processes = processes
        self.parts = parts
        self.monitor = monitor
        self.network = network
        self.stocks = stocks
        self.inout = inout
        self.converting = converting
        self.stock_lag = stock_lag

        self.parts_rec = 0
        self.last_arrival = 0.0

    def put(self, part):
        if self.parts[part.name].parent is not None:  # 상위 블록이 있는 경우
            parent_block = self.parts[part.name].parent

            child_last_work = part.data[(part.step, 'work')]
            tp_flag = False
            if ('F' in child_last_work) or ('G' in child_last_work) or ('H' in child_last_work) or (
                    'J' in child_last_work) or ('M' in child_last_work) or ('K' in child_last_work) or (
                    'L' in child_last_work) or ('N' in child_last_work):
                tp_flag = True
            used_resource = "Transporter" if tp_flag is True else None
            parent_first_process = self.parts[parent_block].data[(0, 'process')]
            if parent_first_process in ['선행의장부', '기장부', '의장1부', '의장2부', '의장3부', '선행도장부', '도장1부',
                                        '도장2부', '발판지원부']:
                self.parts[parent_block].process_not_determined = True

            parent_start_time = self.parts[parent_block].data[(0, 'start_time')]

            erected_residual_time = parent_start_time - self.env.now
            if erected_residual_time >= self.stock_lag:  # Parent Block 시작까지 2일 이상 남음 --> Child는 적치장에 있다가 합침
                standard_process = part.location if parent_first_process in ['선행의장부', '기장부', '의장1부', '의장2부', '의장3부', '선행도장부', '도장1부', '도장2부', '발판지원부'] else parent_first_process

                if standard_process in self.converting.keys():
                    standard_process = self.converting[standard_process]
                next_stock = self._find_stock(standard_process, part)  # 이 때 적치장은 현재 있는 데서 가까운 데로

                self.monitor.record(self.env.now, None, None, part_id=part.name, event="Process to Stockyard",
                                    resource=used_resource, load=part.weight, from_process=part.location,
                                    to_process=next_stock, distance=self.network[self.inout[part.location][1]][self.inout[next_stock][0]],
                                    memo="More than {0} days left to start Parent Block".format(self.stock_lag))
                self.stocks[next_stock].put(part, parent_start_time)  # Child Block은 적치장에 입고
                self.parts[parent_block].in_child['Stock'].append((part.name, next_stock))
            else:  # 바로 Parent Block이랑 합쳐지는 경우
                if self.parts[parent_block].process_not_determined is False:
                    if parent_first_process in self.converting.keys():
                        parent_first_process = self.converting[parent_first_process]
                    if parent_first_process == "Dock":
                        parent_first_process = "{0}도크".format(self.parts[parent_block].blocks[parent_block].dock)
                    self.monitor.record(self.env.now, None, None, part_id=part.name, event="Child to Parent",
                                        resource=used_resource, load=part.weight, from_process=part.location,
                                        to_process=parent_first_process,
                                        distance=self.network[self.inout[part.location][1]][self.inout[parent_first_process][0]])
                self.parts[parent_block].in_child['Parent'].append([part, tp_flag])

            if (len(self.parts[parent_block].part_store.items) == 0) and (
                    self.parts[parent_block].assembly_idx is False):
                self.parts[parent_block].part_store.put(self.parts[parent_block].block)
                self.parts[parent_block].assembly_idx = True
            if (len(self.parts[parent_block].in_child['Parent']) + len(
                    self.parts[parent_block].in_child['Stock'])) == len(self.parts[parent_block].child):
                self.parts[parent_block].assemble_flag = True
                if len(self.parts[parent_block].assemble_delay):
                    end_delay = list(self.parts[parent_block].assemble_delay.keys())
                    self.parts[parent_block].assemble_delay[end_delay[0]].succeed()

            print("Ready to Assemble, parent block = {0} at {1}".format(parent_block, self.env.now))

        self.parts_rec += 1
        self.last_arrival = self.env.now
        self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="Block Completed")
        print("Finished Part {0}/{1}".format(self.parts_rec, len(self.parts)))

    def _find_stock(self, process, part):
        stock_list = list(self.stocks.keys())
        stock_list.remove('Stock')
        if process == 'Dock':
            dock_num = part.dock
            process = "{0}도크".format(dock_num)
        if process in ['도장1부', '도장2부']:
            print(0)
        idx = 0
        for stock_name in stock_list:
            if part.area > self.stocks[stock_name].capacity:
               idx += 1

        if idx == len(stock_list):
            self.monitor.record(self.env.now, None, None, part_id=self.name, memo="No stock can contain its area")

        next_stock = None
        shortest_path = 1e8
        for idx in range(len(stock_list)):
            temp_stock = self.stocks[stock_list[idx]]
            if temp_stock.capacity_unit == 'm2':
                if_to_be_capacity = temp_stock.capacity - temp_stock.capacity_dict['m2'][
                    'preemptive'] - part.area
            elif temp_stock.capacity_unit == 'EA':
                if_to_be_capacity = temp_stock.capacity - temp_stock.capacity_dict['EA']['preemptive'] - 1
            else:
                if_to_be_capacity = temp_stock.capacity - temp_stock.capacity_dict['ton'][
                    'preemptive'] - part.weight
            if if_to_be_capacity > 0:
                if next_stock is None:  # No stock to compare
                    next_stock = stock_list[idx]
                    shortest_path = self.network[self.inout[process][1]][self.inout[stock_list[idx]][0]]
                else:
                    compared_path = self.network[self.inout[process][1]][self.inout[stock_list[idx]][0]]
                    if shortest_path > compared_path:  # replaced
                        next_stock = stock_list[idx]
                        shortest_path = compared_path
                    elif shortest_path == compared_path:
                        shortest_stock_len = len(self.stocks[next_stock].stock_yard.items)
                        compared_stock_len = len(self.stocks[stock_list[idx]].stock_yard.items)
                        next_stock = stock_list[idx] if compared_stock_len < shortest_stock_len else next_stock

        next_stock = next_stock if next_stock is not None else 'Stock'
        self.stocks[next_stock].capacity_dict = record_capacity(self.stocks[next_stock].capacity_dict,
                                                                self.stocks[next_stock].capacity_unit,
                                                                part, type='preemptive', inout='in')

        return next_stock


class Process:
    def __init__(self, env, name, machine_num, processes, parts, monitor, resource=None,
                 process_time=None, capacity=float('inf'), routing_logic='cyclic', priority=None,
                 capa_to_machine=10000, capa_to_process=float('inf'), MTTR=None, MTTF=None,
                 initial_broken_delay=None, delay_time=None, workforce=None, convert_dict=None, unit="m2",
                 process_type=None, stock_lag=2):

        # input data
        self.env = env
        self.name = name
        self.processes = processes
        self.parts = parts
        self.monitor = monitor
        self.resource = resource
        self.capacity = capacity
        self.capacity_unit = unit
        self.machine_num = machine_num
        self.routing_logic = routing_logic
        self.process_time = process_time[self.name] if process_time is not None else [None for _ in
                                                                                      range(machine_num)]
        self.priority = priority[self.name] if priority is not None else [1 for _ in range(machine_num)]
        self.MTTR = MTTR[self.name] if MTTR is not None else [None for _ in range(machine_num)]
        self.MTTF = MTTF[self.name] if MTTF is not None else [None for _ in range(machine_num)]
        self.initial_broken_delay = initial_broken_delay[self.name] if initial_broken_delay is not None else [None
                                                                                                              for _
                                                                                                              in
                                                                                                              range(
                                                                                                                  machine_num)]
        self.delay_time = delay_time[name] if delay_time is not None else None
        self.workforce = workforce[self.name] if workforce is not None else [False for _ in
                                                                             range(machine_num)]  # workforce 사용 여부
        self.converting = convert_dict
        self.process_type = process_type

        # variable defined in class
        self.in_process = 0
        self.parts_sent = 0
        self.parts_sent_to_machine = 0
        self.machine_idx = 0
        self.len_of_server = []
        self.waiting_machine = OrderedDict()
        self.waiting_pre_process = OrderedDict()
        self.finish_time = 0.0
        self.start_time = 0.0
        self.event_area = []
        self.event_time = []
        self.event_block_num = []

        self.capacity_dict = {'EA': {'used': 0, 'preemptive': 0}, 'm2': {'used': 0.0, 'preemptive': 0.0},
                              'ton': {'used': 0, 'preemptive': 0.0}}

        # buffer and machine
        self.buffer_to_machine = simpy.Store(env, capacity=capa_to_machine)
        self.buffer_to_process = simpy.Store(env, capacity=capa_to_process)
        self.machine = [
            Machine(env, '{0}_{1}'.format(self.name, i), self.name, self.parts, self.processes, self.resource,
                    process_time=self.process_time[i], priority=self.priority[i],
                    waiting=self.waiting_machine, monitor=monitor, MTTF=self.MTTF[i], MTTR=self.MTTR[i],
                    initial_broken_delay=self.initial_broken_delay[i],
                    workforce=self.workforce[i], stock_lag=stock_lag) for i in range(self.machine_num)]
        # resource
        self.tp_store = dict()

        # Capacity Delay
        self.capacity_delay = OrderedDict()

        # get run functions in class
        env.process(self.to_machine())

    def to_machine(self):
        while True:
            routing = Routing(self.machine, priority=self.priority)
            if self.delay_time is not None:
                delaying_time = self.delay_time if type(self.delay_time) == float else self.delay_time()
                yield self.env.timeout(delaying_time)

            part = yield self.buffer_to_machine.get()
            if self.in_process == 0:
                self.start_time = self.env.now
            self.in_process += 1
            self.event_block_num.append(self.in_process)

            self.capacity_dict = record_capacity(self.capacity_dict, self.capacity_unit,
                                                 self.parts[part.name].blocks[part.name], type='used', inout='in')
            self.event_area.append(self.parts[part.name].blocks[part.name].area)
            self.event_time.append(self.env.now)

            self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="Process In",
                                process_type=self.process_type, load=self.capacity_dict[self.capacity_unit]['used'],
                                unit=self.capacity_unit)

            ## Rouring logic 추가 할 예정
            if self.routing_logic == 'priority':
                self.machine_idx = routing.priority()
            else:
                self.machine_idx = 0 if (self.parts_sent_to_machine == 0) or (
                        self.machine_idx == self.machine_num - 1) else self.machine_idx + 1

            self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="routing_ended")
            self.machine[self.machine_idx].machine.put(part)
            self.parts_sent_to_machine += 1

            # finish delaying of pre-process
            if (len(self.buffer_to_machine.items) < self.buffer_to_machine.capacity) and (
                    len(self.waiting_pre_process) > 0):
                self.waiting_pre_process.popitem(last=False)[1].succeed()  # delay = (part_id, event)


class Machine:
    def __init__(self, env, name, process_name, parts, processes, resource, process_time, priority, waiting, monitor,
                 MTTF, MTTR, initial_broken_delay, workforce, stock_lag):
        # input data
        self.env = env
        self.name = name
        self.process_name = process_name
        self.parts = parts
        self.processes = processes
        self.resource = resource
        self.process_time = process_time
        self.priority = priority
        self.waiting = waiting
        self.monitor = monitor
        self.MTTR = MTTR
        self.MTTF = MTTF
        self.initial_broken_delay = initial_broken_delay
        self.workforce = workforce
        self.stock_lag = stock_lag

        # variable defined in class
        self.machine = simpy.Store(env)
        self.working_start = 0.0
        self.total_time = 0.0
        self.total_working_time = 0.0
        self.working = False  # whether machine's worked(True) or idled(False)
        self.broken = False  # whether machine is broken or not
        self.unbroken_start = 0.0
        self.planned_proc_time = 0.0

        # broke and re-running
        self.residual_time = 0.0
        self.broken_start = 0.0
        if self.MTTF is not None:
            mttf_time = self.MTTF if type(self.MTTF) == float else self.MTTF()
            self.broken_start = self.unbroken_start + mttf_time
        # get run functions in class
        self.action = env.process(self.work())
        # if (self.MTTF is not None) and (self.MTTR is not None):
        #     env.process(self.break_machine())

    def work(self):
        while True:
            self.broken = True
            part = yield self.machine.get()
            self.working = True
            wf = None
            # process_time
            if self.process_time == None:  # part에 process_time이 미리 주어지는 경우
                proc_time = self.parts[part.name].data[(part.step, "process_time")]
            else:  # service time이 정해진 경우 --> 1) fixed time / 2) Stochastic-time
                proc_time = self.process_time if type(self.process_time) == float else self.process_time()
            self.planned_proc_time = proc_time

            if self.workforce is True:
                resource_item = list(map(lambda item: item.name, self.resource.wf_store.items))
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="workforce_request", resource=resource_item)
                while len(self.resource.wf_store.items) == 0:
                    self.resource.wf_waiting[part.id] = self.env.event()
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="delay_start_machine_cus_no_resource")
                    yield self.resource.wf_waiting[part.id]  # start delaying

                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="delay_finish_machine_cus_yes_resource")
                wf = yield self.resource.wf_store.get()
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="workforce get in the machine", resource=wf.name)
            if proc_time is None:
                print("Error!")


            while proc_time > 0:
                if self.MTTF is not None:
                    self.env.process(self.break_machine())
                try:
                    self.broken = False
                    ## working start

                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="Work Start")

                    self.working_start = self.env.now
                    yield self.env.timeout(proc_time)

                    ## working finish
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="Work Finish")
                    self.total_working_time += self.env.now - self.working_start
                    self.broken = True
                    proc_time = 0.0

                except simpy.Interrupt:
                    self.broken = True
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="machine_broken")
                    print('{0} is broken at '.format(self.name), self.env.now)
                    proc_time -= self.env.now - self.working_start
                    if self.MTTR is not None:
                        repair_time = self.MTTR if type(self.MTTR) == float else self.MTTR()
                        yield self.env.timeout(repair_time)
                        self.unbroken_start = self.env.now
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="machine_rerunning")
                    print(self.name, 'is solved at ', self.env.now)
                    self.broken = False

                    mttf_time = self.MTTF if type(self.MTTF) == float else self.MTTF()
                    self.broken_start = self.unbroken_start + mttf_time

            self.working = False

            if self.parts[part.name].assemble_flag is False:
                self.parts[part.name].assemble_delay[self.name] = self.env.event()
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="process delay",
                                    memo='All child do not arrive to parent {0}'.format(part.name))
                yield self.parts[part.name].assemble_delay[self.name]
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="process delay finish",
                                    memo='All child arrive to parent {0}'.format(part.name))

            # if the part doesn't go to stock before to next process -> stay here before to go to next process
            if (part.data[(part.step + 1, 'process')] != 'Sink') and (part.data[(part.step + 1, 'start_time')] - self.env.now < self.stock_lag):
                residual_time = part.data[(part.step + 1, 'start_time')] - self.env.now
                if residual_time > 0:
                    yield self.env.timeout(residual_time)
            elif part.data[(part.step + 1, 'process')] == 'Sink':
                if self.parts[part.name].parent is not None:
                    parent_block = self.parts[part.name].parent
                    residual_time = self.parts[parent_block].data[(0, 'start_time')] - self.env.now
                    if (residual_time < self.stock_lag) and (residual_time > 0):
                        yield self.env.timeout(residual_time)

            # transfer to 'to_process' function
            its_process = self.processes[self.process_name]
            its_process.capacity_dict = record_capacity(its_process.capacity_dict, its_process.capacity_unit,
                                                        part, type='used', inout='out')
            its_process.capacity_dict = record_capacity(its_process.capacity_dict, its_process.capacity_unit,
                                                        part, type='preemptive', inout='out')
            self.parts[part.name].return_to_part(part)
            self.processes[self.process_name].in_process -= 1
            self.processes[self.process_name].event_block_num.append(self.processes[self.process_name].in_process)
            self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                event="Process Out",
                                load=its_process.capacity_dict[its_process.capacity_unit]['used'],
                                unit=its_process.capacity_unit,
                                process_type=self.processes[self.process_name].process_type)

            self.total_time += self.env.now - self.working_start

    def break_machine(self):
        if (self.working_start == 0.0) and (self.initial_broken_delay is not None):
            initial_delay = self.initial_broken_delay if type(
                self.initial_broken_delay) == float else self.initial_broken_delay()
            yield self.env.timeout(initial_delay)
        residual_time = self.broken_start - self.working_start
        if (residual_time > 0) and (residual_time < self.planned_proc_time):
            yield self.env.timeout(residual_time)
            self.action.interrupt()
        else:
            return


class StockYard:
    def __init__(self, env, name, parts, monitor, capacity=float("inf"), unit='m2'):
        self.name = name
        self.env = env
        self.parts = parts
        self.monitor = monitor
        self.capacity = capacity
        self.capacity_unit = unit

        self.stock_yard = simpy.FilterStore(env)
        self.tp_store = dict()
        self.event_area = []
        self.event_time = []

        self.capacity_dict = {'EA': {'used': 0, 'preemptive': 0}, 'm2': {'used': 0.0, 'preemptive': 0.0},
                              'ton': {'used': 0, 'preemptive': 0.0}}

    def put(self, part, out_time):
        self.capacity_dict = record_capacity(self.capacity_dict, self.capacity_unit, part, type='used', inout='in')
        self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="Stock In",
                            load=self.capacity_dict[self.capacity_unit]['used'], unit=self.capacity_unit,
                            process_type="Stock")
        self.stock_yard.put([out_time, part])


class Monitor:
    def __init__(self, result_path, project_name, initial_date):
        self.file_path = result_path
        self.project_name = project_name
        self.initial_date = initial_date

        self.time = list()
        self.date = list()
        self.event = list()
        self.part_id = list()
        self.process = list()
        self.subprocess = list()
        self.resource = list()
        self.memo = list()
        self.process_type = list()
        self.load = list()
        self.unit = list()
        self.from_process = list()
        self.to_process = list()
        self.distance = list()

        self.created = 0
        self.completed = 0
        self.tp_used = dict()
        self.road_used = dict()

    def record(self, time, process, subprocess, part_id=None, event=None, resource=None, memo=None, process_type=None,
               load=None, unit=None, from_process=None, to_process=None, distance=None):
        self.time.append(time)
        date_time = self.initial_date + timedelta(days=math.floor(time))
        self.date.append(date_time.date())
        self.event.append(event)
        self.part_id.append(part_id)
        self.process.append(process)
        self.subprocess.append(subprocess)
        self.resource.append(resource)
        self.memo.append(memo)
        self.process_type.append(process_type)
        self.load.append(load)
        self.unit.append(unit)
        self.from_process.append(from_process)
        self.to_process.append(to_process)
        self.distance.append(distance)

        if event == 'Block Created':
            self.created += 1
        elif event == 'Block Completed':
            self.completed += 1

    def record_road_used(self, from_process, to_process):
        if from_process not in self.road_used.keys():
            self.road_used[from_process] = dict()
        if to_process not in self.road_used[from_process].keys():
            self.road_used[from_process][to_process] = 0

        self.road_used[from_process][to_process] += 1

    def record_tp_record(self, tp_name, moving_time, moving_distance, loaded=True):
        if tp_name not in self.tp_used.keys():
            self.tp_used[tp_name] = {'loaded': {'moving_time': [], 'moving_distance': []},
                                     'unloaded': {'moving_time': [], 'moving_distance': []}}

        if loaded:
            self.tp_used[tp_name]['loaded']['moving_time'].append(moving_time)
            self.tp_used[tp_name]['loaded']['moving_distance'].append(moving_distance)
        else:
            self.tp_used[tp_name]['unloaded']['moving_time'].append(moving_time)
            self.tp_used[tp_name]['unloaded']['moving_distance'].append(moving_distance)

    def save_information(self):
        # 1. event_tracer
        event_tracer = pd.DataFrame()
        event_tracer['Date'] = self.date
        event_tracer['Part'] = self.part_id
        event_tracer['Process'] = self.process
        event_tracer['SubProcess'] = self.subprocess
        event_tracer['Resource'] = self.resource
        event_tracer['Event'] = self.event
        event_tracer['Process Type'] = self.process_type
        event_tracer['Load'] = self.load
        event_tracer['Unit'] = self.unit
        event_tracer['From'] = self.from_process
        event_tracer['To'] = self.to_process
        event_tracer['Distance'] = self.distance
        event_tracer['Memo'] = self.memo
        event_tracer['Simulation time'] = self.time

        path_event_tracer = self.file_path + 'result_{0}.csv'.format(self.project_name)

        event_tracer.to_csv(path_event_tracer, encoding='utf-8-sig')

        return path_event_tracer


class Routing:
    def __init__(self, server_list=None, priority=None):
        self.server_list = server_list
        self.idx_priority = np.array(priority)

    def priority(self):
        i = min(self.idx_priority)
        idx = 0
        while i <= max(self.idx_priority):
            min_idx = np.argwhere(self.idx_priority == i)  # priority가 작은 숫자의 index부터 추출
            idx_min_list = min_idx.flatten().tolist()
            # 해당 index list에서 machine이 idling인 index만 추출
            idx_list = list(filter(lambda j: (self.server_list[j].working == False), idx_min_list))
            if len(idx_list) > 0:  # 만약 priority가 높은 machine 중 idle 상태에 있는 machine이 존재한다면
                idx = random.choice(idx_list)
                break
            else:  # 만약 idle 상태에 있는 machine이 존재하지 않는다면
                if i == max(self.idx_priority):  # 그 중 모든 priority에 대해 machine이 가동중이라면
                    idx = random.choice([j for j in range(len(self.idx_priority))])  # 그냥 무작위 배정
                    # idx = None
                    break
                else:
                    i += 1  # 다음 priority에 대하여 따져봄
        return idx

    def first_possible(self):
        idx_possible = random.choice([j for j in range(len(self.server_list))])  # random index로 초기화 - 모든 서버가 가동중일 때, 서버에 random하게 파트 할당
        for i in range(len(self.server_list)):
            if self.server_list[i].working is False:  # 만약 미가동중인 server가 존재할 경우, 해당 서버에 part 할당
                idx_possible = i
                break
        return idx_possible
