import simpy, os, random, math, json
import pandas as pd
import numpy as np
from collections import OrderedDict, namedtuple


transporter = namedtuple("Transporter", "name, v_loaded, v_unloaded, moving_dist, moving_time")
workforce = namedtuple("Workforce", "name, skill")
block = namedtuple("Block", "name, location, step, clone, area")


class Resource(object):
    def __init__(self, env, processes, stocks, monitor, tp_info=None, wf_info=None, delay_time=None, network=None,
                 inout=None):
        self.env = env
        self.processes = processes
        self.stocks = stocks
        self.monitor = monitor
        self.delay_time = delay_time
        self.network = network
        self.inout = inout

        # resource 할당
        self.tp_store = simpy.Store(env)
        self.wf_store = simpy.FilterStore(env)
        # resource 위치 파악
        self.tp_location = {}
        self.wf_location = {}

        if tp_info is not None:
            for name in tp_info.keys():
                self.tp_location[name] = []
                self.tp_store.put(transporter(name=name, v_loaded=tp_info[name]["v_loaded"],
                                              v_unloaded=tp_info[name]["v_unloaded"], moving_dist=0.0,
                                              moving_time=0.0))
            # No resource is in resource store -> machine hv to wait
            self.tp_waiting = OrderedDict()

        if wf_info is not None:
            for name in wf_info.keys():
                self.wf_location[name] = []
                self.wf_store.put(workforce(name, wf_info[name]["skill"]))
            # No resource is in resource store -> machine hv to wait
            self.wf_waiting = OrderedDict()

    def request_tp(self, current_process, size, part_name):
        tp, waiting = False, False
        road_size = math.ceil(max(12, size))
        from_to_matrix = self.network[road_size]
        current_process_inout = self.inout[current_process][0]  # in

        # tp store에 남아있는 transporter가 있는 경우
        if len(self.tp_store.items) > 0:
            tp = yield self.tp_store.get()
            self.monitor.record(self.env.now, None, None, part_id=None, event="tp_going_to_requesting_process",
                                resource=tp.name)
            return tp, waiting
        # transporter가 전부 다른 공정에 가 있을 경우
        else:
            tp_location_list = []
            for name in self.tp_location.keys():
                if not self.tp_location[name]:
                    continue
                tp_current_location = self.tp_location[name][-1]
                if tp_current_location not in self.stocks.keys() and \
                        len(self.processes[tp_current_location].tp_store.items) > 0:  # 해당 공정에 놀고 있는 tp가 있다면
                    tp_location_list.append(self.tp_location[name][-1])
                elif tp_current_location in self.stocks.keys() and len(self.stocks[tp_current_location].tp_store.items) > 0:
                    tp_location_list.append(self.tp_location[name][-1])

            if len(tp_location_list) == 0:  # 유휴 tp가 없는 경우
                waiting = True
                # self.tp_waiting[(part_name, current_process)] = self.env.event()
                # yield self.tp_waiting[current_process]
                return tp, waiting

            else:  # 유휴 tp가 있어 part을 실을 수 있는 경우
                # tp를 호출한 공정 ~ 유휴 tp 사이의 거리 중 가장 짧은 데에서 호출
                distance = []
                tp_location_temp = tp_location_list[:]
                for location in tp_location_list:
                    tp_location = self.inout[location][1]  # out
                    if (tp_location in from_to_matrix.index) and (current_process_inout in list(from_to_matrix)) and \
                            (from_to_matrix[tp_location][current_process_inout] is not None) and (
                            from_to_matrix[tp_location][current_process_inout] < 100000):
                        called_distance = from_to_matrix[tp_location][current_process_inout]
                        distance.append(called_distance)
                    else:
                        tp_location_temp.remove(location)

                if len(distance):
                    location_idx = distance.index(min(distance))
                    location = tp_location_temp[location_idx]
                    if location in self.stocks.keys():
                        tp = yield self.stocks[location].tp_store.get()
                    else:
                        tp = yield self.processes[location].tp_store.get()
                    self.monitor.record(self.env.now, location, None, part_id=None, event="tp_going_to_requesting_process", resource=tp.name)
                    tp_start_time = self.env.now
                    yield self.env.timeout(distance[location_idx]/tp.v_unloaded)  # 현재 위치까지 오는 시간
                    self.monitor.record(self.env.now, current_process, None, part_id=None, event="tp_arrived_at_requesting_process", resource=tp.name)
                    tp_finish_time = self.env.now
                    self.monitor.record_tp_record(tp.name, [tp_start_time, tp_finish_time], distance[location_idx],
                                                  loaded=False)
                    # self.tp_post_processing[tp.name]['unloaded']['moving_time'].append([tp_start_time, tp_finish_time])
                    # self.tp_post_processing[tp.name]['unloaded']['moving_distance'].append(distance[location_idx])
                    # if location == "8도크PE" or current_process_inout == "8도크PE":
                    #     print(0)
                    self.monitor.record_road_used(self.inout[location][1], current_process_inout, road_size, loaded=False)  # 도로 사용 기록
                    temp = tp._asdict()
                    temp['moving_dist'] += distance[location_idx]
                    temp['moving_time'] += distance[location_idx] / tp.v_unloaded
                    tp = transporter(name=temp['name'], v_loaded=temp['v_loaded'],
                                     v_unloaded=temp['v_unloaded'], moving_dist=temp['moving_dist'],
                                     moving_time=temp['moving_time'])
                    return tp, waiting
                else:
                    waiting = True
                    return tp, waiting


class Part:
    def __init__(self, name, env, process_data, processes, monitor, resource=None, from_to_matrix=None, size=0, area=0,
                 child=None, parent=None, stocks=None, Inout=None, convert_to_process=None, dock=None):
        self.name = name
        self.env = env
        columns = pd.MultiIndex.from_product([[i for i in range(8)], ['start_time', 'process_time', 'process', 'work']])
        self.data = pd.Series(process_data, index=columns)
        self.processes = processes
        self.monitor = monitor
        self.resource = resource
        self.from_to_matrix = from_to_matrix
        self.size = 12
        self.area = area
        self.child = child
        self.parent = parent
        self.stocks = stocks
        self.inout = Inout
        self.convert_to_process = convert_to_process
        ## 1, 2, 3 도크에서 작업되면 1야드, 8, 9도크에서 작업되면 2야드
        self.yard = 1 if dock in [1, 2, 3, 4, 5] else 2
        self.dock = dock
        self.source_location = '2야드 대조립공장' if dock in [8, 9] else '대조립 1공장'

        self.in_child = []
        self.part_store = simpy.Store(env, capacity=1)
        self.assembly_idx = False
        if self.child is None:
            self.part_store.put(block(name=self.name, location=self.source_location, step=0, clone=0, area=self.area))
        self.num_clone = 0
        self.moving_distance_w_TP = []
        self.moving_time = []

        self.tp_flag = False
        self.assemble_flag = False if self.child is not None else True
        self.in_stock = False
        self.where_stock = None
        self.finish = False

        self.part_delay = []
        self.resource_delay = env.event()
        self.assemble_delay = {}
        env.process(self._sourcing())

    def _sourcing(self):
        step = 0
        process = None
        tp = None
        waiting = None
        while not self.finish:
            tp = None
            waiting = None
            part = None

            # 이번 step에서 가야 할 공정
            global process_name
            process_name = self._convert_process(step)
            process = self.processes[process_name]
            road_size = math.ceil(max(12, self.size))
            if process.name == 'Sink':
                self.finish = True
                self.tp_flag = False

            work = self.data[(step, 'work')] if process.name != 'Sink' else 'A'

            # transporter 사용 여부
            if ('F' in work) or ('G' in work) or ('H' in work) or ('J' in work) or ('M' in work) or ('K' in work) or \
                    ('L' in work) or ('N' in work):
                if process.name != 'Sink':
                    self.tp_flag = True
                    if (self.inout[process_name][0] not in list(self.from_to_matrix[road_size].index)) or (
                            self.inout[process_name][1] not in list(self.from_to_matrix[road_size].index)):
                        process = self.processes['Sink']
                        self.monitor.record(self.env.now, None, None, part_id=self.name, event='go to Sink',
                                            memo='No road where TP can move')
                        self.tp_flag = False
            # part 호출
            start_time = self.data[(step, 'start_time')]
            if (start_time is not None) and (self.tp_flag == False):
                lag = start_time - self.env.now if start_time - self.env.now > 0 else 0.0
                yield self.env.timeout(lag)
            elif (start_time is not None) and (self.tp_flag == True):
                lag = start_time - self.env.now - 1 if start_time - self.env.now - 1 > 0 else 0.0
                yield self.env.timeout(lag)

            if step == 0:
                part = yield self.part_store.get()
                self.monitor.record(self.env.now, None, None, part_id=self.name, event="part_created")
                print("part created {0} at {1}".format(self.name, self.env.now))

            elif self.in_stock:  # 적치장에 있는 경우 -> 시간에 맞춰서 꺼내줘야 함
                if self.tp_flag:  # tp 사용 하는 경우
                    # tp 호출
                    tp, waiting = yield self.env.process(
                        self.resource.request_tp(self.where_stock, self.size, self.name))
                    # 적치장에서 block 꺼내오기
                    temp = yield self.stocks[self.where_stock].stock_yard.get(lambda x: x[1].name == self.name)
                    self.stocks[self.where_stock].area_used -= self.area
                    print(self.name, 'At _sourcing /', 'Stock {0} out at {1}'.format(self.where_stock, self.env.now), 'area_used = {0} and preemptive_area = {1}'.format(self.stocks[self.where_stock].area_used, self.stocks[self.where_stock].preemptive_area))
                    self.stocks[self.where_stock].preemptive_area -= self.area
                    print(self.name, 'At _sourcing /', 'Stock {0} out at {1}, minus preemptive area'.format(self.where_stock, self.env.now),
                          'area_used = {0}, preemptive_area = {1}'.format(self.stocks[self.where_stock].area_used,
                                                                          self.stocks[self.where_stock].preemptive_area))
                    part = temp[1]
                    temp_part = part._asdict()
                    temp_part['location'] = process.name
                    temp_part['step'] = step
                    part = block(temp_part['name'], temp_part['location'], temp_part['step'], temp_part['clone'], temp_part['area'])

                    self.stocks[self.where_stock].event_area.append(self.stocks[self.where_stock].area_used)
                    self.stocks[self.where_stock].event_time.append(self.env.now)
                    self.monitor.record(self.env.now, self.where_stock, None, part_id=self.name, event="Stock_out",
                                        area=self.stocks[self.where_stock].area_used, process_type="Stock")
                    yield self.env.process(self.tp(part.location, self.where_stock, part, tp, waiting, inout='In'))
                    self.monitor.record(self.env.now, part.location, None, part_id=self.name,
                                        event="Part arrive at start location")

                else:  # tp 사용 안 해도 되는 경우
                    # 하루 전에 TP 호출
                    temp = yield self.stocks[self.where_stock].stock_yard.get(lambda x: x[1].name == self.name)
                    self.stocks[self.where_stock].area_used -= self.area
                    print(self.name, 'Stock {0} out at {1}'.format(self.where_stock, self.env.now),
                          'area_used = {0} and preemptive_area = {1}'.format(self.stocks[self.where_stock].area_used,
                                                                             self.stocks[
                                                                                 self.where_stock].preemptive_area))
                    self.stocks[self.where_stock].preemptive_area -= self.area
                    print(self.name,
                          'Stock {0} out at {1}, minus preemptive area'.format(self.where_stock, self.env.now),
                          'area_used = {0}, preemptive_area = {1}'.format(self.stocks[self.where_stock].area_used,
                                                                          self.stocks[
                                                                              self.where_stock].preemptive_area))

                    part = temp[1]
                    temp_part = part._asdict()
                    temp_part['location'] = process.name
                    temp_part['step'] = step
                    part = block(temp_part['name'], temp_part['location'], temp_part['step'], temp_part['clone'], temp_part['area'])

                    self.stocks[self.where_stock].event_area.append(self.stocks[self.where_stock].area_used)
                    self.stocks[self.where_stock].event_time.append(self.env.now)
                    self.monitor.record(self.env.now, self.where_stock, None, part_id=self.name, event="Stock_out",
                                        process_type="Stock", area=self.stocks[self.where_stock].area_used)

                self.in_stock = False
                self.where_stock = None
            else:  # get part from previous process
                if self.tp_flag:
                    # 적치장에서 block 꺼내오기
                    part = yield self.part_store.get()
                    self.monitor.record(self.env.now, part.location, None, part_id=self.name,
                                        event="Part arrive at start location")
                else:
                    if len(self.part_store.items) > 0:
                        part = yield self.part_store.get()
                    else:
                        self.part_delay.append(self.env.event())
                        yield self.part_delay[0]
                        part = yield self.part_store.get()

            if process.name == 'Sink':
                self.tp_flag = False

            # go to process
            if (self.tp_flag is True) and (process.name != 'Sink'):  # need tp used
                if step == 0:
                    # if now is first step -> request tp and send to next process
                    tp, waiting = yield self.env.process(
                        self.resource.request_tp(self.source_location, self.size, self.name))
                    if tp is None:
                        print(0)
                    yield self.env.process(self.tp(process.name, self.source_location, part, tp, waiting,'In'))
                    step += 1
                elif step and process.name != 'Sink':  # if now is not the first step, need to predict when part hv to move to the next process
                    # if tp is None:
                    #     tp, waiting = yield self.env.process(self.resource.request_tp(part.location, self.size))
                    # # 현재 블록이 적치장에 있으면
                    # if self.in_stock:
                    #     # 다음 공정으로 보내기
                    #     # yield self.env.process(self.tp(process_name, self.where_stock, part, tp, waiting, 'In'))
                    #     step += 1
                    # else:
                        # yield self.env.process(self.tp(process_name, part.location, part, tp, waiting, 'In'))
                    step += 1

            else:  # no need to use tp
                # Sink로 갈 경우 --> 시간 지연이나 거리 계산 없이 바로 Sink로 put
                if process.name == 'Sink':
                    temp = part._asdict()
                    temp['location'] = process.name
                    temp['step'] = step
                    part = block(temp['name'], temp['location'], temp['step'], temp['clone'], temp['area'])
                    process.put(part)
                    self.finish = True

                else:  # Transporter를 사용하지 않는 경우
                    temp = part._asdict()
                    temp['location'] = process.name
                    temp['step'] = step
                    part = block(
                        name=temp['name'], location=temp['location'], step=temp['step'], clone=temp['clone'], area=temp['area'])
                    process.buffer_to_machine.put(part)
                    step += 1

    def return_to_part(self, part):
        # temp = part._asdict()
        # temp['step'] += 1
        # part = block(**temp)
        tp_flag = False
        next_start_time = self.data[(part.step + 1, 'start_time')]
        next_process_name = self._convert_process(part.step + 1)
        work = self.data[(part.step, 'work')] if self.data[(part.step, 'process')] != 'Sink' else 'A'

        # transporter 사용 여부
        if ('F' in work) or ('G' in work) or ('H' in work) or ('J' in work) or ('M' in work) or ('K' in work) or \
                ('L' in work) or ('N' in work):
            tp_flag = True

        if (next_start_time is not None) and (next_start_time <= 100000) and (next_start_time - self.env.now > 1):  # 적치장 가야 함
            next_process = self.inout[next_process_name][0]
            # tp 필요

            if tp_flag:
                size = math.ceil(max(12, self.size))
                stock = self._find_stock(next_process, part, size)
                self.in_stock = True
                self.where_stock = stock

                tp, waiting = yield self.env.process(self.resource.request_tp(part.location, self.size, self.name))
                self.env.process(self.tp(stock, part.location, part, tp, waiting, 'In'))
                # self.stocks[stock].put(part, next_start_time)
                # matrix = self.from_to_matrix[size]
                # self.moving_distance_wo_TP += matrix[part.location][stock]
                self.monitor.record(self.env.now, part.location, None, part_id=self.name, event="go_to_stock")
            # tp 불필요
            else:  ###
                # next_process = self.inout[self.data[(part.step+1, 'process')]][0]
                next_stock = self._find_stock(next_process, part, 12)
                self.stocks[next_stock].put(part, next_start_time)
                self.in_stock = True
                self.where_stock = next_stock
        else:  # 적치장을 갈 필요도 없고, clone이 아닌 경우에만 return
            self.part_store.put(part)
            self.tp_flag = False
            self.where_stock = None
            if len(self.part_delay) > 0:
                self.part_delay.pop(0).succeed()

    def tp(self, next_process, current_process, part, tp, waiting, inout=None):
        if next_process == 'Sink':
           self.processes['Sink'].put(part)

        else:
            self.monitor.record(self.env.now, current_process, None, part_id=self.name, event="tp_request")
            next_process_name = self.inout[next_process][0] if inout == 'In' else self.inout[next_process][1]
            current_process_name = self.inout[current_process][1]  # 현재 공정 -> 다음 공정이므로, 현재 공정은 out 위치 잡아야
            repeat = 0
            while waiting:
                if not waiting:
                    break
                # if waiting is True == All tp is moving == process hv to delay
                else:
                    # if repeat == 2:
                    #     tp = None
                    #     self.processes['Sink'].put(part)
                    #     self.monitor.record(self.env.now, current_process, None, part_id=self.name, event="go to Sink",
                    #                         reason='No TP available from {0} to {1} at part size {2}'.format(current_process, next_process, self.size))
                    #     break
                    self.monitor.record(self.env.now, current_process, None, part_id=self.name,
                                        event="delay_start_cus_no_tp")
                    self.resource.tp_waiting[(self.name, current_process)] = self.env.event()
                    print("waiting event yield, part={0}, process={1}, time={2}".format(self.name, current_process, self.env.now))
                    yield self.resource.tp_waiting[(self.name, current_process)]
                    print("waiting event is ended yielding, part={0}, process={1}, time={2}".format(self.name, current_process,
                                                                                        self.env.now))
                    self.monitor.record(self.env.now, current_process, None, part_id=self.name,
                                        event="delay_finish_cus_yes_tp")
                    # yield self.env.timeout(1)
                    tp, waiting = yield self.env.process(self.resource.request_tp(current_process, self.size, self.name))
                    # repeat += 1
                    # self.monitor.record(self.env.now, current_process, None, part_id=self.name,
                    #                     event="repeat {0}".format(repeat),
                    #                     reason='No TP available from {1} to {2} at part size {0}'.format(self.size, current_process, next_process))
                    print(part.name, part.step, repeat, self.env.now)
                    continue
            if tp:
                road_size = max(12, math.ceil(self.size))
                network = self.from_to_matrix[road_size]
                self.monitor.record(self.env.now, current_process, None, part_id=self.name,
                                    event="tp_going_to_next_process", resource=tp.name)
                if (current_process_name in list(network.index)) and (next_process_name in list(network)) and (
                        network[current_process_name][next_process_name] != None) and (
                        network[current_process_name][next_process_name] < 100000):
                    distance_to_move = network[current_process_name][next_process_name]
                    start_moving = self.env.now
                    yield self.env.timeout(distance_to_move / tp.v_loaded)
                    finish_moving = self.env.now
                    # 블록 이동 시간, 거리 기록
                    self.moving_time.append([start_moving, finish_moving])
                    self.moving_distance_w_TP.append(distance_to_move)
                    self.monitor.record_tp_record(tp.name, [start_moving, finish_moving], distance_to_move,
                                                  loaded=True)
                    # self.resource.tp_post_processing[tp.name]['loaded']['moving_time'].append([start_moving, finish_moving])
                    # self.resource.tp_post_processing[tp.name]['loaded']['moving_distance'].append(distance_to_move)
                    # 도로 사용량 기록
                    self.monitor.record_road_used(current_process_name, next_process_name, road_size)

                    temp_tp = tp._asdict()
                    temp_tp['moving_dist'] += distance_to_move
                    temp_tp['moving_time'] += distance_to_move / tp.v_loaded
                    tp = transporter(name=temp_tp['name'], v_loaded=temp_tp['v_loaded'], v_unloaded=temp_tp['v_unloaded'],
                                     moving_dist=temp_tp['moving_dist'], moving_time=temp_tp['moving_time'])
                    temp_part = part._asdict()
                    temp_part['location'] = next_process
                    part = block(name=temp_part['name'], location=temp_part['location'], step=temp_part['step'],
                                 clone=temp_part['clone'], area=temp_part['area'])
                    self.monitor.record(self.env.now, next_process, None, part_id=self.name,
                                        event="tp_finished_transferred_to_next_process", resource=tp.name)

                    # 적치장이 아닌 공장으로 갈 때
                    if next_process is None:
                        print(0)
                    if next_process not in self.stocks.keys():
                        self.processes[next_process].buffer_to_machine.put(part)
                        self.processes[next_process].tp_store.put(tp)
                    # 적치장으로 갈 때
                    else:
                        self.stocks[next_process].put(part, self.data[(part.step + 1, 'start_time')])
                        self.stocks[next_process].tp_store.put(tp)

                    self.monitor.record(self.env.now, current_process, None, part_id=self.name,
                                        event="part_transferred_to_next_process_with_tp")

                    self.resource.tp_location[tp.name].append(next_process)
                    # 가용한 tp 하나 발생 -> delay 끝내줌
                    if len(self.resource.tp_waiting) > 0:
                        for event_key in list(self.resource.tp_waiting.keys()):
                            if self.resource.tp_waiting[event_key].triggered is False:
                                self.resource.tp_waiting.pop(event_key).succeed()
                                print("waiting event has been succeeded, part={0}, process={1}, time={2}".format(event_key[0],
                                                                                                    event_key[1],
                                                                                                    self.env.now))
                else:
                    print('Impossible cus cannot move block with tp', part.name, 'from = {}, to = {}, road_size = {}'.format(current_process, next_process_name, road_size))

    def _find_stock(self, next_process, part, size):
        stock_list = list(self.stocks.keys())
        stock_list.remove('Stock')
        next_stock = None
        shortest_path = 1e8
        network = self.from_to_matrix[size]
        for idx in range(len(stock_list)):
            if (next_process in list(network.index)) and (stock_list[idx] in list(network)) and (
                    network[next_process][stock_list[idx]] != None) and (
                    network[next_process][stock_list[idx]] < 1e8) and (
                    self.stocks[stock_list[idx]].capacity - self.stocks[stock_list[idx]].preemptive_area - self.area > 0):
                if next_stock is None:   # No stock to compare
                    next_stock = stock_list[idx]
                    shortest_path = network[next_process][stock_list[idx]]
                else:
                    compared_path = network[next_process][stock_list[idx]]
                    if shortest_path > compared_path:  # replaced
                        next_stock = stock_list[idx]
                        shortest_path = compared_path
                    elif shortest_path == compared_path:
                        shortest_stock_len = len(self.stocks[next_stock].stock_yard.items)
                        compared_stock_len = len(self.stocks[stock_list[idx]].stock_yard.items)
                        next_stock = stock_list[idx] if compared_stock_len < shortest_stock_len else next_stock
        # if self.name == "A0001_E82P0" and next_stock == 'E4':
        #     print(0)
        # if self.name == "A0001_E82S0" and next_stock == 'E4':
        #     print(0)
        next_stock = next_stock if next_stock is not None else 'Stock'
        print(self.name, 'At _find_stock /', 'Stock {0} in at {1}'.format(next_stock, self.env.now),
              'area_used = {0} and preemptive_area = {1}'.format(self.stocks[next_stock].area_used,
                                                                 self.stocks[next_stock].preemptive_area))

        self.stocks[next_stock].preemptive_area += self.area
        print(self.name, 'At _find_stock /', 'Stock {0} in at {1}, add preemptive area'.format(next_stock, self.env.now),
              'area_used = {0}, preemptive_area = {1}'.format(self.stocks[next_stock].area_used,
                                                              self.stocks[next_stock].preemptive_area))
        return next_stock

    def _convert_process(self, step):
        present_process = self.data[(step, 'process')]
        if step == 0:
            previous_process = self.source_location
        else:
            previous_process = self.data[(step-1, 'process')]

        # 1:1 대응
        if present_process not in ['선행도장부', '선행의장부', '기장부', '의장1부', '의장3부', '의장2부']:
            return present_process
        # 그냥 process인 경우 + 경우의 수가 여러 개인 경우
        else:
            process_convert_by_dict = self.convert_to_process[present_process]
            distance = []
            pre_choice = process_convert_by_dict[:]
            road_size = math.ceil(max(12, self.size))
            network = self.from_to_matrix[road_size]
            compared_process = self.inout[previous_process][1]
            for process in process_convert_by_dict:
                process_temp = self.inout[process][0]
                if (process_temp in network.index) and (compared_process in list(network)) and \
                        (network[process_temp][compared_process] is not None) and \
                        (network[process_temp][compared_process] < 100000) and \
                        (self.processes[process].area - self.processes[process].preemptive_area - self.area > 0):
                    distance.append(network[process_temp][compared_process])
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

            self.data[(step, 'process')] = process
            self.processes[process].preemptive_area += self.area
            return process


class Sink:
    def __init__(self, env, processes, parts, monitor):
        self.env = env
        self.name = 'Sink'
        self.processes = processes
        self.parts = parts
        self.monitor = monitor

        # self.tp_store = simpy.FilterStore(env)  # transporter가 입고 - 출고 될 store
        self.parts_rec = 0
        self.last_arrival = 0.0
        self.completed_part = []
        self.tp_store = simpy.Store(env)
        self.tp_used = []

    def put(self, part):
        if self.parts[part.name].parent is not None:  # 상위 블록이 있는 경우
            parent_block = self.parts[part.name].parent
            self.parts[parent_block].in_child.append(part.name)
            if (len(self.parts[parent_block].part_store.items) == 0) and (self.parts[parent_block].assembly_idx is False):
                self.parts[parent_block].part_store.put(
                    block(name=parent_block, location=self.parts[parent_block].source_location, step=0, clone=0,
                          area=self.parts[parent_block].area))
                self.parts[parent_block].assembly_idx = True
            if len(self.parts[parent_block].in_child) == len(self.parts[parent_block].child):
                self.parts[parent_block].assemble_flag = True
                if len(self.parts[parent_block].assemble_delay):
                    end_delay = list(self.parts[parent_block].assemble_delay.keys())
                    self.parts[parent_block].assemble_delay[end_delay[0]].succeed()

            self.tp_used.append(self.name)
            self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="go to parent block")
            print("Ready to Assemble, parent block = {0} at {1}".format(parent_block, self.env.now))

        self.parts_rec += 1
        self.last_arrival = self.env.now
        self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="completed")


class Process:
    def __init__(self, env, name, machine_num, processes, parts, monitor, resource=None,
                 process_time=None, capacity=float('inf'), routing_logic='cyclic', priority=None,
                 capa_to_machine=10000, capa_to_process=float('inf'), MTTR=None, MTTF=None,
                 initial_broken_delay=None, delay_time=None, workforce=None, convert_dict=None, area=float('inf'),
                 process_type=None):

        # input data
        self.env = env
        self.name = name
        self.processes = processes
        self.parts = parts
        self.monitor = monitor
        self.resource = resource
        self.capa = capacity
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
        self.area = area
        self.process_type = process_type

        # variable defined in class
        self.in_process = 0
        self.parts_sent = 0
        self.parts_sent_to_machine = 0
        self.machine_idx = 0
        self.len_of_server = []
        self.waiting_machine = OrderedDict()
        self.waiting_pre_process = OrderedDict()
        self.area_used = 0.0
        self.preemptive_area = 0.0
        self.finish_time = 0.0
        self.start_time = 0.0
        self.event_area = []
        self.event_time = []
        self.event_block_num = []


        # buffer and machine
        self.buffer_to_machine = simpy.Store(env, capacity=capa_to_machine)
        self.buffer_to_process = simpy.Store(env, capacity=capa_to_process)
        self.machine = [Machine(env, '{0}_{1}'.format(self.name, i), self.name, self.parts, self.processes, self.resource,
                                process_time=self.process_time[i], priority=self.priority[i],
                                waiting=self.waiting_machine, monitor=monitor, MTTF=self.MTTF[i], MTTR=self.MTTR[i],
                                initial_broken_delay=self.initial_broken_delay[i],
                                workforce=self.workforce[i]) for i in range(self.machine_num)]
        # resource
        self.tp_store = simpy.Store(self.env)
        self.wf_store = simpy.Store(self.env)

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
            self.area_used += self.parts[part.name].area

            self.event_area.append(self.area_used)
            self.event_time.append(self.env.now)

            self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="Process_entered",
                                process_type=self.process_type, area=self.area_used)

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
    def __init__(self, env, name, process_name, parts, processes, resource, process_time, priority, waiting, monitor, MTTF,
                 MTTR, initial_broken_delay, workforce):
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
            while proc_time > 0:
                if self.MTTF is not None:
                    self.env.process(self.break_machine())
                try:
                    self.broken = False
                    ## working start

                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="work_start")

                    self.working_start = self.env.now
                    yield self.env.timeout(proc_time)

                    ## working finish
                    self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                        event="work_finish")
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

            if self.workforce is True:
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="workforce_used_out", resource=wf.name)
                self.resource.wf_store.put(wf)
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="workforce get out the machine", resource=wf.name)
                if (len(self.resource.wf_store.items) > 0) and (len(self.resource.wf_waiting) > 0):
                    self.resource.wf_waiting.popitem(last=False)[1].succeed()  # delay = (part_id, event)
            if self.parts[part.name].assemble_flag is False:
                self.parts[part.name].assemble_delay[self.name] = self.env.event()
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="process delay", memo='All child do not arrive to parent {0}'.format(part.name))
                yield self.parts[part.name].assemble_delay[self.name]
                self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                    event="process delay finish",
                                    memo='All child arrive to parent {0}'.format(part.name))

            # transfer to 'to_process' function
            self.processes[self.process_name].area_used -= self.parts[part.name].area
            self.processes[self.process_name].preemptive_area -= self.parts[part.name].area
            self.processes[self.process_name].event_area.append(self.processes[self.process_name].area_used)
            self.processes[self.process_name].event_time.append(self.env.now)
            self.env.process(self.parts[part.name].return_to_part(part))
            self.processes[self.process_name].in_process -= 1
            self.processes[self.process_name].event_block_num.append(self.processes[self.process_name].in_process)
            self.monitor.record(self.env.now, self.process_name, self.name, part_id=part.name,
                                event="part_transferred_to_out_buffer, step = {0}".format(part.step),
                                area=self.processes[self.process_name].area_used,
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
    def __init__(self, env, name, parts, monitor, capacity=float("inf")):
        self.name = name
        self.env = env
        self.parts = parts
        self.monitor = monitor
        self.capacity = capacity

        self.stock_yard = simpy.FilterStore(env)
        self.tp_store = simpy.Store(env)
        self.area_used = 0.0
        self.preemptive_area = 0.0
        self.event_area = []
        self.event_time = []

    def put(self, part, out_time):
        self.area_used += self.parts[part.name].area
        self.monitor.record(self.env.now, self.name, None, part_id=part.name, event="Stock_in", area=self.area_used,
                            process_type="Stock")
        temp = part._asdict()
        temp['location'] = self.name
        part = block(temp['name'], temp['location'], temp['step'], temp['clone'], temp['area'])
        self.stock_yard.put([out_time, part])
        self.event_area.append(self.area_used)
        self.event_time.append(self.env.now)


class Monitor:
    def __init__(self, result_path, project_name, network):
        self.file_path = result_path
        self.project_name = project_name

        self.time = []
        self.event = []
        self.part_id = []
        self.process = []
        self.subprocess = []
        self.resource = []
        self.memo = []
        self.process_type = []
        self.area = []

        self.tp_post_processing = {}
        self.area_post_processing = {}

        self.created = 0
        self.completed = 0

        self.network = network
        self.num_used_road_unloaded = {}
        self.num_used_road_loaded = {}
        road = list(network.keys())
        self.location_list = list(network[road[0]].index)
        for road_size in network.keys():
            self.num_used_road_unloaded[road_size] = {}
            self.num_used_road_loaded[road_size] = {}
            for finish_location in self.location_list:
                self.num_used_road_unloaded[road_size][finish_location] = {}
                self.num_used_road_loaded[road_size][finish_location] = {}
                for start_location in self.location_list:
                    self.num_used_road_unloaded[road_size][finish_location][start_location] = 0
                    self.num_used_road_loaded[road_size][finish_location][start_location] = 0

    def record(self, time, process, subprocess, part_id=None, event=None, resource=None, memo=None, process_type=None,
               area=None):
        self.time.append(time)
        self.event.append(event)
        self.part_id.append(part_id)
        self.process.append(process)
        self.subprocess.append(subprocess)
        self.resource.append(resource)
        self.memo.append(memo)
        self.process_type.append(process_type)
        self.area.append(area)

        if event == 'part_created':
            self.created += 1
        elif event == 'completed':
            self.completed += 1

    def record_road_used(self, from_matrix, to_matrix, size, loaded=True):
        if loaded:
            self.num_used_road_loaded[size][to_matrix][from_matrix] += 1
        else:
            self.num_used_road_unloaded[size][to_matrix][from_matrix] += 1

    def record_tp_record(self, tp_name, moving_time, moving_distance, loaded=True):
        if tp_name not in self.tp_post_processing.keys():
            self.tp_post_processing[tp_name] = {'loaded': {'moving_time': [], 'moving_distance': []},
                                                'unloaded': {'moving_time': [], 'moving_distance': []}}

        if loaded:
            self.tp_post_processing[tp_name]['loaded']['moving_time'].append(moving_time)
            self.tp_post_processing[tp_name]['loaded']['moving_distance'].append(moving_distance)
        else:
            self.tp_post_processing[tp_name]['unloaded']['moving_time'].append(moving_time)
            self.tp_post_processing[tp_name]['unloaded']['moving_distance'].append(moving_distance)

    def save_information(self):
        # 1. event_tracer
        event_tracer = pd.DataFrame()
        event_tracer['Time'] = self.time
        event_tracer['Part'] = self.part_id
        event_tracer['Process'] = self.process
        event_tracer['SubProcess'] = self.subprocess
        event_tracer['Resource'] = self.resource
        event_tracer['Event'] = self.event
        event_tracer['Process_Type'] = self.process_type
        event_tracer['Area'] = self.area
        event_tracer['Memo'] = self.memo

        path_event_tracer = self.file_path + 'result_{0}.csv'.format(self.project_name)
        event_tracer.to_csv(path_event_tracer, encoding='utf-8-sig')

        # 2. road_used_information
        # unloaded transporter 이용 횟수 저장 경로
        path_road_info = self.file_path + 'road_info.json'
        road_info = {"loaded": self.num_used_road_loaded, "unloaded" : self.num_used_road_unloaded}
        with open(path_road_info, 'w') as f:
            json.dump(road_info, f)
        # save_road_unloaded_path = self.file_path + './road_unloaded'
        # if not os.path.exists(save_road_unloaded_path):
        #     os.makedirs(save_road_unloaded_path)
        # # unloaded transporter 이용 횟수 저장 경로
        # save_road_loaded_path = self.file_path + './road_loaded'
        # if not os.path.exists(save_road_loaded_path):
        #     os.makedirs(save_road_loaded_path)
        #
        # for road_size in self.num_used_road_unloaded.keys():
        #     df_unloaded = pd.DataFrame(self.num_used_road_unloaded[road_size])
        #     df_unloaded.to_excel(save_road_unloaded_path + "/road_unloaded_{0}.xlsx".format(road_size),
        #                          encoding='utf-8-sig')
        #
        #     df_loaded = pd.DataFrame(self.num_used_road_loaded[road_size])
        #     df_loaded.to_excel(save_road_loaded_path + "/road_loaded_{0}.xlsx".format(road_size),
        #                        encoding='utf-8-sig')

        # 3. tp_used_information
        path_tp_info = self.file_path + 'result_tp_{0}.json'.format(self.project_name)
        with open(path_tp_info, 'w') as f:
            json.dump(self.tp_post_processing, f)

        return path_event_tracer, path_tp_info, path_road_info


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
        idx_possible = random.choice(len(self.server_list))  # random index로 초기화 - 모든 서버가 가동중일 때, 서버에 random하게 파트 할당
        for i in range(len(self.server_list)):
            if self.server_list[i].working is False:  # 만약 미가동중인 server가 존재할 경우, 해당 서버에 part 할당
                idx_possible = i
                break
        return idx_possible