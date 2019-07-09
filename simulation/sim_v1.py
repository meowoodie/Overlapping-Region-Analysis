#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import math
import arrow
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# helper function for simulation, calculate the distance between two arbitrary
# positions.
def distance(position_a, position_b):
    return math.hypot(
        position_a[0] - position_b[0],
        position_a[1] - position_b[1])

# helper function for generating uniform spatio-temporal points
def simulate_points(
    lam=5,                 # lambda value for poisson distribution
    T=[0, 10],             # time window
    S=[[-1, 1], [-1, 1]]): # spatial region
    '''
    Simulate uniform spatio-temporal points
    '''
    _S     = [T] + S
    # sample the number of events from S
    N      = np.random.poisson(size=1, lam=lam)
    # simulate spatial sequence and temporal sequence separately.
    points = [ np.random.uniform(_S[i][0], _S[i][1], N) for i in range(len(_S)) ]
    points = np.array(points).transpose()
    # sort the sequence regarding the ascending order of the temporal sample.
    points = points[points[:, 0].argsort()]
    return points

class Event(object):
    '''
    Event Object
    '''

    def __init__(self, id, time, location):
        self.id            = id
        self.location      = location
        self.time          = time
        self.dispatch_time = 0
        self.service_time  = 0
        self.end_time      = 0
        self.travel_duration  = 0
        self.waiting_duration = 0
        self.assigned_server  = -1
        # self.proc_time = np.random.exponential(lam, 1)[0]

    def __str__(self):
        return 'Event [%d]: occurred at %f -> dispatch at %f -> end at %f' % (self.id, self.time, self.dispatch_time, self.end_time)

class Server(object):
    '''
    Server Object
    '''

    def __init__(self, id, start_time=0., start_location=[50., 50.], speed=10, mu=.01):
        self.id             = id
        self.start_time     = start_time
        self.start_location = start_location
        self.speed          = speed
        self.mu             = mu
        # history
        self.served_events  = []
        self.idle_times     = []
        # status
        self.cur_time       = start_time
        self.cur_location   = start_location

    def serve(self, event):
        event.assigned_server = self.id
        # server has to wait until the new event occours.
        # otherwise event has to wait until the server finish the previous jobs.
        if self.cur_time < event.time:
            # self.idle_times.append(event.time - self.cur_time)
            self.cur_time = event.time
        # after (event waiting server / server waiting event)
        # server start to serve current event
        event.dispatch_time    = self.cur_time
        event.travel_duration  = distance(event.location, self.cur_location) / self.speed
        event.service_time     = self.cur_time + event.travel_duration
        event.end_time         = event.service_time + np.random.exponential(self.mu, 1)[0]
        event.waiting_duration = event.service_time - event.time
        self.cur_time          = event.end_time
        self.cur_location      = event.location
        self.served_events.append(event.id)

    def __str__(self):
        return 'Server [%d]: total idle time %f, number of idles %d' % \
            (self.id, sum(self.idle_times), len(self.idle_times))

class Simulation(object):
    '''
    Simulation
    '''

    def __init__(self, lam=10, overlapping_ratio=0.,
        T=[0, 10], S=[[-1, 1], [-1, 1]],
        init_server_location=[(-.5, .5), (.5, .5)]): # start location for two servers.
        '''
        The entire region should cover all the area of each of the sub-regions.
        The region is defined by a absolute leftbottom point, its width and its
        height. However, a sub-region is defined by a specific polygon.

        Each server takes charge of one specific sub-region according to their
        ids (index of the list), which means first server in the list serve the
        first sub-region, and so on so forth.
        '''
        # self.n_subr     = len(subregion_polygons)
        # self.subregions = [ Polygon(polygon) for polygon in subregion_polygons ]

        # print('[%s] initializing random events ...' % arrow.now())
        # positions for each of the events
        self.S, self.T = S, T
        self.ratio     = overlapping_ratio
        self.points    = simulate_points(lam, T, S)

        # print('[%s] %d events have been created.' % (arrow.now(), len(self.points)))
        # print('[%s] creating events and servers objects ...' % arrow.now())
        # subregions that each of the events belongs to
        # self.decisions  = [ self._check_event_subr(point[1:]) for point in self.points ]

        # event objects
        self.events  = [
            Event(id=i, time=self.points[i][0], location=self.points[i][1:])
            for i in range(len(self.points)) ]
        # server objects
        self.servers = [
            Server(id=i, start_time=0., start_location=location)
            for i, location in zip(range(len(init_server_location)), init_server_location) ]
        # decisions
        self.decisions = []

    def dispatch_policy(self, event_id):
        left_zone_end    = self.S[0][0] + (self.S[0][1] - self.S[0][0]) * (1 - self.ratio) / 2
        right_zone_start = self.S[0][1] - (self.S[0][1] - self.S[0][0]) * (1 - self.ratio) / 2
        if self.events[event_id].location[1] <= left_zone_end:     # event is at left zone
            return self.servers[0]
        elif self.events[event_id].location[1] > right_zone_start: # event is at right zone
            return self.servers[1]
        else:                                                   # event is at the overlapping zone
            # min_q_idx = np.array([ len(server.queue) for server in self.servers ]).argmin()
            # min_q_idx = np.random.randint(2)
            server_0_q = [ i
                for i in range(0, event_id) 
                if self.events[i].assigned_server == 0 and self.events[i].end_time > self.events[event_id].time ]
            server_1_q = [ i
                for i in range(0, event_id) 
                if self.events[i].assigned_server == 1 and self.events[i].end_time > self.events[event_id].time ]
            min_q_idx = 0 if len(server_0_q) <= len(server_1_q) else 1
            # print(len(server_0_q), len(server_1_q))
            return self.servers[min_q_idx]

    def start_service(self):
        '''
        Start Service

        Each event in turn will be assigned to corresponding server by a given
        policy. In general, the event tends to look for the nearest idle server
        to complete the job. Once the assignment was established, the server will
        move to the location of the event, and start service, then move the next
        assigned event after the completion of current job, so on so forth.
        '''
        # print('[%s] start service simulation ...' % arrow.now())
        for event_id in range(len(self.events)):
            # dispatch server to event according to the policy
            server = self.dispatch_policy(event_id)
            server.serve(self.events[event_id])

        # for event in self.events:
        #     if event.assigned_server == 0:
        #         print(event)

    def get_avg_waiting_time(self):
        # get average waiting time of events in terms of each server.
        avg_waiting_times = []
        for server in self.servers:
            served_events    = [ self.events[event_id] for event_id in server.served_events ]
            waiting_times    = [ event.waiting_duration for event in served_events ]
            avg_waiting_time = np.array(waiting_times).mean()
            avg_waiting_times.append(avg_waiting_time)
        return avg_waiting_times

    def get_max_waiting_time(self):
        # get average waiting time of events in terms of each server.
        max_waiting_times = []
        for server in self.servers:
            served_events    = [ self.events[event_id] for event_id in server.served_events ]
            waiting_times    = [ event.waiting_duration for event in served_events ]
            max_waiting_time = np.array(waiting_times).max() if len(waiting_times) > 0 else 0
            max_waiting_times.append(max_waiting_time)
        return max_waiting_times

    # def get_waiting_time_greater_than(self, threshold=3000.):
    #     # get average waiting time of events in terms of each server.
    #     greater_counts = []
    #     for server in self.servers:
    #         served_events  = [ self.events[event_id] for event_id in server.served_events ]
    #         greater_count  = [ float(event.waiting_duration > threshold) for event in served_events ]
    #         greater_count  = sum(greater_count) if len(greater_count) > 0 else 0
    #         greater_counts.append(greater_count)
    #     return greater_counts

    # def print_service_history(self):
    #     # print the service history for each of the servers.
    #     for server in self.servers:
    #         served_events = [ self.events[event_id] for event_id in server.served_events ]
    #         print('Server [%d]' % server.id)
    #         for event in served_events:
    #             print('Event [%d] occurred at %s (in sub-regions %s), took %f to process, finished at %f, have been waiting for %f.' %\
    #                   (event.id, event.position, event.subr, event.proc_time, event.service_time, event.waiting_duration))

    def _check_event_subr(self, position):
        # return the sub-region id of each event belongs to.
        subr = [ id
            for id, subregion in zip(range(self.n_subr), self.subregions)
            if subregion.contains(Point(position))]
        return subr



if __name__ == '__main__':
    np.random.seed(1)

    lam       = 100
    n_epoches = 100

    overlap_ratio_list    = np.linspace(0., 99., 100) / 100
    avg_waiting_time_list = []
    max_waiting_time_list = []
    # greater_count_list    = []
    for epoch in range(n_epoches):
        print('[%s] Simulation epoch %d' % (arrow.now(), epoch))
        avg_waiting_times = []
        max_waiting_times = []
        # greater_counts    = []
        for overlap_ratio in overlap_ratio_list:

            sim = Simulation(
                lam=lam, overlapping_ratio=overlap_ratio,
                T=[0, 10], S=[[-1, 1], [-1, 1]],
                init_server_location=[(-.5, .5), (.5, .5)])
            sim.start_service()

            avg_waiting_time = np.mean(sim.get_avg_waiting_time())
            # max_waiting_time = np.max(sim.get_max_waiting_time())

            avg_waiting_times.append(avg_waiting_time)
            # max_waiting_times.append(max_waiting_time)
    
        avg_waiting_time_list.append(avg_waiting_times)
        # max_waiting_time_list.append(max_waiting_times)


    avg_waiting_time_list = np.array(avg_waiting_time_list).mean(axis=0)
    # max_waiting_time_list = np.array(max_waiting_time_list).mean(axis=0)
    plt.plot(overlap_ratio_list, avg_waiting_time_list)
    plt.ylabel('average waiting time')
    plt.xlabel('overlap ratio')
    plt.show()

    # # plt.plot(overlap_ratio_list, avg_waiting_time_list)
    # plt.plot(overlap_ratio_list, max_waiting_time_list)
    # plt.ylabel('maximum waiting time')
    # plt.xlabel('overlap ratio')
    # plt.show()
