"""create agents for simulation"""
import numpy as np
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tools.utility import get_central_vertices, kinematic_model
import copy
from tools.utility import get_intersection_point

'''##========Check Target=======##'''

TARGET = 'simulation'
# TARGET = 'nds simulation'
# TARGET = 'nds analysis'

'''==============================='''

# simulation setting
dt = 0.12
if TARGET in {'nds analysis', 'nds simulation'}:
    dt = 0.12  # stable for nds analysis
elif TARGET == 'simulation':
    dt = 0.15  # stable for simulation
TRACK_LEN = 10
MAX_DELTA_UT = 1e-4
# weights for calculate interior cost
WEIGHT_DELAY = 0.6
if TARGET in {'nds analysis', 'nds simulation'}:
    WEIGHT_DELAY = 0.3
WEIGHT_DEVIATION = 0.8
WEIGHT_OVERSPEED = 0.2
weight_metric = np.array([WEIGHT_DELAY, WEIGHT_DEVIATION, WEIGHT_OVERSPEED])
weight_metric = weight_metric / weight_metric.sum()

# parameters of action bounds
MAX_STEERING_ANGLE = math.pi / 6
MAX_ACCELERATION = 3
# MAX_SPEED = 20
MAX_SPEED = 10

# initial guess on interacting agent's IPV
INITIAL_IPV_GUESS = 0
virtual_agent_IPV_range = np.array([-3, -2, -1, 0, 1, 2, 3]) * math.pi / 8

# weight of interior and group cost
WEIGHT_INT = 1
if TARGET == 'nds analysis':
    WEIGHT_GRP = 0.4  # stable for nds analysis
elif TARGET == 'simulation':
    WEIGHT_GRP = 0.3  # stable for simulation
elif TARGET == 'nds simulation':
    WEIGHT_GRP = 0.4

# likelihood function
sigma = 0.02
sigma2 = 0.4


class Agent:
    def __init__(self, position, velocity, heading, target):
        self.position = position
        self.velocity = velocity
        self.heading = heading
        self.target = target
        # conducted trajectory
        self.observed_trajectory = np.array([[self.position[0],
                                              self.position[1],
                                              self.velocity[0],
                                              self.velocity[1],
                                              self.heading], ])
        self.trj_solution = np.repeat([position], TRACK_LEN, axis=0)
        self.trj_solution_collection = []
        self.action = []
        self.action_collection = []
        self.estimated_inter_agent = None
        self.ipv = 0
        self.ipv_error = None
        self.ipv_collection = []
        self.ipv_error_collection = []
        self.virtual_track_collection = []

    def solve_game_IBR(self, inter_track):
        track_len = np.size(inter_track, 0)
        self_info = [self.position,
                     self.velocity,
                     self.heading,
                     self.ipv,
                     self.target]

        p, v, h = self_info[0:3]
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]  # initial state
        fun = utility_IBR(self_info, inter_track)  # objective function
        u0 = np.concatenate([1 * np.zeros([(track_len - 1), 1]),
                             np.zeros([(track_len - 1), 1])])  # initialize solution
        bds = [(-MAX_ACCELERATION, MAX_ACCELERATION) for i in range(track_len - 1)] + \
              [(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE) for i in range(track_len - 1)]  # boundaries

        res = minimize(fun, u0, bounds=bds, method='SLSQP')
        x = np.reshape(res.x, [2, track_len - 1]).T
        self.action = x
        self.trj_solution = kinematic_model(x, init_state_4_kine, track_len, dt)
        return self.trj_solution

    def interact_with_parallel_virtual_agents(self, agent_inter, iter_limit=10):
        """
        generate copy of the interacting agent and interact with them
        :param iter_limit:
        :param agent_inter: Agent:interacting agent
        :return:
        """
        virtual_agent_track_collection = []

        for ipv_temp in virtual_agent_IPV_range:
            virtual_inter_agent = copy.deepcopy(agent_inter)
            virtual_inter_agent.ipv = ipv_temp
            agent_self_temp = copy.deepcopy(self)

            count_iter = 0  # count number of iteration
            last_self_track = np.zeros_like(self.trj_solution)  # initialize a track reservation
            while np.linalg.norm(agent_self_temp.trj_solution[:, 0:2] - last_self_track[:, 0:2]) > 1e-3:
                count_iter += 1
                last_self_track = agent_self_temp.trj_solution
                agent_self_temp.solve_game_IBR(virtual_inter_agent.trj_solution)
                virtual_inter_agent.solve_game_IBR(agent_self_temp.trj_solution)
                if count_iter > iter_limit:
                    break
            virtual_agent_track_collection.append(virtual_inter_agent.trj_solution)
        self.estimated_inter_agent.virtual_track_collection.append(virtual_agent_track_collection)

    def interact_with_estimated_agents(self, iter_limit=10, controller_type='VGIM'):
        """
        interact with the estimated interacting agent. this agent's IPV is continuously updated.
        :return:
        """
        if controller_type in {'VGIM-coop', 'VGIM-dyna', 'VGIM'}:
            count_iter = 0  # count number of iteration
            last_self_track = np.zeros_like(self.trj_solution)  # initialize a track reservation
            while np.linalg.norm(self.trj_solution[:, 0:2] - last_self_track[:, 0:2]) > 1e-3:
                count_iter += 1
                last_self_track = self.trj_solution
                self.solve_game_IBR(self.estimated_inter_agent.trj_solution)
                self.estimated_inter_agent.solve_game_IBR(self.trj_solution)
                if count_iter > iter_limit:  # limited to less than 10 iterations
                    break
        elif controller_type in {'OPT-coop', 'OPT-dyna', 'OPT-safe'}:
            self.estimated_inter_agent.solve_game_IBR(self.trj_solution)
            self.solve_game_IBR(self.estimated_inter_agent.trj_solution)

    def update_state(self, inter_agent, controller_type='VGIM'):
        self.position = self.trj_solution[1, 0:2]
        self.velocity = self.trj_solution[1, 2:4]
        self.heading = self.trj_solution[1, -1]
        self.estimated_inter_agent.position = inter_agent.trj_solution[1, 0:2]
        self.estimated_inter_agent.velocity = inter_agent.trj_solution[1, 2:4]
        self.estimated_inter_agent.heading = inter_agent.trj_solution[1, -1]

        new_track_point = np.array([[self.position[0],
                                     self.position[1],
                                     self.velocity[0],
                                     self.velocity[1],
                                     self.heading], ])
        self.observed_trajectory = np.concatenate((self.observed_trajectory, new_track_point), axis=0)

        self.trj_solution_collection.append(self.trj_solution)
        self.action_collection.append(self.action)

        if controller_type in {'VGIM-coop', 'VGIM-dyna', 'VGIM'}:
            # update IPV
            current_time = np.size(self.observed_trajectory, 0) - 2
            if current_time > 1:
                start_time = max(0, current_time - 6)
                time_duration = current_time - start_time

                # if method == 1:
                "====parallel game method===="
                candidates = self.estimated_inter_agent.virtual_track_collection[start_time]
                virtual_track_collection = []
                for i in range(len(candidates)):
                    virtual_track_collection.append(candidates[i][0:time_duration, 0:2])
                actual_track = inter_agent.observed_trajectory[start_time:current_time, 0:2]

                ipv_weight = cal_reliability([], actual_track, virtual_track_collection, [])

                # weighted sum of all candidates' IPVs
                self.estimated_inter_agent.ipv = sum(virtual_agent_IPV_range * ipv_weight)

                # save updated ipv and estimation error
                self.estimated_inter_agent.ipv_collection.append(self.estimated_inter_agent.ipv)
                error = 1 - np.sqrt(sum(ipv_weight ** 2))
                self.estimated_inter_agent.ipv_error_collection.append(error)
                "====end of parallel game method===="

            # # modify ipv for 'dyna' models
            # if controller_type in {'VGIM-dyna'} and self.estimated_inter_agent.ipv > math.pi * 3/16:
            #     self.ipv = 0

    def estimate_self_ipv_in_NDS(self, self_actual_track, inter_track):
        self_virtual_track_collection = []
        # ipv_range = np.random.normal(self.ipv, math.pi/6, 6)
        ipv_range = virtual_agent_IPV_range
        for ipv_temp in ipv_range:
            agent_self_temp = copy.deepcopy(self)
            agent_self_temp.ipv = ipv_temp
            # generate track with varied ipv
            virtual_track_temp = agent_self_temp.solve_game_IBR(inter_track)
            # save track into a collection
            self.virtual_track_collection.append(virtual_track_temp[:, 0:2])

        # calculate reliability of each track
        ipv_weight = cal_reliability(inter_track,
                                     self_actual_track,
                                     self.virtual_track_collection,
                                     self.target)

        # weighted sum of all candidates' IPVs
        self.ipv = sum(ipv_range * ipv_weight)
        self.ipv_error = 1 - np.sqrt(sum(ipv_weight ** 2))

        # # save updated ipv and estimation error
        # self.ipv_collection.append(self.ipv)
        # self.ipv_error_collection.append(self.ipv_error)

    def draw(self):
        cv_init_it, _ = get_central_vertices('lt')
        cv_init_gs, _ = get_central_vertices('gs')
        plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b--')
        for t in range(np.size(self.trj_solution, 0)):
            plt.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
            plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')
            plt.plot(self.trj_solution[t, 0], self.trj_solution[t, 1], 'r*')
            plt.plot(self.estimated_inter_agent.trj_solution[t, 0],
                     self.estimated_inter_agent.trj_solution[t, 1], 'b*')
            plt.axis('equal')
            plt.xlim(5, 25)
            plt.ylim(-10, 10)
            plt.pause(0.3)
        plt.show()


def utility_IBR(self_info, track_inter):
    def fun(u):
        """
        Calculate the utility from the perspective of "self" agent

        :param u: num_step by 4 array, where the 1\2 columns are control of "self" agent
                  and 3\4 columns "inter" agent
        :return: utility of the "self" agent under the control u
        """
        p, v, h = self_info[0:3]
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]
        track_info_self = kinematic_model(u, init_state_4_kine, np.size(track_inter, 0), dt)
        track_self = track_info_self[:, 0:2]
        track_all = [track_self, track_inter[:, 0:2]]
        # print(np.sin(self_info[3]))
        interior_cost = cal_interior_cost(track_self, self_info[4])
        group_cost = cal_group_cost(track_all, self_info[4])
        util = np.cos(self_info[3]) * interior_cost + np.sin(self_info[3]) * group_cost
        # print('interior_cost:', interior_cost)
        # print('group_cost:', group_cost)
        return util

    return fun


def cal_interior_cost(track, target):
    if target in {'gs_nds', 'lt_nds'}:
        cv, s = get_central_vertices(target, track[0, :])
    else:
        cv, s = get_central_vertices(target, None)

    # initialize an array to store distance from each point in the track to cv
    dis2cv = np.zeros([np.size(track, 0), 1])
    for i in range(np.size(track, 0)):
        dis2cv[i] = np.amin(np.linalg.norm(cv - track[i, 0:2], axis=1))

    "1. cost of travel delay"
    # calculate the on-reference distance of the given track (the longer the better)
    travel_distance = np.linalg.norm(track[-1, 0:2] - track[0, 0:2]) / np.size(track, 0)
    cost_travel_distance = - travel_distance
    # print('cost of travel delay:', cost_travel_distance)

    "2. cost of lane deviation"
    cost_mean_deviation = max(0.2, dis2cv.mean())
    # print('cost of lane deviation:', cost_mean_deviation)

    "3. cost of overspeed"
    dis = np.linalg.norm(track[1:, :] - track[0:-1, :], axis=1)
    vel = (dis[1:] - dis[0:-1]) / dt
    cost_overspeed = max(max(vel) - MAX_SPEED, 0)

    "4. cost of jerk"
    # dis = np.linalg.norm(track[1:, :] - track[0:-1, :], axis=1)
    # vel = (dis[1:] - dis[0:-1]) / dt
    # acc = (vel[1:] - vel[0:-1]) / dt
    # jerk = (acc[1:] - acc[0:-1]) / dt
    # cost_jerk = max(np.abs(jerk))
    cost_jerk = 0

    cost_metric = np.array([cost_travel_distance, cost_mean_deviation, cost_overspeed])

    "overall cost"
    cost_interior = weight_metric.dot(cost_metric.T)

    return cost_interior * WEIGHT_INT


def cal_group_cost(track_packed, self_target):
    track_self, track_inter = track_packed
    pos_rel = track_inter - track_self
    dis_rel = np.linalg.norm(pos_rel, axis=1)

    vel_self = (track_self[1:, :] - track_self[0:-1, :]) / dt
    vel_inter = (track_inter[1:, :] - track_inter[0:-1, :]) / dt
    vel_rel = vel_self - vel_inter

    acc_self = (vel_self[1:, :] - vel_self[0:-1, :]) / dt
    # acc_inter = (vel_inter[1:, :] - vel_inter[0:-1, :]) / dt
    # acc_rel = acc_self - acc_inter

    "version 1"
    # min_rel_distance = np.amin(rel_distance)  # minimal distance
    # min_index = np.where(min_rel_distance == rel_distance)[0]  # the time step when reach the minimal distance
    # cost_group1 = -min_rel_distance * min_index[0] / (np.size(track_self, 0)) / rel_distance[0]

    cost_group = 0
    if TARGET == 'simulation':
        "version 2: stable for simulation"
        vel_rel_along_sum = 0
        for i in range(np.size(vel_rel, 0)):
            if dis_rel[i + 1] > 3:
                collision_factor = 0.5
            else:
                collision_factor = 1.5
            nearness_temp = collision_factor * pos_rel[i + 1, :].dot(vel_rel[i, :]) / dis_rel[i + 1]
            # do not give reward to negative nearness (flee action)
            vel_rel_along_sum = vel_rel_along_sum + (nearness_temp + np.abs(nearness_temp)) * 0.5
        cost_group = vel_rel_along_sum / TRACK_LEN

    elif TARGET in {'nds analysis', 'nds simulation'}:
        "version 3: stable for nds analysis"
        acc_self_along_sum = 0
        for i in range(np.size(acc_self, 0)):
            nearness_temp = pos_rel[i + 2, :].dot(acc_self[i, :]) / dis_rel[i + 2]
            acc_self_along_sum = acc_self_along_sum + nearness_temp
            # acc_self_along_sum = acc_self_along_sum + (nearness_temp + np.abs(nearness_temp)) * 0.5
        cost_group = acc_self_along_sum / TRACK_LEN / MAX_ACCELERATION  # [-1,1]

    # print('group cost:', cost_group)
    return cost_group * WEIGHT_GRP


def cal_reliability(inter_track, act_trck, vir_trck_coll, target):
    """

    :param target:
    :param inter_track:
    :param act_trck: actual_track
    :param vir_trck_coll: virtual_track_collection
    :return:
    """
    candidates_num = len(vir_trck_coll)
    var = np.zeros(candidates_num)
    interior_cost_vir = np.zeros(candidates_num)
    group_cost_vir = np.zeros(candidates_num)
    delta_pref = np.zeros(candidates_num)
    cost_preference_vir = np.zeros(candidates_num)
    if np.size(inter_track) == 0:
        # calculate with trj similarity
        for i in range(candidates_num):
            virtual_track = vir_trck_coll[i]
            rel_dis = np.linalg.norm(virtual_track - act_trck, axis=1)  # distance vector
            var[i] = np.power(
                np.prod(
                    (1 / sigma / np.sqrt(2 * math.pi))
                    * np.exp(- rel_dis ** 2 / (2 * sigma ** 2))
                )
                , 1 / np.size(act_trck, 0))

            if var[i] < 0:
                var[i] = 0

    else:
        # calculate with cost preference similarity
        interior_cost_observed = cal_interior_cost(act_trck, target)
        group_cost_observed = cal_group_cost([act_trck, inter_track], target)
        cost_preference_observed = math.atan(group_cost_observed / interior_cost_observed)

        for i in range(candidates_num):
            virtual_track = vir_trck_coll[i]
            interior_cost_vir[i] = cal_interior_cost(virtual_track, target)
            group_cost_vir[i] = cal_group_cost([virtual_track, inter_track], target)
            cost_preference_vir[i] = math.atan(group_cost_vir[i] / interior_cost_vir[i])
            delta_pref[i] = cost_preference_vir[i] - cost_preference_observed
            p1 = (1 / sigma2 / np.sqrt(2 * math.pi))
            p2 = np.exp(- delta_pref[i] ** 2 / (2 * sigma2 ** 2))
            var[i] = (1 / sigma2 / np.sqrt(2 * math.pi)) * np.exp(- delta_pref[i] ** 2 / (2 * sigma2 ** 2))

    if sum(var):
        weight = var / (sum(var))
    else:
        weight = np.ones(candidates_num) / candidates_num
    # print(weight)
    return weight


if __name__ == '__main__':
    "test kinematic_model"
    # action = np.concatenate((1 * np.ones((TRACK_LEN, 1)),  0.3 * np.ones((TRACK_LEN, 1))), axis=1)
    # position = np.array([0, 0])
    # velocity = np.array([3, 3])
    # heading = np.array([math.pi/4])
    # init_info = [position, velocity, heading]
    # observed_trajectory = kinematic_model(action, init_info)
    # x = observed_trajectory[:, 0]
    # y = observed_trajectory[:, 1]
    # plt.figure()
    # plt.plot(x, y, 'r-')
    # # plt.plot(observed_trajectory[:, 3], 'r-')
    # plt.axis('equal')
    # plt.show()

    "test get_central_vertices"
    # target = 'gs'
    # cv, s = get_central_vertices(target)
    # x = cv[:, 0]
    # y = cv[:, 1]
    # plt.plot(x, y, 'r-')
    # plt.axis('equal')
    # plt.show()

    "test cal_cost"
    # track_test = [np.array([[0, -15], [5, -13], [10, -10]]), np.array([[20, -2], [10, -2], [0, -2]])]
    # # track_test = [np.array([[0, -15], [5, -13], [10, -10]]), np.array([[0, -14], [5, -11], [10, -9]])]
    # target1 = 'lt'
    # target2 = 'gs'
    # cost_it = cal_group_cost(track_test)
    # print('cost is :', cost_it)

    "test solve game"
    init_position_lt = np.array([13, -7])
    init_velocity_lt = np.array([0, 2])
    init_heading_lt = np.array([math.pi / 3])
    # initial state of the go-straight vehicle
    init_position_gs = np.array([20, -2])
    init_velocity_gs = np.array([-2, 0])
    init_heading_gs = np.array([math.pi])

    # generate LT and virtual GS agents
    agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt')
    agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs')
    agent_lt.estimated_inter_agent = copy.deepcopy(agent_gs)
    agent_gs.estimated_inter_agent = copy.deepcopy(agent_lt)
    agent_lt.ipv = math.pi / 3
    agent_gs.ipv = 0
    # agent_lt.trj_solution_for_inter_agent = np.repeat([init_position_gs], TRACK_LEN + 1, axis=0)
    # agent_gs.trj_solution_for_inter_agent = np.repeat([init_position_lt], TRACK_LEN + 1, axis=0)

    # KKT planning
    # track_s, track_i = agent_lt.solve_game_KKT(agent_gs)

    # IBR planning for left-turn agent
    # track_init = np.repeat([init_position_lt], TRACK_LEN + 1, axis=0)
    track_last = np.zeros_like(agent_lt.trj_solution)
    count = 0
    while np.linalg.norm(agent_lt.trj_solution[:, 0:2] - track_last[:, 0:2]) > 1e-3:
        count += 1
        print(count)
        track_last = agent_lt.trj_solution
        agent_lt.solve_game_IBR(agent_lt.estimated_inter_agent.trj_solution)
        agent_lt.estimated_inter_agent.solve_game_IBR(agent_lt.trj_solution)
        if count > 30:
            break

    cv_init_it, _ = get_central_vertices('lt')
    cv_init_gs, _ = get_central_vertices('gs')
    plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b--')
    for t in range(TRACK_LEN):
        plt.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
        plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')
        plt.plot(agent_lt.trj_solution[t, 0], agent_lt.trj_solution[t, 1], 'r*')
        plt.plot(agent_lt.estimated_inter_agent.trj_solution[t, 0],
                 agent_lt.estimated_inter_agent.trj_solution[t, 1], 'b*')
        plt.axis('equal')
        plt.xlim(5, 25)
        plt.ylim(-10, 10)
        plt.pause(0.3)
    plt.show()
