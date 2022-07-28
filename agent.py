"""
Agents for interaction simulations
"""
import numpy as np
import math
from scipy.optimize import minimize
from tools.utility import get_central_vertices, kinematic_model
import copy

# simulation setting
dt = 0.12
TRACK_LEN = 6
MAX_DELTA_UT = 1e-4

# weights for calculate interior cost
WEIGHT_DELAY = 0.6
WEIGHT_DEVIATION = 0.8
weight_metric = np.array([WEIGHT_DELAY, WEIGHT_DEVIATION])
weight_metric = weight_metric / weight_metric.sum()

# weight of interior and group cost
WEIGHT_INT = 1
WEIGHT_GRP = 0.4

# parameters of action bounds
MAX_STEERING_ANGLE = math.pi / 6
MAX_ACCELERATION = 3.0

# initial guess on interacting agent's IPV
INITIAL_IPV_GUESS = 0
virtual_agent_IPV_range = np.array([-3, -2, -1, 0, 1, 2, 3]) * math.pi / 8

# likelihood function
sigma = 0.02
sigma2 = 0.4


class Agent:
    def __init__(self, position, velocity, heading, target):
        self.position = position
        self.velocity = velocity
        self.heading = heading
        self.target = target
        self.track_len = TRACK_LEN
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
        self.conl_type = 'gt'
        self.ipv_error = None
        self.ipv_collection = []
        self.ipv_error_collection = []
        self.virtual_track_collection = []

    def solve_optimization(self, inter_track):
        """
        Solve optimization to output best solution given interacting counterpart's track
        """
        track_len = np.size(inter_track, 0)
        self_info = [self.position,
                     self.velocity,
                     self.heading,
                     self.ipv,
                     self.target]

        p, v, h = self_info[0:3]
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]  # initial state
        fun = utility_ibr(self_info, inter_track)  # objective function
        u0 = np.concatenate([1 * np.zeros([(track_len - 1), 1]),
                             np.zeros([(track_len - 1), 1])])  # initialize solution
        bds_acc = [(-MAX_ACCELERATION, MAX_ACCELERATION) for i in range(track_len - 1)]
        bds_str = [(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE) for i in range(track_len - 1)]
        bds = bds_acc + bds_str

        res = minimize(fun, u0, bounds=bds, method='SLSQP')
        x = np.reshape(res.x, [2, track_len - 1]).T
        self.action = x
        self.trj_solution = kinematic_model(x, init_state_4_kine, track_len, dt)  # get trajectory
        return self.trj_solution

    def ibr_interact_with_virtual_agents(self, agent_inter, iter_limit=10):
        """
        Generate copy of the interacting agent and interact with them
        :param iter_limit: max iteration number
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
                agent_self_temp.solve_optimization(virtual_inter_agent.trj_solution)
                virtual_inter_agent.solve_optimization(agent_self_temp.trj_solution)
                if count_iter > iter_limit:
                    break
            virtual_agent_track_collection.append(virtual_inter_agent.trj_solution)
        self.estimated_inter_agent.virtual_track_collection.append(virtual_agent_track_collection)

    def ibr_interact(self, iter_limit=10):
        """
        Interact with the estimated interacting agent. This agent's IPV is continuously updated.
        """
        count_iter = 0  # count number of iteration
        last_self_track = np.zeros_like(self.trj_solution)  # initialize a track reservation
        while np.linalg.norm(self.trj_solution[:, 0:2] - last_self_track[:, 0:2]) > 1e-3:
            count_iter += 1
            last_self_track = self.trj_solution
            self.solve_optimization(self.estimated_inter_agent.trj_solution)
            self.estimated_inter_agent.solve_optimization(self.trj_solution)
            if count_iter > iter_limit:  # limited to less than 10 iterations
                break

    def opt_plan(self):
        self.estimated_inter_agent.solve_optimization(self.trj_solution)
        self.solve_optimization(self.estimated_inter_agent.trj_solution)

    def update_state(self, inter_agent):
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

        if self.conl_type in {'gt'}:
            # update IPV
            current_time = np.size(self.observed_trajectory, 0) - 2
            if current_time > 1:
                start_time = max(0, current_time - 6)
                time_duration = current_time - start_time

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


def utility_ibr(self_info, track_inter):
    def fun(u):
        """
        Calculate the utility from the perspective of "self" agent

        :param u: num_step by 4 array, where the 1\2 columns are control of "self" agent
                  and 3\4 columns control of "interacting" agent
        :return: utility of the "self" agent under the control u
        """
        p, v, h = self_info[0:3]
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]
        track_info_self = kinematic_model(u, init_state_4_kine, np.size(track_inter, 0), dt)
        track_self = track_info_self[:, 0:2]
        track_all = [track_self, track_inter[:, 0:2]]
        # print(np.sin(self_info[3]))
        interior_cost = cal_individual_cost(track_self, self_info[4])
        group_cost = cal_group_cost(track_all)
        util = np.cos(self_info[3]) * interior_cost + np.sin(self_info[3]) * group_cost
        # print('interior_cost:', interior_cost)
        # print('group_cost:', group_cost)
        return util

    return fun


def cal_individual_cost(track, target):
    """
    Cost that related to a single agent
    """

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

    cost_metric = np.array([cost_travel_distance, cost_mean_deviation])

    "overall cost"
    cost_interior = weight_metric.dot(cost_metric.T)

    return cost_interior * WEIGHT_INT


def cal_group_cost(track_packed):
    """
    Cost that related to both two interacting agents
    """
    track_self, track_inter = track_packed
    pos_rel = track_inter - track_self
    dis_rel = np.linalg.norm(pos_rel, axis=1)

    vel_self = (track_self[1:, :] - track_self[0:-1, :]) / dt
    vel_inter = (track_inter[1:, :] - track_inter[0:-1, :]) / dt
    vel_rel = vel_self - vel_inter

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

    # print('group cost:', cost_group)
    return cost_group * WEIGHT_GRP


def cal_reliability(inter_track, act_track, vir_track_coll, target):
    """
    Calculate the reliability of a certain virtual agent by comparing observed and simulated tracks of the under
    estimating agent (or the cost preference thereof)

    Parameters
    ----------
    inter_track: (only used when calculate with cost preference) track of the interacting counterpart
    act_track: observed track of the under estimating agent
    vir_track_coll: collection of the track conducted by simulated virtual agents
    target: task of the under estimating agent (left-turn or go-straight)

    """

    candidates_num = len(vir_track_coll)
    var = np.zeros(candidates_num)
    interior_cost_vir = np.zeros(candidates_num)
    group_cost_vir = np.zeros(candidates_num)
    delta_pref = np.zeros(candidates_num)
    cost_preference_vir = np.zeros(candidates_num)

    if np.size(inter_track) == 0:
        # calculate with trj similarity
        for i in range(candidates_num):
            virtual_track = vir_track_coll[i]
            rel_dis = np.linalg.norm(virtual_track - act_track, axis=1)  # distance vector
            var[i] = np.power(
                np.prod(
                    (1 / sigma / np.sqrt(2 * math.pi))
                    * np.exp(- rel_dis ** 2 / (2 * sigma ** 2))
                )
                , 1 / np.size(act_track, 0))

            if var[i] < 0:
                var[i] = 0

    else:
        # calculate with cost preference similarity
        interior_cost_observed = cal_individual_cost(act_track, target)
        group_cost_observed = cal_group_cost([act_track, inter_track])
        cost_preference_observed = math.atan(group_cost_observed / interior_cost_observed)

        for i in range(candidates_num):
            virtual_track = vir_track_coll[i]
            interior_cost_vir[i] = cal_individual_cost(virtual_track, target)
            group_cost_vir[i] = cal_group_cost([virtual_track, inter_track])
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
