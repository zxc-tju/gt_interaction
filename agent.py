"""
Agents for interaction simulations
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize
from tools.utility import get_central_vertices, kinematic_model, mass_point_model, get_intersection_point, CalcRefLine
from tools.Lattice import TrajPoint
import copy
from concurrent.futures import ProcessPoolExecutor
import viztracer
import time

# simulation setting
dt = 0.12
TRACK_LEN = 10
MAX_DELTA_UT = 1e-4
MIN_DIS = 2

# weights for calculate interior cost
WEIGHT_DELAY = 0.3
WEIGHT_DEVIATION = 0.8
weight_metric = np.array([WEIGHT_DELAY, WEIGHT_DEVIATION])
weight_metric = weight_metric / weight_metric.sum()

# weight of interior and group cost
WEIGHT_INT = 1
WEIGHT_GRP = 0.22
# WEIGHT_GRP = 0.4

# parameters of action bounds
MAX_STEERING_ANGLE = math.pi / 6
MAX_ACCELERATION = 1.0

# initial guess on interacting agent's IPV
INITIAL_IPV_GUESS = 0
virtual_agent_IPV_range = np.array([-3, -2, -1, 0, 1, 2, 3]) * math.pi / 8

# likelihood function
sigma = 0.02
sigma2 = 0.05


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

    def ibr_interact_with_virtual_agents_parallel(self, agent_inter, iter_limit=10):
        """
        Generate copy of the interacting agent and interact with them
        :param iter_limit: max iteration number
        :param agent_inter: Agent:interacting agent
        :return:
        """
        virtual_agent_track_collection = []

        pool = ProcessPoolExecutor(max_workers=8)
        tasks = []
        for i, ipv_temp in enumerate(virtual_agent_IPV_range):
            virtual_inter_agent = copy.deepcopy(agent_inter)
            virtual_inter_agent.ipv = ipv_temp
            agent_self_temp = copy.deepcopy(self)
            tasks.append(pool.submit(game_thread, agent_self_temp, virtual_inter_agent, iter_limit))

        for t in tasks:
            virtual_agent_track_collection.append(t.result())
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

    def linear_ibr_interact(self, iter_limit=10):
        """
        Interact with the estimated interacting agent. This agent's IPV is continuously updated.
        """
        p, v, h = self.position, self.velocity, self.heading
        init_state = [p[0], p[1], v[0], v[1], h]  # initial state
        last_track_self = mass_point_model(np.zeros([TRACK_LEN * 2 - 2]), init_state, TRACK_LEN, dt)
        last_track_self = last_track_self[:, 0:2]

        p, v, h = self.estimated_inter_agent.position, \
                  self.estimated_inter_agent.velocity, \
                  self.estimated_inter_agent.heading
        init_state = [p[0], p[1], v[0], v[1], h]  # initial state
        last_track_inter = mass_point_model(np.zeros([TRACK_LEN * 2 - 2]), init_state, TRACK_LEN, dt)
        last_track_inter = last_track_inter[:, 0:2]

        record_track_inter = np.zeros([TRACK_LEN, 2])

        # plan for interacting agent
        last_track_inter = self.estimated_inter_agent. \
            solve_linear_optimization(last_track_inter[:, 0:2], last_track_self[:, 0:2])

        count_iter = 0  # count number of iteration
        while (np.linalg.norm(record_track_inter - last_track_inter) > 1e-3) \
                and (count_iter < iter_limit):
            count_iter += 1

            # plan for subjective agent
            last_track_self = self.solve_linear_optimization(last_track_self, last_track_inter)

            #  plan for interacting agent
            last_track_inter = self.estimated_inter_agent. \
                solve_linear_optimization(last_track_inter, last_track_self)

        #  final plan for subjective agent
        self.solve_linear_optimization(last_track_self, last_track_inter)
        # self.in_loop_draw(last_track_self, last_track_inter)

    def opt_plan(self):
        self.estimated_inter_agent.solve_optimization(self.trj_solution)
        self.solve_optimization(self.estimated_inter_agent.trj_solution)

    def idm_plan(self, inter_agent):

        # get central vertices
        cv, s_accumu = get_central_vertices(cv_type=self.target, origin_point=self.observed_trajectory[0, 0:2])
        cv_inter, s_accumu_inter = get_central_vertices(cv_type=inter_agent.target,
                                                        origin_point=inter_agent.observed_trajectory[0, 0:2])

        # plt.plot(cv[:,0],cv[:,1])
        # plt.plot(cv_inter[:,0], cv_inter[:,1])
        # plt.show()

        "find the conflict point"
        conflict_point_str = get_intersection_point(cv, cv_inter)
        conflict_point = np.array(conflict_point_str)
        if conflict_point_str.is_empty:  # there is no intersection between given polylines
            min_dis = 99
            min_dis2cv_index_a = None
            min_dis2cv_index_b = 0
            for i in range(np.size(cv, 0)):
                point_b = cv[i, :]
                dis2cv_lt_temp = np.linalg.norm(cv_inter - point_b, axis=1)
                min_dis2cv_temp = np.amin(dis2cv_lt_temp)
                min_dis2cv_index_temp = np.where(min_dis2cv_temp == dis2cv_lt_temp)
                if min_dis2cv_temp < min_dis:
                    min_dis = min_dis2cv_temp
                    min_dis2cv_index_a = min_dis2cv_index_temp[0]
                    min_dis2cv_index_b = i
            conflict_point = (cv_inter[min_dis2cv_index_a[0], :] + cv[min_dis2cv_index_b, :]) / 2

        "index of conflict point on each cv"
        cp_index = get_index_on_cv(cv, conflict_point)
        cp_index_inter = get_index_on_cv(cv_inter, conflict_point)

        "index of current position on cv"
        pos_index = get_index_on_cv(cv, self.position)
        pos_index_inter = get_index_on_cv(cv_inter, inter_agent.position)

        "lateral distance to conflict point"
        dis2cp = s_accumu[cp_index] - s_accumu[pos_index]
        dis2cp = dis2cp[0]
        dis2cp_inter = s_accumu_inter[cp_index_inter] - s_accumu_inter[pos_index_inter]
        dis2cp_inter = dis2cp_inter[0]

        if (dis2cp < 0 or dis2cp_inter < 0) or dis2cp < dis2cp_inter:  # passed the conflict point
            vel_temp = np.linalg.norm(self.velocity) + 0.1
        else:
            idm_param = [1.5, 1.5, 1.5, 2, 30]
            vel = np.linalg.norm(self.velocity)
            vel_rela = vel - np.linalg.norm(inter_agent.velocity)
            dis_rela = dis2cp - dis2cp_inter
            acc = idm_model(idm_param, vel, vel_rela, dis_rela)
            vel_temp = vel + acc * dt

        lat_dis_new = s_accumu[pos_index] + vel_temp * dt
        new_pos2cv = np.abs(s_accumu - lat_dis_new)
        min_dis2cv = np.amin(new_pos2cv)
        new_pos_index = np.where(min_dis2cv == new_pos2cv)
        new_position = cv[new_pos_index, :]
        new_position = new_position[0, 0, :]
        new_heading = - math.atan((cv[new_pos_index[0], 1] - cv[new_pos_index[0] - 1, 1]) /
                                  (cv[new_pos_index[0], 0] - cv[new_pos_index[0] - 1, 0]))
        new_velocity = [vel_temp * math.cos(new_heading), vel_temp * math.sin(new_heading)]
        self.trj_solution = np.array([[self.position[0],
                                       self.position[1],
                                       self.velocity[0],
                                       self.velocity[1],
                                       self.heading],
                                      [new_position[0],
                                       new_position[1],
                                       new_velocity[0],
                                       new_velocity[1],
                                       new_heading]])

    def cruise_plan(self):
        p, v, h = self.position, self.velocity, self.heading
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]  # initial state

        self.trj_solution = np.array([init_state_4_kine, [p[0] + dt * v[0], p[1] + dt * v[1], v[0], v[1], h]])

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

        u0 = np.concatenate([1 * np.zeros([(track_len - 1), 1]),
                             np.zeros([(track_len - 1), 1])])  # initialize solution
        bds_acc = [(-MAX_ACCELERATION, MAX_ACCELERATION) for _ in range(track_len - 1)]
        bds_str = [(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE) for _ in range(track_len - 1)]
        bds = bds_acc + bds_str

        fun = utility_fun(self_info, inter_track)  # objective function
        res = minimize(fun, u0, bounds=bds, method='SLSQP')

        x = np.reshape(res.x, [2, track_len - 1]).T
        self.action = x
        self.trj_solution = kinematic_model(x, init_state_4_kine, track_len, dt)  # get trajectory
        return self.trj_solution

    def solve_linear_optimization(self, last_track_self, last_track_inter):
        """
        Solve optimization to output best solution given interacting counterpart's track
        """

        p, v, h = self.position, self.velocity, self.heading
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]  # initial state

        opt_params = self.get_opt_params(last_track_self, last_track_inter)
        opt_params = [[p, v, h, self.ipv], opt_params]

        # constraints (collision avoidance)
        ineq_cons = {
                'type': 'ineq',
                'fun': lambda u: collision_cons(u, init_state_4_kine, last_track_self, last_track_inter)
            }

        # initial solution
        u0 = 0.1 * np.ones([(2 * TRACK_LEN - 2), 1])

        # boundary
        bds = [(-MAX_ACCELERATION, MAX_ACCELERATION) for _ in range(2 * TRACK_LEN - 2)]

        # objective function
        fun = linear_utility_fun(opt_params, last_track_self, last_track_inter)

        # time1 = time.perf_counter()
        res = minimize(fun, u0, bounds=bds,  constraints=ineq_cons)
        # res = minimize(fun, u0, bounds=bds)
        # time2 = time.perf_counter()
        # print(time2 - time1)

        x = res.x
        self.action = x
        self.trj_solution = mass_point_model(x, init_state_4_kine, TRACK_LEN, dt)  # get trajectory

        # self.in_loop_draw(last_track_self, last_track_inter)

        return self.trj_solution[:, 0:2]

    def get_opt_params(self, last_track_self, last_track_inter):

        # calculate model parameters
        dis_ps2pi = np.linalg.norm(last_track_self - last_track_inter, axis=1)
        miu = np.zeros_like(dis_ps2pi)
        miu[np.where(dis_ps2pi - MIN_DIS < 0.5)] = 1

        if self.target in {'gs_nds', 'lt_nds'}:
            cv, _ = get_central_vertices(self.target, last_track_self[0, :])
        else:
            cv, _ = get_central_vertices(self.target, None)
        path_points = CalcRefLine([cv[:, 0], cv[:, 1]])  # reference path

        # match track point onto the reference path
        matched_points = []
        tao = []
        vec_n = []
        vec_t = []
        kappa = []
        for i, point in enumerate(last_track_self):
            tp = TrajPoint([point[0], point[1], 0, 0, 0, 0])
            matched_point = tp.MatchPath(path_points)
            matched_points.append(matched_point)
            tao.append([matched_point.rx, matched_point.ry])
            vec_t.append([math.cos(matched_point.rtheta), math.sin(matched_point.rtheta)])
            vec_n.append(np.dot(np.array(vec_t[i]), np.array([[0, -1], [1, 0]])))
            kappa.append(matched_point.rkappa)

        tao = np.array(tao)
        vec_t = np.array(vec_t)
        vec_n = np.array(vec_n)

        return [miu, tao, vec_t, vec_n, kappa]

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

    def estimate_self_ipv_in_nds(self, self_actual_track, inter_track):
        self_virtual_track_collection = []
        # ipv_range = np.random.normal(self.ipv, math.pi/6, 6)
        ipv_range = virtual_agent_IPV_range
        for ipv_temp in ipv_range:
            agent_self_temp = copy.deepcopy(self)
            agent_self_temp.ipv = ipv_temp
            # generate track with varied ipv
            virtual_track_temp = agent_self_temp.solve_optimization(inter_track)
            # save track into a collection
            self.virtual_track_collection.append(virtual_track_temp[:, 0:2])

        # calculate reliability of each track
        ipv_weight = cal_reliability([],
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
        plt.figure()
        cv_init_it, _ = get_central_vertices('lt')
        cv_init_gs, _ = get_central_vertices('gs')
        plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b--')
        for t in range(np.size(self.trj_solution, 0)):
            plt.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-')
            plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-')
            plt.plot(self.trj_solution[t, 0], self.trj_solution[t, 1], 'r*')
            plt.plot(self.estimated_inter_agent.trj_solution[t, 0],
                     self.estimated_inter_agent.trj_solution[t, 1], 'b*')

            plt.plot([self.trj_solution[t, 0], self.estimated_inter_agent.trj_solution[t, 0]],
                     [self.trj_solution[t, 1], self.estimated_inter_agent.trj_solution[t, 1]], alpha=0.2, color='black')

            plt.axis('equal')
            plt.xlim(5, 25)
            plt.ylim(-10, 10)
            # plt.pause(0.01)

        plt.title('gs ipv: ' + str(self.estimated_inter_agent.ipv) + '--lt ipv: ' + str(self.ipv))
        plt.show()

    def in_loop_draw(self, last_track_self, last_track_inter):
        plt.figure()
        cv_init_it, _ = get_central_vertices('lt')
        cv_init_gs, _ = get_central_vertices('gs')
        plt.plot(cv_init_it[:, 0], cv_init_it[:, 1], 'r-', alpha=0.2)
        plt.plot(cv_init_gs[:, 0], cv_init_gs[:, 1], 'b-', alpha=0.2)

        plt.scatter(self.trj_solution[:, 0], self.trj_solution[:, 1], color='black')
        plt.plot(last_track_self[:, 0], last_track_self[:, 1], 'red')
        plt.plot(last_track_inter[:, 0], last_track_inter[:, 1], 'blue')
        plt.axis('equal')
        plt.xlim(5, 25)
        plt.ylim(-10, 10)

        plt.show()


def collision_cons(u, init_state_4_kine, last_track_self, last_track_inter):
    min_dis = MIN_DIS

    track_info_self = mass_point_model(u, init_state_4_kine, TRACK_LEN, dt)
    new_track_self = track_info_self[:, 0:2]
    cons = []
    for i in range(TRACK_LEN - 1):
        direction_vector = (last_track_self[i, :] - last_track_inter[i, :]) \
                           / np.linalg.norm(last_track_self[i, :] - last_track_inter[i, :])
        cons.append(np.dot(direction_vector, (new_track_self[i, :]-last_track_inter[i, :]))-min_dis)
    return np.array(cons)


def linear_utility_fun(opt_params, last_track_self, last_track_inter):
    def fun(u):
        """
        Calculate the utility from the perspective of "self" agent

        :param u: num_step by 4 array, where the 1\2 columns are control of "self" agent
                  and 3\4 columns control of "interacting" agent
        :return: utility of the "self" agent under the control u
        """
        self_info, last_track_params = opt_params
        p, v, h = self_info[0:3]
        init_state_4_kine = [p[0], p[1], v[0], v[1], h]
        track_info_self = mass_point_model(u, init_state_4_kine, TRACK_LEN, dt)
        new_track_self = track_info_self[:, 0:2]
        util = cal_linear_util(new_track_self, self_info[3], last_track_params, last_track_self, last_track_inter)
        return util

    return fun


def cal_linear_util(new_track, self_ipv, last_track_params, last_track, last_track_inter):
    """

    Parameters
    ----------
    new_track: self track generated by mass point model given action u
    self_ipv:
    last_track_params:
        miu: lagrange multiplier associated to collision avoidance constraint
        tao: tao vector of on-reference point associated to the self last track
        vec_t: tangent vector of ...
        vec_n: normal vector of ...
        kappa: local curvature
    last_track: self track in last iteration
    last_track_inter: interacting agent's track in last iteration (planned by subjective agent)

    Returns
    -------
    symbolic utility function
    """

    miu, tao, vec_t, vec_n, kappa = last_track_params

    '---- define cost of lane keeping ----'
    # c_dev_cv = 0
    # for k in range(np.size(vec_n, 0)):
    #     dis_new2last = new_track[k, :] - tao[k, :]
    #     c_dev_cv = c_dev_cv + abs(np.dot(vec_n[k, :], dis_new2last[:]))

    c_dev_cv = abs(np.dot(vec_n[-1, :], new_track[-1, :] - tao[-1, :])) \
               + abs(np.dot(vec_n[-2, :], new_track[-2, :] - tao[-2, :]))

    '---- define reward of travel progress ----'
    travel_progress = np.dot(vec_t[-1], (new_track[-1]-last_track[-1])) / \
                      (1 + kappa[-1] * np.dot((tao[-1] - last_track[-1, :]), vec_n[-1, :]))

    '---- interacting agent reward ----'
    r_other = 0
    for k in range(len(miu)):
        if miu[k]:
            r_other = r_other + \
                      miu[k] / np.linalg.norm(last_track[k, :] - last_track_inter[k, :]) \
                      * np.dot((last_track[k, :] - last_track_inter[k, :]), (new_track[k, :]-last_track[k, :]))

    '---- jerk'
    # c_jerk = sum(abs(u[1:]-u[:-1]))

    return math.cos(self_ipv) * (2 * c_dev_cv - travel_progress) - math.sin(self_ipv) * 3 * r_other


def utility_fun(self_info, track_inter):
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


def get_index_on_cv(cv, point):
    cp_on_trj = np.linalg.norm(cv - point, axis=1)
    min_dcp2trj = np.amin(cp_on_trj)
    index = np.where(min_dcp2trj == cp_on_trj)
    return index[0]


def idm_model(para, vel_self, vel_rel, gap):
    akgs = vel_self * para[1] + vel_self * vel_rel / 2 / np.sqrt(para[2] * para[3])
    if akgs < 0:
        sss = para[0]
    else:
        sss = para[0] + akgs

    acc = para[2] * (1 - np.power((vel_self / para[4]), 4) - np.power((sss / gap), 2))
    if acc > 5:
        acc = 5
    if acc < -5:
        acc = -5

    return acc


def game_thread(agent, agent_iter, iter_limit=10):
    count_iter = 0  # count number of iteration
    last_self_track = np.zeros_like(agent.trj_solution)  # initialize a track reservation
    while np.linalg.norm(agent.trj_solution[:, 0:2] - last_self_track[:, 0:2]) > 1e-3:
        count_iter += 1
        last_self_track = agent.trj_solution
        agent.solve_optimization(agent_iter.trj_solution)
        agent_iter.solve_optimization(agent.trj_solution)
        if count_iter > iter_limit:
            break
    return agent.trj_solution


if __name__ == '__main__':
    '---- set initial state of the left-turn vehicle ----'
    init_position_lt = [11.7, -5]
    init_velocity_lt = [1, 2]
    init_heading_lt = math.pi / 4

    '---- set initial state of the go-straight vehicle ----'
    init_position_gs = [19, -2]
    init_velocity_gs = [-3, 0]
    init_heading_gs = math.pi

    agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt')
    agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs')
    agent_lt.estimated_inter_agent = copy.deepcopy(agent_gs)

    agent_lt.ipv = 2 * math.pi / 8
    agent_lt.estimated_inter_agent.ipv = -1 * math.pi / 8

    agent_lt.linear_ibr_interact(iter_limit=5)

    agent_lt.draw()
