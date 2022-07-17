import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
import math
from shapely.geometry import LineString
import matplotlib.patches as patches
import matplotlib.transforms as mt


def smooth_ployline(cv_init, point_num=1000):
    cv = cv_init
    list_x = cv[:, 0]
    list_y = cv[:, 1]
    if type(cv) is not np.ndarray:
        cv = np.array(cv)
    delta_cv = cv[1:, ] - cv[:-1, ]
    s_cv = np.linalg.norm(delta_cv, axis=1)

    s_cv = np.array([0] + list(s_cv))
    s_cv = np.cumsum(s_cv)

    bspl_x = splrep(s_cv, list_x, s=0.1)
    bspl_y = splrep(s_cv, list_y, s=0.1)
    # values for the x axis
    s_smooth = np.linspace(0, max(s_cv), point_num)
    # get y values from interpolated curve
    x_smooth = splev(s_smooth, bspl_x)
    y_smooth = splev(s_smooth, bspl_y)
    new_cv = np.array([x_smooth, y_smooth]).T

    delta_new_cv = new_cv[1:, ] - new_cv[:-1, ]
    s_accumulated = np.cumsum(np.linalg.norm(delta_new_cv, axis=1))
    s_accumulated = np.concatenate(([0], s_accumulated), axis=0)
    return new_cv, s_accumulated


def get_central_vertices(cv_type, origin_point=None):
    cv_init = None
    if cv_type == 'lt':  # left turn
        cv_init = np.array([[0, -10], [9, -7.5], [12, -5.2], [13.5, 0], [14, 10], [14, 20], [14, 30]])
    elif cv_type == 'gs':  # go straight
        cv_init = np.array([[20, -2], [10, -2], [0, -2], [-150, -2]])
    elif cv_type == 'lt_nds':  # left turn in NDS
        cv_init = np.array([origin_point, [34.9, 16.6], [45.2, 18.8], [51.6, 20.3]])
    elif cv_type == 'gs_nds':  # go straight in NDS
        cv_init = np.array([origin_point, [23, 45.77], [23.55, 49.654], [24.58, 56.65]])
    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_ployline(cv_init)
    return cv_smoothed, s_accumulated


def kinematic_model(u, init_state, TRACK_LEN, dt):
    if not np.size(u, 0) == TRACK_LEN - 1:
        u = np.array([u[0:TRACK_LEN - 1], u[TRACK_LEN - 1:]]).T
    r_len = 0.8
    f_len = 1
    x, y, vx, vy, h = init_state
    # track = [init_state]
    psi = h
    track = [[x, y, vx, vy, h]]
    v_temp = np.sqrt(vx ** 2 + vy ** 2)

    for i in range(len(u)):
        a = u[i][0]
        delta = u[i][1]
        beta = math.atan((r_len / (r_len + f_len)) * math.tan(delta))
        x = x + v_temp * np.cos(psi + beta) * dt
        y = y + v_temp * np.sin(psi + beta) * dt
        psi = psi + (v_temp / f_len) * np.sin(beta) * dt
        v_temp = v_temp + a * dt

        vx = v_temp * np.cos(psi)
        vy = v_temp * np.sin(psi)

        track.append([x, y, vx, vy, psi])
    return np.array(track)


def draw_rectangle(x, y, deg, ax, para_alpha=0.5, para_color='blue'):
    car_len = 2
    car_wid = 4
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # r1 = patches.Rectangle((x - car_wid / 2, y - car_len / 2), car_wid, car_len, color="blue", alpha=transp)
    r2 = patches.Rectangle((x - car_wid / 2, y - car_len / 2), car_wid, car_len, color=para_color, alpha=para_alpha)

    t2 = mt.Affine2D().rotate_deg_around(x, y, deg) + ax.transData
    r2.set_transform(t2)

    # ax.add_patch(r1)
    ax.add_patch(r2)

    # plt.grid(True)
    # plt.axis('equal')
    #
    # plt.show()