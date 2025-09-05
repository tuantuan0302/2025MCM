import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib后端，避免PyCharm中的兼容性问题
import matplotlib

matplotlib.use('Agg')  # 使用TkAgg后端，这在大多数环境中都可用

# ============================ 常量定义 ============================
G = 9.8  # 重力加速度(m/s^2)
SMOKE_SINK_SPEED = 3  # 烟幕云团下沉速度(m/s)
EFFECTIVE_RADIUS = 10  # 有效遮蔽半径(m)
EFFECTIVE_DURATION = 20  # 有效遮蔽持续时间(s)
MISSILE_SPEED = 300  # 导弹速度(m/s)
MIN_RELEASE_INTERVAL = 1  # 烟幕弹最小投放间隔(s)

# 目标位置
FAKE_TARGET = np.array([0, 0, 0])  # 假目标位置(原点)
REAL_TARGET = np.array([0, 200, 0])  # 真实目标下底面圆心
TARGET_RADIUS = 7  # 目标半径(m)
TARGET_HEIGHT = 10  # 目标高度(m)

# 初始位置
MISSILES = {
    'M1': np.array([20000, 0, 2000]),
    'M2': np.array([19000, 600, 2100]),
    'M3': np.array([18000, -600, 1900])
}

UAVS = {
    'FY1': np.array([17800, 0, 1800]),
    'FY2': np.array([12000, 1400, 1400]),
    'FY3': np.array([6000, -3000, 700]),
    'FY4': np.array([11000, 2000, 1800]),
    'FY5': np.array([13000, -2000, 1300])
}


# ============================ 基础计算函数 ============================
def calculate_missile_direction(missile_pos):
    """计算导弹飞行方向向量（指向假目标）"""
    direction = FAKE_TARGET - missile_pos
    return direction / np.linalg.norm(direction)


def calculate_missile_position(missile_pos, direction, t):
    """计算导弹在时间t的位置"""
    return missile_pos + direction * MISSILE_SPEED * t


def calculate_smoke_trajectory(release_pos, release_vel, t):
    """计算烟幕干扰弹在时间t的位置（考虑重力）"""
    x = release_pos[0] + release_vel[0] * t
    y = release_pos[1] + release_vel[1] * t
    z = release_pos[2] + release_vel[2] * t - 0.5 * G * t ** 2
    return np.array([x, y, z])


def calculate_smoke_cloud_position(detonation_pos, t_after_detonation):
    """计算烟幕云团在起爆后时间t的位置"""
    return np.array([
        detonation_pos[0],
        detonation_pos[1],
        max(0, detonation_pos[2] - SMOKE_SINK_SPEED * t_after_detonation)  # 确保不低于地面
    ])


def distance_between(point1, point2):
    """计算两点之间的距离"""
    return np.linalg.norm(point1 - point2)


def is_effective(smoke_pos, missile_pos):
    """判断烟幕是否有效遮蔽导弹"""
    return distance_between(smoke_pos, missile_pos) <= EFFECTIVE_RADIUS


def calculate_effective_period(detonation_pos, missile_pos, missile_direction, detonation_time):
    """
    计算有效遮蔽时间段

    参数:
    detonation_pos: 烟幕起爆点位置
    missile_pos: 起爆时刻导弹位置
    missile_direction: 导弹飞行方向
    detonation_time: 起爆时刻(从发现导弹开始计时)

    返回:
    (start_time, end_time): 有效遮蔽的开始和结束时间(相对于起爆时刻)
    """
    effective_periods = []
    time_step = 0.1  # 时间步长(s)
    current_effective = False
    start_time = None

    for t in np.arange(0, EFFECTIVE_DURATION + time_step, time_step):
        # 计算烟幕云团位置
        smoke_pos = calculate_smoke_cloud_position(detonation_pos, t)

        # 计算导弹位置
        missile_current_pos = missile_pos + missile_direction * MISSILE_SPEED * t

        # 判断是否有效
        effective = is_effective(smoke_pos, missile_current_pos)

        if effective and not current_effective:
            # 开始有效
            start_time = t
            current_effective = True
        elif not effective and current_effective:
            # 结束有效
            effective_periods.append((start_time, t))
            current_effective = False
            start_time = None

    # 如果最后仍然有效
    if current_effective and start_time is not None:
        effective_periods.append((start_time, EFFECTIVE_DURATION))

    return effective_periods


def calculate_total_effective_time(effective_periods):
    """计算总有效遮蔽时间"""
    total_time = 0
    for start, end in effective_periods:
        total_time += (end - start)
    return min(total_time, EFFECTIVE_DURATION)


# ============================ 问题1解答 ============================
def problem1():
    """解决问题1：计算给定参数下的有效遮蔽时间"""
    print("=" * 50)
    print("问题1解答:")
    print("=" * 50)

    # FY1初始位置和速度
    uav_pos = UAVS['FY1'].copy()
    uav_speed = 120  # m/s

    # 计算FY1飞行方向（朝向假目标）
    direction_to_target = FAKE_TARGET - uav_pos
    uav_direction = direction_to_target / np.linalg.norm(direction_to_target)
    uav_velocity = uav_direction * uav_speed

    # 受领任务1.5s后投放
    release_time = 1.5  # s
    release_pos = uav_pos + uav_velocity * release_time

    # 间隔3.6s后起爆
    detonation_delay = 3.6  # s
    detonation_time = release_time + detonation_delay

    # 计算烟幕干扰弹起爆点位置
    detonation_pos = calculate_smoke_trajectory(release_pos, uav_velocity, detonation_delay)

    # 计算导弹M1的飞行方向
    missile_direction = calculate_missile_direction(MISSILES['M1'])

    # 计算起爆时刻导弹位置
    missile_pos_at_detonation = calculate_missile_position(MISSILES['M1'], missile_direction, detonation_time)

    # 计算有效遮蔽时间段
    effective_periods = calculate_effective_period(
        detonation_pos, missile_pos_at_detonation, missile_direction, detonation_time
    )

    # 计算总有效遮蔽时间
    effective_time = calculate_total_effective_time(effective_periods)

    print(f"无人机FY1初始位置: ({uav_pos[0]:.2f}, {uav_pos[1]:.2f}, {uav_pos[2]:.2f}) m")
    print(f"无人机飞行速度: {uav_speed:.2f} m/s")
    print(f"无人机飞行方向: ({uav_direction[0]:.4f}, {uav_direction[1]:.4f}, {uav_direction[2]:.4f})")
    print(f"烟幕干扰弹投放时间: {release_time:.2f} s")
    print(f"烟幕干扰弹投放点: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f}) m")
    print(f"烟幕干扰弹起爆延迟: {detonation_delay:.2f} s")
    print(f"烟幕干扰弹起爆点: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f}) m")
    print(
        f"起爆时刻导弹位置: ({missile_pos_at_detonation[0]:.2f}, {missile_pos_at_detonation[1]:.2f}, {missile_pos_at_detonation[2]:.2f}) m")
    print(f"有效遮蔽时间段: {effective_periods}")
    print(f"有效遮蔽时长: {effective_time:.2f} s")

    # 可视化
    visualize_problem1(uav_pos, uav_velocity, release_pos, detonation_pos,
                       MISSILES['M1'], missile_direction, detonation_time, effective_periods)

    return effective_time


def visualize_problem1(uav_pos, uav_velocity, release_pos, detonation_pos,
                       missile_pos, missile_direction, detonation_time, effective_periods):
    """可视化问题1的仿真结果"""
    fig = plt.figure(figsize=(15, 10))

    # 3D轨迹图
    ax1 = fig.add_subplot(121, projection='3d')

    # 绘制无人机轨迹
    flight_duration = 20  # 仿真时长(s)
    t_values = np.linspace(0, flight_duration, 100)
    uav_traj = np.array([uav_pos + uav_velocity * t for t in t_values])
    ax1.plot(uav_traj[:, 0], uav_traj[:, 1], uav_traj[:, 2], 'b-', label='UAV FY1 Trajectory')
    ax1.plot([uav_pos[0]], [uav_pos[1]], [uav_pos[2]], 'bo', markersize=8, label='UAV Initial Position')
    ax1.plot([release_pos[0]], [release_pos[1]], [release_pos[2]], 'go', markersize=8, label='Smoke Release Point')

    # 绘制烟幕弹轨迹
    smoke_traj = np.array([calculate_smoke_trajectory(release_pos, uav_velocity, t) for t in np.linspace(0, 5, 50)])
    ax1.plot(smoke_traj[:, 0], smoke_traj[:, 1], smoke_traj[:, 2], 'g--', label='Smoke Trajectory')
    ax1.plot([detonation_pos[0]], [detonation_pos[1]], [detonation_pos[2]], 'ro', markersize=8,
             label='Detonation Point')

    # 绘制烟幕云团下沉轨迹
    if effective_periods:
        max_t = max([end for start, end in effective_periods])
        smoke_cloud_traj = np.array(
            [calculate_smoke_cloud_position(detonation_pos, t) for t in np.linspace(0, max_t, 50)])
        ax1.plot(smoke_cloud_traj[:, 0], smoke_cloud_traj[:, 1], smoke_cloud_traj[:, 2], 'r--', alpha=0.5,
                 label='Smoke Cloud Sinking')

    # 绘制导弹轨迹
    missile_traj = np.array(
        [calculate_missile_position(missile_pos, missile_direction, t) for t in np.linspace(0, flight_duration, 100)])
    ax1.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'm-', label='Missile M1 Trajectory')

    # 标记目标
    ax1.plot([FAKE_TARGET[0]], [FAKE_TARGET[1]], [FAKE_TARGET[2]], 'kx', markersize=10, label='Fake Target')
    ax1.plot([REAL_TARGET[0]], [REAL_TARGET[1]], [REAL_TARGET[2]], 'k*', markersize=10, label='Real Target')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectories')
    ax1.legend()

    # 2D平面图 (X-Z平面)
    ax2 = fig.add_subplot(122)

    # 绘制轨迹投影
    ax2.plot(uav_traj[:, 0], uav_traj[:, 2], 'b-', label='UAV FY1')
    ax2.plot([uav_pos[0]], [uav_pos[2]], 'bo', markersize=8)
    ax2.plot([release_pos[0]], [release_pos[2]], 'go', markersize=8)

    ax2.plot(smoke_traj[:, 0], smoke_traj[:, 2], 'g--', label='Smoke')
    ax2.plot([detonation_pos[0]], [detonation_pos[2]], 'ro', markersize=8)

    if effective_periods:
        ax2.plot(smoke_cloud_traj[:, 0], smoke_cloud_traj[:, 2], 'r--', alpha=0.5, label='Smoke Cloud')

    ax2.plot(missile_traj[:, 0], missile_traj[:, 2], 'm-', label='Missile M1')

    # 标记目标
    ax2.plot([FAKE_TARGET[0]], [FAKE_TARGET[2]], 'kx', markersize=10, label='Fake Target')
    ax2.plot([REAL_TARGET[0]], [REAL_TARGET[2]], 'k*', markersize=10, label='Real Target')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('X-Z Plane Projection')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('problem1_visualization.png', dpi=300)
    plt.close(fig)  # 关闭图形，避免显示问题


# ============================ 问题2解答 ============================
def objective_function_problem2(params, uav_pos, missile_pos, missile_direction):
    """问题2的优化目标函数：最大化遮蔽时间"""
    # 解析参数: [速度, 投放时间, 起爆延迟]
    uav_speed = params[0]
    release_time = params[1]
    detonation_delay = params[2]

    # 约束检查
    if not (70 <= uav_speed <= 140):
        return 1000  # 惩罚值
    if release_time < 0:
        return 1000
    if detonation_delay < 0:
        return 1000

    # 计算无人机飞行方向（朝向假目标）
    direction_to_target = FAKE_TARGET - uav_pos
    uav_direction = direction_to_target / np.linalg.norm(direction_to_target)
    uav_velocity = uav_direction * uav_speed

    # 计算投放点
    release_pos = uav_pos + uav_velocity * release_time

    # 计算起爆点
    detonation_pos = calculate_smoke_trajectory(release_pos, uav_velocity, detonation_delay)

    # 计算起爆时刻导弹位置
    detonation_time = release_time + detonation_delay
    missile_pos_at_detonation = calculate_missile_position(missile_pos, missile_direction, detonation_time)

    # 计算有效遮蔽时间段
    effective_periods = calculate_effective_period(
        detonation_pos, missile_pos_at_detonation, missile_direction, detonation_time
    )

    # 计算总有效遮蔽时间（取负值因为我们要最小化函数）
    effective_time = calculate_total_effective_time(effective_periods)

    return -effective_time  # 返回负值以便最小化函数


def problem2():
    """解决问题2：优化单无人机单烟幕弹对M1的干扰策略"""
    print("=" * 50)
    print("问题2解答:")
    print("=" * 50)

    uav_pos = UAVS['FY1'].copy()
    missile_pos = MISSILES['M1'].copy()
    missile_direction = calculate_missile_direction(missile_pos)

    # 定义参数范围和初始值
    bounds = [(70, 140), (0, 10), (0, 10)]  # 速度范围70-140m/s，投放时间和起爆延迟范围0-10s
    initial_guess = [120, 1.5, 3.6]  # 初始猜测值

    # 使用差分进化算法进行全局优化
    result = differential_evolution(
        objective_function_problem2, bounds,
        args=(uav_pos, missile_pos, missile_direction),
        strategy='best1bin', maxiter=100, popsize=15, tol=0.01,
        recombination=0.7, mutation=(0.5, 1), seed=42
    )

    # 提取优化结果
    opt_speed = result.x[0]
    opt_release_time = result.x[1]
    opt_detonation_delay = result.x[2]

    # 计算无人机飞行方向
    direction_to_target = FAKE_TARGET - uav_pos
    uav_direction = direction_to_target / np.linalg.norm(direction_to_target)
    uav_velocity = uav_direction * opt_speed

    # 计算投放点和起爆点
    release_pos = uav_pos + uav_velocity * opt_release_time
    detonation_pos = calculate_smoke_trajectory(release_pos, uav_velocity, opt_detonation_delay)

    # 计算有效遮蔽时间
    detonation_time = opt_release_time + opt_detonation_delay
    missile_pos_at_detonation = calculate_missile_position(missile_pos, missile_direction, detonation_time)
    effective_periods = calculate_effective_period(
        detonation_pos, missile_pos_at_detonation, missile_direction, detonation_time
    )
    effective_time = calculate_total_effective_time(effective_periods)

    print(f"优化结果:")
    print(f"无人机飞行速度: {opt_speed:.2f} m/s")
    print(f"无人机飞行方向: ({uav_direction[0]:.4f}, {uav_direction[1]:.4f}, {uav_direction[2]:.4f})")
    print(f"烟幕干扰弹投放时间: {opt_release_time:.2f} s")
    print(f"烟幕干扰弹投放点: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f}) m")
    print(f"烟幕干扰弹起爆延迟: {opt_detonation_delay:.2f} s")
    print(f"烟幕干扰弹起爆点: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f}) m")
    print(
        f"起爆时刻导弹位置: ({missile_pos_at_detonation[0]:.2f}, {missile_pos_at_detonation[1]:.2f}, {missile_pos_at_detonation[2]:.2f}) m")
    print(f"有效遮蔽时间段: {effective_periods}")
    print(f"有效遮蔽时长: {effective_time:.2f} s")

    # 保存结果
    result_df = pd.DataFrame({
        'UAV': ['FY1'],
        'Speed': [opt_speed],
        'Release_Time': [opt_release_time],
        'Detonation_Delay': [opt_detonation_delay],
        'Effective_Time': [effective_time]
    })
    result_df.to_excel('problem2_result.xlsx', index=False)

    # 可视化
    visualize_problem2(uav_pos, uav_velocity, release_pos, detonation_pos,
                       missile_pos, missile_direction, detonation_time, effective_periods)

    return {
        'speed': opt_speed,
        'direction': uav_direction,
        'release_time': opt_release_time,
        'release_pos': release_pos,
        'detonation_delay': opt_detonation_delay,
        'detonation_pos': detonation_pos,
        'effective_time': effective_time
    }


def visualize_problem2(uav_pos, uav_velocity, release_pos, detonation_pos,
                       missile_pos, missile_direction, detonation_time, effective_periods):
    """可视化问题2的优化结果"""
    fig = plt.figure(figsize=(15, 10))

    # 3D轨迹图
    ax1 = fig.add_subplot(121, projection='3d')

    # 绘制无人机轨迹
    flight_duration = 20  # 仿真时长(s)
    t_values = np.linspace(0, flight_duration, 100)
    uav_traj = np.array([uav_pos + uav_velocity * t for t in t_values])
    ax1.plot(uav_traj[:, 0], uav_traj[:, 1], uav_traj[:, 2], 'b-', label='UAV FY1 Trajectory')
    ax1.plot([uav_pos[0]], [uav_pos[1]], [uav_pos[2]], 'bo', markersize=8, label='UAV Initial Position')
    ax1.plot([release_pos[0]], [release_pos[1]], [release_pos[2]], 'go', markersize=8, label='Smoke Release Point')

    # 绘制烟幕弹轨迹
    smoke_traj = np.array([calculate_smoke_trajectory(release_pos, uav_velocity, t) for t in np.linspace(0, 5, 50)])
    ax1.plot(smoke_traj[:, 0], smoke_traj[:, 1], smoke_traj[:, 2], 'g--', label='Smoke Trajectory')
    ax1.plot([detonation_pos[0]], [detonation_pos[1]], [detonation_pos[2]], 'ro', markersize=8,
             label='Detonation Point')

    # 绘制烟幕云团下沉轨迹
    if effective_periods:
        max_t = max([end for start, end in effective_periods])
        smoke_cloud_traj = np.array(
            [calculate_smoke_cloud_position(detonation_pos, t) for t in np.linspace(0, max_t, 50)])
        ax1.plot(smoke_cloud_traj[:, 0], smoke_cloud_traj[:, 1], smoke_cloud_traj[:, 2], 'r--', alpha=0.5,
                 label='Smoke Cloud Sinking')

    # 绘制导弹轨迹
    missile_traj = np.array(
        [calculate_missile_position(missile_pos, missile_direction, t) for t in np.linspace(0, flight_duration, 100)])
    ax1.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'm-', label='Missile M1 Trajectory')

    # 标记目标
    ax1.plot([FAKE_TARGET[0]], [FAKE_TARGET[1]], [FAKE_TARGET[2]], 'kx', markersize=10, label='Fake Target')
    ax1.plot([REAL_TARGET[0]], [REAL_TARGET[1]], [REAL_TARGET[2]], 'k*', markersize=10, label='Real Target')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Optimized 3D Trajectories')
    ax1.legend()

    # 2D平面图 (X-Z平面)
    ax2 = fig.add_subplot(122)

    # 绘制轨迹投影
    ax2.plot(uav_traj[:, 0], uav_traj[:, 2], 'b-', label='UAV FY1')
    ax2.plot([uav_pos[0]], [uav_pos[2]], 'bo', markersize=8)
    ax2.plot([release_pos[0]], [release_pos[2]], 'go', markersize=8)

    ax2.plot(smoke_traj[:, 0], smoke_traj[:, 2], 'g--', label='Smoke')
    ax2.plot([detonation_pos[0]], [detonation_pos[2]], 'ro', markersize=8)

    if effective_periods:
        ax2.plot(smoke_cloud_traj[:, 0], smoke_cloud_traj[:, 2], 'r--', alpha=0.5, label='Smoke Cloud')

    ax2.plot(missile_traj[:, 0], missile_traj[:, 2], 'm-', label='Missile M1')

    # 标记目标
    ax2.plot([FAKE_TARGET[0]], [FAKE_TARGET[2]], 'kx', markersize=10, label='Fake Target')
    ax2.plot([REAL_TARGET[0]], [REAL_TARGET[2]], 'k*', markersize=10, label='Real Target')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('X-Z Plane Projection')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('problem2_optimized_visualization.png', dpi=300)
    plt.close(fig)  # 关闭图形，避免显示问题


# ============================ 问题3解答 ============================
def objective_function_problem3(params, uav_pos, missile_pos, missile_direction):
    """问题3的优化目标函数：最大化三枚烟幕弹的总遮蔽时间"""
    # 解析参数: [速度, 投放时间1, 起爆延迟1, 投放时间2, 起爆延迟2, 投放时间3, 起爆延迟3]
    uav_speed = params[0]
    release_times = params[1:4]
    detonation_delays = params[4:7]

    # 约束检查
    if not (70 <= uav_speed <= 140):
        return 1000  # 惩罚值

    # 检查投放时间间隔
    for i in range(2):
        if release_times[i + 1] - release_times[i] < MIN_RELEASE_INTERVAL:
            return 1000

    for rt in release_times:
        if rt < 0:
            return 1000

    for dd in detonation_delays:
        if dd < 0:
            return 1000

    # 计算无人机飞行方向（朝向假目标）
    direction_to_target = FAKE_TARGET - uav_pos
    uav_direction = direction_to_target / np.linalg.norm(direction_to_target)
    uav_velocity = uav_direction * uav_speed

    total_effective_time = 0

    # 计算每枚烟幕弹的效果
    for i in range(3):
        release_time = release_times[i]
        detonation_delay = detonation_delays[i]

        # 计算投放点
        release_pos = uav_pos + uav_velocity * release_time

        # 计算起爆点
        detonation_pos = calculate_smoke_trajectory(release_pos, uav_velocity, detonation_delay)

        # 计算起爆时刻导弹位置
        detonation_time = release_time + detonation_delay
        missile_pos_at_detonation = calculate_missile_position(missile_pos, missile_direction, detonation_time)

        # 计算有效遮蔽时间段
        effective_periods = calculate_effective_period(
            detonation_pos, missile_pos_at_detonation, missile_direction, detonation_time
        )

        # 累加有效遮蔽时间
        total_effective_time += calculate_total_effective_time(effective_periods)

    return -total_effective_time  # 返回负值以便最小化函数


def problem3():
    """解决问题3：优化单无人机三枚烟幕弹对M1的干扰策略"""
    print("=" * 50)
    print("问题3解答:")
    print("=" * 50)

    uav_pos = UAVS['FY1'].copy()
    missile_pos = MISSILES['M1'].copy()
    missile_direction = calculate_missile_direction(missile_pos)

    # 定义参数范围和初始值
    bounds = [(70, 140)]  # 速度范围
    bounds.extend([(0, 10) for _ in range(3)])  # 三个投放时间
    bounds.extend([(0, 10) for _ in range(3)])  # 三个起爆延迟

    # 初始猜测值
    initial_guess = [120, 1.0, 2.5, 4.0, 3.0, 3.5, 4.0]

    # 使用差分进化算法进行全局优化
    result = differential_evolution(
        objective_function_problem3, bounds,
        args=(uav_pos, missile_pos, missile_direction),
        strategy='best1bin', maxiter=200, popsize=20, tol=0.01,
        recombination=0.7, mutation=(0.5, 1), seed=42
    )

    # 提取优化结果
    opt_speed = result.x[0]
    opt_release_times = result.x[1:4]
    opt_detonation_delays = result.x[4:7]

    # 计算无人机飞行方向
    direction_to_target = FAKE_TARGET - uav_pos
    uav_direction = direction_to_target / np.linalg.norm(direction_to_target)
    uav_velocity = uav_direction * opt_speed

    results = []
    total_effective_time = 0

    for i in range(3):
        release_time = opt_release_times[i]
        detonation_delay = opt_detonation_delays[i]

        # 计算投放点和起爆点
        release_pos = uav_pos + uav_velocity * release_time
        detonation_pos = calculate_smoke_trajectory(release_pos, uav_velocity, detonation_delay)

        # 计算有效遮蔽时间
        detonation_time = release_time + detonation_delay
        missile_pos_at_detonation = calculate_missile_position(missile_pos, missile_direction, detonation_time)
        effective_periods = calculate_effective_period(
            detonation_pos, missile_pos_at_detonation, missile_direction, detonation_time
        )
        effective_time = calculate_total_effective_time(effective_periods)
        total_effective_time += effective_time

        results.append({
            'Smoke_ID': i + 1,
            'Release_Time': release_time,
            'Release_Pos': release_pos,
            'Detonation_Delay': detonation_delay,
            'Detonation_Pos': detonation_pos,
            'Effective_Time': effective_time,
            'Effective_Periods': effective_periods
        })

        print(f"\n烟幕弹 {i + 1}:")
        print(f"  投放时间: {release_time:.2f} s")
        print(f"  投放点: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f}) m")
        print(f"  起爆延迟: {detonation_delay:.2f} s")
        print(f"  起爆点: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f}) m")
        print(f"  有效遮蔽时间段: {effective_periods}")
        print(f"  有效遮蔽时长: {effective_time:.2f} s")

    print(f"\n总有效遮蔽时长: {total_effective_time:.2f} s")

    # 保存结果到Excel
    result_df = pd.DataFrame({
        'UAV': ['FY1'] * 3,
        'Smoke_ID': [1, 2, 3],
        'Speed': [opt_speed] * 3,
        'Release_Time': opt_release_times,
        'Detonation_Delay': opt_detonation_delays,
        'Effective_Time': [r['Effective_Time'] for r in results]
    })
    result_df.to_excel('result1.xlsx', index=False)

    # 可视化
    visualize_problem3(uav_pos, uav_velocity, results, missile_pos, missile_direction)

    return {
        'speed': opt_speed,
        'direction': uav_direction,
        'results': results,
        'total_effective_time': total_effective_time
    }


def visualize_problem3(uav_pos, uav_velocity, smoke_results, missile_pos, missile_direction):
    """可视化问题3的优化结果"""
    fig = plt.figure(figsize=(15, 10))

    # 3D轨迹图
    ax1 = fig.add_subplot(121, projection='3d')

    # 绘制无人机轨迹
    flight_duration = 20  # 仿真时长(s)
    t_values = np.linspace(0, flight_duration, 100)
    uav_traj = np.array([uav_pos + uav_velocity * t for t in t_values])
    ax1.plot(uav_traj[:, 0], uav_traj[:, 1], uav_traj[:, 2], 'b-', label='UAV FY1 Trajectory')
    ax1.plot([uav_pos[0]], [uav_pos[1]], [uav_pos[2]], 'bo', markersize=8, label='UAV Initial Position')

    # 绘制烟幕弹轨迹
    colors = ['g', 'c', 'y']
    for i, result in enumerate(smoke_results):
        color = colors[i]
        release_pos = result['Release_Pos']
        detonation_pos = result['Detonation_Pos']

        ax1.plot([release_pos[0]], [release_pos[1]], [release_pos[2]], f'{color}o', markersize=8,
                 label=f'Smoke {i + 1} Release')

        # 绘制烟幕弹轨迹
        smoke_traj = np.array([calculate_smoke_trajectory(release_pos, uav_velocity, t) for t in np.linspace(0, 5, 50)])
        ax1.plot(smoke_traj[:, 0], smoke_traj[:, 1], smoke_traj[:, 2], f'{color}--', label=f'Smoke {i + 1} Trajectory')
        ax1.plot([detonation_pos[0]], [detonation_pos[1]], [detonation_pos[2]], f'{color}s', markersize=8,
                 label=f'Smoke {i + 1} Detonation')

        # 绘制烟幕云团下沉轨迹
        if result['Effective_Periods']:
            max_t = max([end for start, end in result['Effective_Periods']])
            smoke_cloud_traj = np.array(
                [calculate_smoke_cloud_position(detonation_pos, t) for t in np.linspace(0, max_t, 50)])
            ax1.plot(smoke_cloud_traj[:, 0], smoke_cloud_traj[:, 1], smoke_cloud_traj[:, 2], f'{color}-', alpha=0.5,
                     label=f'Smoke {i + 1} Cloud')

    # 绘制导弹轨迹
    missile_traj = np.array(
        [calculate_missile_position(missile_pos, missile_direction, t) for t in np.linspace(0, flight_duration, 100)])
    ax1.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'm-', label='Missile M1 Trajectory')

    # 标记目标
    ax1.plot([FAKE_TARGET[0]], [FAKE_TARGET[1]], [FAKE_TARGET[2]], 'kx', markersize=10, label='Fake Target')
    ax1.plot([REAL_TARGET[0]], [REAL_TARGET[1]], [REAL_TARGET[2]], 'k*', markersize=10, label='Real Target')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3 Smoke Bombs - 3D Trajectories')
    ax1.legend()

    # 2D平面图 (X-Z平面)
    ax2 = fig.add_subplot(122)

    # 绘制轨迹投影
    ax2.plot(uav_traj[:, 0], uav_traj[:, 2], 'b-', label='UAV FY1')
    ax2.plot([uav_pos[0]], [uav_pos[2]], 'bo', markersize=8)

    for i, result in enumerate(smoke_results):
        color = colors[i]
        release_pos = result['Release_Pos']
        detonation_pos = result['Detonation_Pos']

        ax2.plot([release_pos[0]], [release_pos[2]], f'{color}o', markersize=8, label=f'Smoke {i + 1} Release')

        # 绘制烟幕弹轨迹
        smoke_traj = np.array([calculate_smoke_trajectory(release_pos, uav_velocity, t) for t in np.linspace(0, 5, 50)])
        ax2.plot(smoke_traj[:, 0], smoke_traj[:, 2], f'{color}--', label=f'Smoke {i + 1} Trajectory')
        ax2.plot([detonation_pos[0]], [detonation_pos[2]], f'{color}s', markersize=8, label=f'Smoke {i + 1} Detonation')

        # 绘制烟幕云团下沉轨迹
        if result['Effective_Periods']:
            max_t = max([end for start, end in result['Effective_Periods']])
            smoke_cloud_traj = np.array(
                [calculate_smoke_cloud_position(detonation_pos, t) for t in np.linspace(0, max_t, 50)])
            ax2.plot(smoke_cloud_traj[:, 0], smoke_cloud_traj[:, 2], f'{color}-', alpha=0.5,
                     label=f'Smoke {i + 1} Cloud')

    ax2.plot(missile_traj[:, 0], missile_traj[:, 2], 'm-', label='Missile M1')

    # 标记目标
    ax2.plot([FAKE_TARGET[0]], [FAKE_TARGET[2]], 'kx', markersize=10, label='Fake Target')
    ax2.plot([REAL_TARGET[0]], [REAL_TARGET[2]], 'k*', markersize=10, label='Real Target')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('X-Z Plane Projection')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('problem3_visualization.png', dpi=300)
    plt.close(fig)  # 关闭图形，避免显示问题


# ============================ 问题4解答 ============================
def objective_function_problem4(params, uavs_pos, missile_pos, missile_direction):
    """问题4的优化目标函数：最大化三架无人机各一枚烟幕弹的总遮蔽时间"""
    # 解析参数: 每架无人机有 [速度, 投放时间, 起爆延迟]
    total_effective_time = 0

    for i in range(3):
        uav_speed = params[i * 3]
        release_time = params[i * 3 + 1]
        detonation_delay = params[i * 3 + 2]

        # 约束检查
        if not (70 <= uav_speed <= 140):
            return 1000  # 惩罚值

        if release_time < 0 or detonation_delay < 0:
            return 1000

        # 计算无人机飞行方向（朝向假目标）
        direction_to_target = FAKE_TARGET - uavs_pos[i]
        uav_direction = direction_to_target / np.linalg.norm(direction_to_target)
        uav_velocity = uav_direction * uav_speed

        # 计算投放点
        release_pos = uavs_pos[i] + uav_velocity * release_time

        # 计算起爆点
        detonation_pos = calculate_smoke_trajectory(release_pos, uav_velocity, detonation_delay)

        # 计算起爆时刻导弹位置
        detonation_time = release_time + detonation_delay
        missile_pos_at_detonation = calculate_missile_position(missile_pos, missile_direction, detonation_time)

        # 计算有效遮蔽时间段
        effective_periods = calculate_effective_period(
            detonation_pos, missile_pos_at_detonation, missile_direction, detonation_time
        )

        # 累加有效遮蔽时间
        total_effective_time += calculate_total_effective_time(effective_periods)

    return -total_effective_time  # 返回负值以便最小化函数


def problem4():
    """解决问题4：优化三架无人机各一枚烟幕弹对M1的干扰策略"""
    print("=" * 50)
    print("问题4解答:")
    print("=" * 50)

    uavs_pos = [UAVS['FY1'].copy(), UAVS['FY2'].copy(), UAVS['FY3'].copy()]
    missile_pos = MISSILES['M1'].copy()
    missile_direction = calculate_missile_direction(missile_pos)

    # 定义参数范围和初始值
    bounds = []
    for _ in range(3):  # 三架无人机
        bounds.append((70, 140))  # 速度
        bounds.append((0, 10))  # 投放时间
        bounds.append((0, 10))  # 起爆延迟

    # 初始猜测值
    initial_guess = [120, 1.5, 3.6] * 3

    # 使用差分进化算法进行全局优化
    result = differential_evolution(
        objective_function_problem4, bounds,
        args=(uavs_pos, missile_pos, missile_direction),
        strategy='best1bin', maxiter=200, popsize=20, tol=0.01,
        recombination=0.7, mutation=(0.5, 1), seed=42
    )

    results = []
    total_effective_time = 0

    for i in range(3):
        uav_speed = result.x[i * 3]
        release_time = result.x[i * 3 + 1]
        detonation_delay = result.x[i * 3 + 2]
        uav_pos = uavs_pos[i]

        # 计算无人机飞行方向
        direction_to_target = FAKE_TARGET - uav_pos
        uav_direction = direction_to_target / np.linalg.norm(direction_to_target)
        uav_velocity = uav_direction * uav_speed

        # 计算投放点和起爆点
        release_pos = uav_pos + uav_velocity * release_time
        detonation_pos = calculate_smoke_trajectory(release_pos, uav_velocity, detonation_delay)

        # 计算有效遮蔽时间
        detonation_time = release_time + detonation_delay
        missile_pos_at_detonation = calculate_missile_position(missile_pos, missile_direction, detonation_time)
        effective_periods = calculate_effective_period(
            detonation_pos, missile_pos_at_detonation, missile_direction, detonation_time
        )
        effective_time = calculate_total_effective_time(effective_periods)
        total_effective_time += effective_time

        results.append({
            'UAV': f'FY{i + 1}',
            'Speed': uav_speed,
            'Release_Time': release_time,
            'Release_Pos': release_pos,
            'Detonation_Delay': detonation_delay,
            'Detonation_Pos': detonation_pos,
            'Effective_Time': effective_time,
            'Effective_Periods': effective_periods
        })

        print(f"\n无人机 FY{i + 1}:")
        print(f"  飞行速度: {uav_speed:.2f} m/s")
        print(f"  投放时间: {release_time:.2f} s")
        print(f"  投放点: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f}) m")
        print(f"  起爆延迟: {detonation_delay:.2f} s")
        print(f"  起爆点: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f}) m")
        print(f"  有效遮蔽时间段: {effective_periods}")
        print(f"  有效遮蔽时长: {effective_time:.2f} s")

    print(f"\n总有效遮蔽时长: {total_effective_time:.2f} s")

    # 保存结果到Excel
    result_df = pd.DataFrame({
        'UAV': [r['UAV'] for r in results],
        'Speed': [r['Speed'] for r in results],
        'Release_Time': [r['Release_Time'] for r in results],
        'Detonation_Delay': [r['Detonation_Delay'] for r in results],
        'Effective_Time': [r['Effective_Time'] for r in results]
    })
    result_df.to_excel('result2.xlsx', index=False)

    # 可视化
    visualize_problem4(uavs_pos, results, missile_pos, missile_direction)

    return {
        'results': results,
        'total_effective_time': total_effective_time
    }


def visualize_problem4(uavs_pos, smoke_results, missile_pos, missile_direction):
    """可视化问题4的优化结果"""
    fig = plt.figure(figsize=(15, 10))

    # 3D轨迹图
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)  # ✅ 提前定义 ax2，避免作用域问题

    colors = ['b', 'g', 'r']
    uav_names = ['FY1', 'FY2', 'FY3']

    for i, (uav_pos, result) in enumerate(zip(uavs_pos, smoke_results)):
        color = colors[i]
        uav_speed = result['Speed']

        # 计算无人机飞行方向
        direction_to_target = FAKE_TARGET - uav_pos
        uav_direction = direction_to_target / np.linalg.norm(direction_to_target)
        uav_velocity = uav_direction * uav_speed

        # 绘制无人机轨迹
        flight_duration = 20
        t_values = np.linspace(0, flight_duration, 100)
        uav_traj = np.array([uav_pos + uav_velocity * t for t in t_values])
        ax1.plot(uav_traj[:, 0], uav_traj[:, 1], uav_traj[:, 2], f'{color}-', label=f'UAV {uav_names[i]} Trajectory')
        ax1.plot([uav_pos[0]], [uav_pos[1]], [uav_pos[2]], f'{color}o', markersize=8, label=f'UAV {uav_names[i]} Initial')

        # 烟幕弹轨迹
        release_pos = result['Release_Pos']
        detonation_pos = result['Detonation_Pos']
        ax1.plot([release_pos[0]], [release_pos[1]], [release_pos[2]], f'{color}^', markersize=8, label=f'Smoke {i+1} Release')
        smoke_traj = np.array([calculate_smoke_trajectory(release_pos, uav_velocity, t) for t in np.linspace(0, 5, 50)])
        ax1.plot(smoke_traj[:, 0], smoke_traj[:, 1], smoke_traj[:, 2], f'{color}--', label=f'Smoke {i+1} Trajectory')
        ax1.plot([detonation_pos[0]], [detonation_pos[1]], [detonation_pos[2]], f'{color}s', markersize=8, label=f'Smoke {i+1} Detonation')

        if result['Effective_Periods']:
            max_t = max([end for start, end in result['Effective_Periods']])
            smoke_cloud_traj = np.array([calculate_smoke_cloud_position(detonation_pos, t) for t in np.linspace(0, max_t, 50)])
            ax1.plot(smoke_cloud_traj[:, 0], smoke_cloud_traj[:, 1], smoke_cloud_traj[:, 2], f'{color}-', alpha=0.5, label=f'Smoke {i+1} Cloud')

        # ✅ 2D 投影图
        ax2.plot(uav_traj[:, 0], uav_traj[:, 2], f'{color}-', label=f'UAV {uav_names[i]}')
        ax2.plot([uav_pos[0]], [uav_pos[2]], f'{color}o', markersize=8)
        ax2.plot([release_pos[0]], [release_pos[2]], f'{color}^', markersize=8, label=f'Smoke {i+1} Release')
        ax2.plot(smoke_traj[:, 0], smoke_traj[:, 2], f'{color}--', label=f'Smoke {i+1} Trajectory')
        ax2.plot([detonation_pos[0]], [detonation_pos[2]], f'{color}s', markersize=8, label=f'Smoke {i+1} Detonation')
        if result['Effective_Periods']:
            ax2.plot(smoke_cloud_traj[:, 0], smoke_cloud_traj[:, 2], f'{color}-', alpha=0.5, label=f'Smoke {i+1} Cloud')

    # 绘制导弹轨迹
    missile_traj = np.array([calculate_missile_position(missile_pos, missile_direction, t) for t in np.linspace(0, flight_duration, 100)])
    ax1.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'm-', label='Missile M1 Trajectory')
    ax2.plot(missile_traj[:, 0], missile_traj[:, 2], 'm-', label='Missile M1')

    # 标记目标
    ax1.plot([FAKE_TARGET[0]], [FAKE_TARGET[1]], [FAKE_TARGET[2]], 'kx', markersize=10, label='Fake Target')
    ax1.plot([REAL_TARGET[0]], [REAL_TARGET[1]], [REAL_TARGET[2]], 'k*', markersize=10, label='Real Target')
    ax2.plot([FAKE_TARGET[0]], [FAKE_TARGET[2]], 'kx', markersize=10, label='Fake Target')
    ax2.plot([REAL_TARGET[0]], [REAL_TARGET[2]], 'k*', markersize=10, label='Real Target')

    ax1.set_title('3 UAVs - 3D Trajectories')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('X-Z Plane Projection')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('problem4_visualization.png', dpi=300)
    plt.close(fig)
    # ============================ 问题5解答 ============================


def problem5():
    """解决问题5：五架无人机对三枚导弹的干扰策略"""
    print("=" * 50)
    print("问题5解答:")
    print("=" * 50)
    print("问题5涉及多无人机多导弹协同优化，计算复杂度较高")
    print("这里提供一个简化版的解决方案框架")

    # 简化策略：每架无人机针对最近的导弹进行干扰
    results = []

    for uav_name, uav_pos in UAVS.items():
        # 找到最近的导弹
        min_dist = float('inf')
        target_missile = None
        missile_direction = None

        for missile_name, missile_pos in MISSILES.items():
            dist = distance_between(uav_pos, missile_pos)
            if dist < min_dist:
                min_dist = dist
                target_missile = missile_name
                missile_direction = calculate_missile_direction(missile_pos)

        # 使用问题2的优化方法针对该导弹进行优化
        missile_pos = MISSILES[target_missile]

        # 定义参数范围和初始值
        bounds = [(70, 140), (0, 10), (0, 10)]
        initial_guess = [120, 1.5, 3.6]

        # 使用优化算法
        result = minimize(
            objective_function_problem2, initial_guess,
            args=(uav_pos, missile_pos, missile_direction),
            bounds=bounds,
            method='L-BFGS-B'
        )

        # 提取优化结果
        opt_speed = result.x[0]
        opt_release_time = result.x[1]
        opt_detonation_delay = result.x[2]

        # 计算无人机飞行方向
        direction_to_target = FAKE_TARGET - uav_pos
        uav_direction = direction_to_target / np.linalg.norm(direction_to_target)
        uav_velocity = uav_direction * opt_speed

        # 计算投放点和起爆点
        release_pos = uav_pos + uav_velocity * opt_release_time
        detonation_pos = calculate_smoke_trajectory(release_pos, uav_velocity, opt_detonation_delay)

        # 计算有效遮蔽时间
        detonation_time = opt_release_time + opt_detonation_delay
        missile_pos_at_detonation = calculate_missile_position(missile_pos, missile_direction, detonation_time)
        effective_periods = calculate_effective_period(
            detonation_pos, missile_pos_at_detonation, missile_direction, detonation_time
        )
        effective_time = calculate_total_effective_time(effective_periods)

        results.append({
            'UAV': uav_name,
            'Target_Missile': target_missile,
            'Speed': opt_speed,
            'Release_Time': opt_release_time,
            'Detonation_Delay': opt_detonation_delay,
            'Effective_Time': effective_time
        })

        print(f"\n{uav_name} 针对 {target_missile}:")
        print(f"  飞行速度: {opt_speed:.2f} m/s")
        print(f"  投放时间: {opt_release_time:.2f} s")
        print(f"  起爆延迟: {opt_detonation_delay:.2f} s")
        print(f"  有效遮蔽时长: {effective_time:.2f} s")

    # 保存结果到Excel
    result_df = pd.DataFrame(results)
    result_df.to_excel('result3.xlsx', index=False)

    return results


# ============================ 主程序 ============================
if __name__ == "__main__":
    print("烟幕干扰弹投放策略优化与仿真")
    print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S"))

    try:
        # 解决问题1
        p1_result = problem1()

        # 解决问题2
        p2_result = problem2()

        # 解决问题3
        p3_result = problem3()

        # 解决问题4
        p4_result = problem4()

        # 解决问题5
        p5_result = problem5()

        print("\n所有问题解答完成!")
        print("结束时间:", time.strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
        import traceback

        traceback.print_exc()