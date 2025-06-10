import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm以显示进度条
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者使用 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 计算有效样本大小 (ESS)
def compute_ess(weights):
    """计算有效样本大小 (ESS)"""
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        print("Error: Invalid weight values (NaN or inf) detected.")
        return np.nan
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    return ess


# 重采样函数
def resample(weights):
    """根据权重进行重采样"""
    indices = np.random.choice(len(weights), size=len(weights), p=weights)
    return indices


# SEIR模型，添加年龄分层
def seir_model_age_stratified(params, N, days, age_groups=5):
    betas, sigmas, gammas = params  # 传染率、潜伏期、恢复期
    S = np.ones(age_groups) * (N / age_groups)  # 每个年龄组的初始易感人数
    E = np.ones(age_groups)  # 每个年龄组的初始潜伏人数
    I = np.zeros(age_groups)  # 每个年龄组的初始感染人数
    R = np.zeros(age_groups)  # 每个年龄组的初始康复人数

    # 接触矩阵
    M = np.array([
        [0.60, 0.50, 0.30, 0.20, 0.10],  # 0-4岁
        [0.50, 1.80, 0.60, 0.40, 0.20],  # 5-14岁
        [0.30, 0.60, 2.0, 0.50, 0.30],  # 15-24岁
        [0.20, 0.40, 0.90, 0.80, 0.50],  # 25-64岁
        [0.10, 0.20, 0.30, 0.50, 0.80]  # 65岁及以上
    ])

    S_vals, E_vals, I_vals, R_vals = [S.copy()], [E.copy()], [I.copy()], [R.copy()]

    for _ in range(days):
        # 计算每个年龄组的传染力
        lambda_vals = np.dot(M, I / N)  # 接触矩阵与当前感染者比例的乘积

        # 计算每个年龄组的变化
        dS = -betas * S * lambda_vals  # 传染使易感者变成潜伏者
        dE = betas * S * lambda_vals - sigmas * E  # 潜伏者进入感染者
        dI = sigmas * E - gammas * I  # 感染者恢复
        dR = gammas * I  # 康复

        S += dS
        E += dE
        I += dI
        R += dR
        S_vals.append(S.copy())
        E_vals.append(E.copy())
        I_vals.append(I.copy())
        R_vals.append(R.copy())

    return np.array(S_vals), np.array(E_vals), np.array(I_vals), np.array(R_vals)


def calculate_likelihood(simulated_I, real_I):
    """计算粒子的似然（使用均方误差）"""
    mse = np.mean((simulated_I - real_I) ** 2)  # 均方误差
    likelihood = np.exp(-mse / 2)  # 假设误差服从正态分布

    # 防止似然度过小导致权重为零
    likelihood = max(likelihood, 1e-10)  # 设置最小似然度值

    return likelihood


# SMC算法，添加年龄分层部分
def smc_seir_age_stratified(N, days, num_particles=1000, age_groups=5, I_real=None):
    betas = np.random.uniform(1.3, 1.8, (num_particles, age_groups))  # 每个粒子的每个年龄组的传染率
    sigmas = np.random.uniform(1 / 7, 1 / 4, (num_particles, age_groups))  # 潜伏期
    gammas = np.random.uniform(1 / 14, 1 / 7, (num_particles, age_groups))  # 恢复期

    params = np.stack([betas, sigmas, gammas], axis=1)  # params形状为(num_particles, 3, age_groups)
    weights = np.ones(num_particles) / num_particles  # 初始均匀权重

    I_simulations = np.zeros((num_particles, days + 1, age_groups))
    ess_values = []  # 存储每个时间步的ESS值
    betas_history, sigmas_history, gammas_history = [], [], []  # 存储参数演变

    # 使用tqdm为循环添加进度条
    for t in tqdm(range(days + 1), desc="Running SMC simulation", ncols=100):
        for i in range(num_particles):
            S_vals, E_vals, I_vals, R_vals = seir_model_age_stratified(params[i], N, days, age_groups)
            I_simulations[i] = I_vals

        # 计算每个粒子与观测数据的似然，更新权重
        for i in range(num_particles):
            likelihood = calculate_likelihood(I_simulations[i, t, :], I_real[t])  # 比较当天的感染人数
            weights[i] *= likelihood  # 更新粒子的权重

        # 检查权重是否出现无效值
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            print(f"Warning: Invalid weights detected at time step {t}.")
            weights = np.ones(num_particles) / num_particles  # 重置权重

        # 归一化权重
        weights /= np.sum(weights)

        # 计算有效样本大小 (ESS)
        ess = compute_ess(weights)
        ess_values.append(ess)

        # 存储每个时间步的参数
        betas_history.append(betas.mean(axis=0))  # 每个年龄组的beta平均值
        sigmas_history.append(sigmas.mean(axis=0))  # 每个年龄组的sigma平均值
        gammas_history.append(gammas.mean(axis=0))  # 每个年龄组的gamma平均值

        # 如果ESS太小，则进行重采样
        if ess < num_particles / 2:
            indices = resample(weights)
            params = params[indices]  # 重采样粒子的参数
            weights = np.ones(num_particles) / num_particles  # 重置均匀权重

    # 保存参数演变到CSV文件
    betas_df = pd.DataFrame(betas_history, columns=[f'Age Group {i + 1}' for i in range(age_groups)])
    sigmas_df = pd.DataFrame(sigmas_history, columns=[f'Age Group {i + 1}' for i in range(age_groups)])
    gammas_df = pd.DataFrame(gammas_history, columns=[f'Age Group {i + 1}' for i in range(age_groups)])

    betas_df.to_csv('betas_evolution.csv', index=False)
    sigmas_df.to_csv('sigmas_evolution.csv', index=False)
    gammas_df.to_csv('gammas_evolution.csv', index=False)

    return I_simulations, ess_values,betas_df,sigmas_df,gammas_df


# 参数设置
N = 1000  # 总人口
days = 80  # 模拟天数
age_groups = 5  # 假设将人群分为5个年龄组

# 生成真实感染数据
true_infection_trend = np.zeros(days + 1)
infection_peak_day = 40  # 假设前40天内感染人数逐渐达到峰值

for t in range(81):
    if t < infection_peak_day:
        true_infection_trend[t] = (70 * (t / infection_peak_day))  # 模拟上升
    else:
        true_infection_trend[t] = 80 - (70 * ((t - infection_peak_day) / (80 - infection_peak_day)))  # 模拟下降

# 后面的天数（81到256天）感染人数在0到10之间波动
for t in range(2 * infection_peak_day + 1, days + 1):
    true_infection_trend[t] = np.random.uniform(0, 5)  # 生成0到10之间的波动数据

# 加入一些噪声，模拟现实中的观测数据
noise = np.random.uniform(0, 3, days + 1)  # 加入噪声
I_real = true_infection_trend + noise  # 真实数据

# SMC算法模拟
I_simulations_smc, ess_values_smc,betas_df,sigmas_df,gammas_df = smc_seir_age_stratified(N, days, age_groups=age_groups, I_real=I_real)

# 可视化每个年龄组的感染人数
plt.figure(figsize=(10, 6))

# 可视化传染率参数（Beta）的变化
plt.figure(figsize=(10, 6))
for i in range(age_groups):
    plt.plot(betas_df[f'Age Group {i + 1}'], label=f'Beta Age Group {i + 1}')
plt.title('传染率（Beta）随时间变化')
plt.xlabel('时间（天）')
plt.ylabel('传染率')
plt.legend()
plt.show()

# 可视化潜伏期参数（Sigma）的变化
plt.figure(figsize=(10, 6))
for i in range(age_groups):
    plt.plot(sigmas_df[f'Age Group {i + 1}'], label=f'Sigma Age Group {i + 1}')
plt.title('潜伏期（Sigma）随时间变化')
plt.xlabel('时间（天）')
plt.ylabel('潜伏期')
plt.legend()
plt.show()

# 可视化恢复期参数（Gamma）的变化
plt.figure(figsize=(10, 6))
for i in range(age_groups):
    plt.plot(gammas_df[f'Age Group {i + 1}'], label=f'Gamma Age Group {i + 1}')
plt.title('恢复期（Gamma）随时间变化')
plt.xlabel('时间（天）')
plt.ylabel('恢复期')
plt.legend()
plt.show()
