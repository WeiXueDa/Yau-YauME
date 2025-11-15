import streamlit as st
import numpy as np
import math
import time
from scipy.special import eval_hermite
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.sparse as sp
from scipy.sparse import spdiags, eye
from scipy.fft import dst, idst
from scipy.sparse.linalg import spsolve
from sklearn.metrics import mean_squared_error, r2_score
import inspect

# ==============================
# Common Utility Functions
# ==============================
def safe_eval_func(func_str, var_names):
    """Safely evaluate function string into a callable"""
    try:
        if len(var_names) == 2:
            # 尝试解析为双参数函数
            func = eval(f"lambda {var_names[0]},{var_names[1]}: {func_str}")
            # 检查函数是否需要时间参数
            sig = inspect.signature(func)
            if len(sig.parameters) == 1:
                # 如果实际只需要一个参数，包装成双参数函数但忽略时间参数
                return lambda x, t: func(x)
            return func
        else:
            return eval(f"lambda {var_names[0]}: {func_str}")
    except Exception as e:
        st.error(f"Function parsing error: {str(e)}")
        return None

# ==============================
# Hermite Spectral Method Classes
# ==============================
class StochasticModelHermite:
    def __init__(self, f, h, sigma_x_func, sigma_y_func):
        self.f = f
        self.h = h
        self.sigma_x_func = sigma_x_func
        self.sigma_y_func = sigma_y_func

class HermiteBasis:
    def __init__(self, N, scale=1.0):
        self.N = N
        self.scale = scale
        self.log_factorials = np.zeros(N + 1)
        for i in range(1, N + 1):
            self.log_factorials[i] = self.log_factorials[i-1] + math.log(i)

    def get_coeff(self, n):
        log_denom = 0.5 * (n * math.log(2.0) + self.log_factorials[n] + 0.5 * math.log(math.pi) + math.log(self.scale))
        return math.exp(-log_denom)

    def phi(self, n, x, mu=0.0):
        s = self.scale
        z = (x - mu) / s
        coeff = self.get_coeff(n)
        return coeff * np.exp(-0.5 * z**2) * eval_hermite(n, z)

    def phi_x(self, n, x, mu=0.0):
        s = self.scale
        z = (x - mu) / s
        coeff = self.get_coeff(n)
        Hn = eval_hermite(n, z)
        Hn_1 = eval_hermite(n - 1, z) if n > 0 else np.zeros_like(z)
        deriv_z = 2.0 * n * Hn_1 - z * Hn
        return coeff * np.exp(-0.5 * z**2) * (deriv_z / s)

    def phi_xx(self, n, x, mu=0.0):
        s = self.scale
        z = (x - mu) / s
        coeff = self.get_coeff(n)
        Hn = eval_hermite(n, z)
        Hn_1 = eval_hermite(n - 1, z) if n > 0 else np.zeros_like(z)
        deriv_zz = (z**2 - 1.0 - 2.0*n) * Hn + 2.0 * z * n * Hn_1
        return coeff * np.exp(-0.5 * z**2) * (deriv_zz / s**2)
    
    def evaluate_basis_set(self, x, mu):
        K = self.N + 1
        PHI = np.zeros((K, x.size))
        PHI_x = np.zeros_like(PHI)
        PHI_xx = np.zeros_like(PHI)
        for n in range(K):
            PHI[n, :] = self.phi(n, x, mu)
            PHI_x[n, :] = self.phi_x(n, x, mu)
            PHI_xx[n, :] = self.phi_xx(n, x, mu)
        return PHI, PHI_x, PHI_xx

class Quadrature:
    def __init__(self, M_quad, scale):
        self.M_quad = M_quad
        self.scale = scale
        self.z_phys, self.w_phys = hermgauss(M_quad)
        
    def get_nodes_weights(self, mu):
        s = self.scale
        X = s * self.z_phys + mu
        W = s * self.w_phys * np.exp(self.z_phys**2)
        return X, W

class GalerkinOperator:
    def __init__(self, model, basis, M_quad=80):
        self.model = model
        self.basis = basis
        self.quadrature = Quadrature(M_quad, basis.scale)
        self.K = basis.N + 1

    def get_nodes_weights(self, mu):
        return self.quadrature.get_nodes_weights(mu)

    def build_matrices(self, t, mu):
        f, h, sigma_x_func = self.model.f, self.model.h, self.model.sigma_x_func
        sx = sigma_x_func(t)
        X, W = self.get_nodes_weights(mu)
        PHI, PHI_x, PHI_xx = self.basis.evaluate_basis_set(X, mu)
        hx = h(X, t)
        fx = f(X, t)
        Lphi = fx[None, :] * PHI_x + 0.5 * (sx**2) * PHI_xx
        W_PHI = PHI * W[None, :]
        Mmat = PHI @ W_PHI.T
        Amat = PHI @ (Lphi * W[None, :]).T
        return Mmat, Amat, PHI, hx, X, W

def simulate_state_observation_hermite(model, LT, dt, x0, y0, seed):
    np.random.seed(seed)
    xs, ys = np.zeros(LT + 1), np.zeros(LT + 1)
    xs[0], ys[0] = x0, y0
    for i in range(LT):
        t = i * dt
        dW = np.sqrt(dt) * np.random.randn()
        dB = np.sqrt(dt) * np.random.randn()
        sigma_x = model.sigma_x_func(t)
        sigma_y = model.sigma_y_func(t)
        xs[i + 1] = xs[i] + model.f(xs[i], t) * dt + sigma_x * dW
        ys[i + 1] = ys[i] + model.h(xs[i], t) * dt + sigma_y * dB
    return xs, ys

class AdaptiveMemorylessFilter:
    def __init__(self, model, operator, dt, integrator="expm"):
        self.model = model
        self.op = operator
        self.dt = dt
        self.integrator = integrator

    def run(self, x_true, y_obs, rho0, mu0):
        est = np.zeros_like(x_true)
        mu = mu0
        est[0] = mu0
        X_grid, W_grid = self.op.get_nodes_weights(mu=mu)
        rho = rho0.copy()
        
        for k in range(1, len(y_obs)):
            t = (k - 1) * self.dt
            y_prev, y_curr = y_obs[k - 1], y_obs[k]
            dy = y_curr - y_prev
            
            sigma_y_t = self.model.sigma_y_func(t)
            sigma_y_sq_t = sigma_y_t**2
            X_old = X_grid
            rho_old = rho

            Mmat, Amat, PHI, hX, X_grid, W_grid = self.op.build_matrices(t, mu)
            B = np.linalg.solve(Mmat, Amat)

            if np.array_equal(X_old, X_grid):
                rho_interp = rho_old
            else:
                interp_func = interp1d(X_old, rho_old, kind='linear', bounds_error=False, fill_value=0.0)
                rho_interp = interp_func(X_grid)
            
            rho_interp = np.maximum(rho_interp, 0)
            norm_interp = np.sum(rho_interp * W_grid)
            if norm_interp > 1e-15:
                rho_interp /= norm_interp
            else:
                rho_interp = W_grid / np.sum(W_grid**2)

            b = np.sum(PHI * (rho_interp[None, :] * W_grid[None, :]), axis=1)
            a0 = np.linalg.solve(Mmat, b)

            if self.integrator == "expm":
                a_new = expm(self.dt * B) @ a0
            else:
                def rhs(t, a): return B @ a
                sol = solve_ivp(rhs, [0, self.dt], a0)
                a_new = sol.y[:, -1]

            rho_pre = (a_new[:, None] * PHI).sum(axis=0)
            rho_pre = np.maximum(rho_pre, 0)
            norm_pre = np.sum(rho_pre * W_grid)
            if norm_pre > 1e-15:
                rho_pre /= norm_pre
            else:
                rho_pre = W_grid / np.sum(W_grid**2)

            likelihood_exp = (hX * dy - 0.5 * hX**2 * self.dt) / sigma_y_sq_t
            likelihood_exp = np.clip(likelihood_exp, -50, 50)
            likelihood = np.exp(likelihood_exp)
            
            rho_k_unnorm = likelihood * rho_pre
            rho_k_unnorm = np.maximum(rho_k_unnorm, 0)
            norm_k = np.sum(rho_k_unnorm * W_grid)
            if norm_k > 1e-15:
                rho_k = rho_k_unnorm / norm_k
            else:
                rho_k = W_grid / np.sum(W_grid**2)

            mu_new = np.sum(X_grid * rho_k * W_grid)
            est[k] = mu_new
            mu = mu_new
            rho = rho_k

        return est, rho, X_grid

class EKF_Hermite:
    def __init__(self, model, dt):
        self.model = model
        self.dt = dt
        self.x_est = None
        self.P_est = None

    def predict(self, x_prev, P_prev, t):
        x_prev = float(x_prev)
        f_val = float(self.model.f(x_prev, t))
        x_pred = x_prev + f_val * self.dt

        eps = 1e-6
        f_plus = float(self.model.f(x_prev + eps, t))
        F_scalar = (f_plus - f_val) / eps
        F_mat = np.array([[F_scalar]])

        sigma_x = float(self.model.sigma_x_func(t))
        Q = np.array([[sigma_x**2 * self.dt]])
        P_pred = F_mat @ P_prev @ F_mat.T + Q
        return float(x_pred), P_pred

    def update(self, x_pred, P_pred, y_prev, y_curr, t):
        x_pred = float(x_pred)
        dy = float(y_curr - y_prev)
        h_pred = float(self.model.h(x_pred, t))

        eps = 1e-6
        h_plus = float(self.model.h(x_pred + eps, t))
        H_scalar = (h_plus - h_pred) / eps
        H_mat = np.array([[H_scalar]])

        sigma_y = float(self.model.sigma_y_func(t))
        R = np.array([[sigma_y**2 * self.dt]])
        S = H_mat @ P_pred @ H_mat.T + R
        K_mat = P_pred @ H_mat.T @ np.linalg.inv(S)

        innovation = dy - h_pred * self.dt
        K_scalar = float(K_mat.item())
        x_est = x_pred + K_scalar * innovation

        I = np.eye(1)
        P_est = (I - K_mat @ H_mat) @ P_pred
        return float(x_est), P_est

    def run(self, y_obs, x0, P0):
        est = np.zeros_like(y_obs, dtype=float)
        est[0] = float(x0)
        self.x_est = float(x0)
        self.P_est = P0
        for k in range(1, len(y_obs)):
            t = (k-1)*self.dt
            x_pred, P_pred = self.predict(self.x_est, self.P_est, t)
            x_est, P_est = self.update(x_pred, P_pred, y_obs[k-1], y_obs[k], t)
            est[k] = float(x_est)
            self.x_est = float(x_est)
            self.P_est = P_est
        return est

class ParticleFilter_Hermite:
    def __init__(self, model, dt, n_particles=100):
        self.model = model
        self.dt = dt
        self.n_particles = n_particles
        self.particles = None
        self.weights = None

    def initialize(self, x0, std=0.1):
        self.particles = np.array([float(x0) + std * np.random.randn() for _ in range(self.n_particles)])
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self, t):
        sigma_x = float(self.model.sigma_x_func(t))
        for i in range(self.n_particles):
            particle = float(self.particles[i])
            f_val = float(self.model.f(particle, t))
            dW = np.sqrt(self.dt) * np.random.randn()
            self.particles[i] = particle + f_val * self.dt + sigma_x * dW

    def update(self, y_prev, y_curr, t):
        dy = float(y_curr - y_prev)
        sigma_y = float(self.model.sigma_y_func(t))
        sigma_y_sq_dt = sigma_y**2 * self.dt
        
        h_vals = np.array([float(self.model.h(p, t)) for p in self.particles])
        residuals = dy - h_vals * self.dt
        
        weights = np.exp(-residuals**2 / (2 * sigma_y_sq_dt))
        weights += 1e-10
        self.weights = weights / np.sum(weights)

    def resample(self):
        indices = np.random.choice(
            range(self.n_particles),
            size=self.n_particles,
            replace=True,
            p=self.weights
        )
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(self.n_particles) / self.n_particles

    def get_estimate(self):
        return np.sum(self.particles * self.weights)

    def run(self, y_obs, x0, seed):
        np.random.seed(seed)
        est = np.zeros_like(y_obs, dtype=float)
        self.initialize(x0)
        est[0] = self.get_estimate()
        for k in range(1, len(y_obs)):
            t = (k-1)*self.dt
            self.predict(t)
            self.update(y_obs[k-1], y_obs[k], t)
            self.resample()
            est[k] = self.get_estimate()
        return est

# ==============================
# Finite Difference Method (QIEM) Classes/Functions
# ==============================
def set_random_seed_qiem(seed=22):
    np.random.seed(seed)

def check_stability_condition(p_max, q_max, ds, dt):
    left_term = (ds**2) * (2 * q_max + (q_max**2) * dt) + dt * (p_max**2)
    is_stable = left_term <= 2
    return is_stable

def initialize_domain_qiem(Ds, R):
    n_total = int((R - (-R)) / Ds) + 1
    s = np.linspace(-R, R, n_total)
    n_internal = n_total - 2

    e = np.ones(n_internal)
    L_1d = spdiags([e, -2*e, e], [-1, 0, 1], n_internal, n_internal).tocsc()
    K_1d = spdiags([-e, e], [-1, 1], n_internal, n_internal).tocsc()
    
    return s, n_total, n_internal, L_1d, K_1d

def simulate_processes_qiem(f_func, h_func, sigma_theta_func, sigma_S, T, Dt, Dtau, seed):
    np.random.seed(seed)
    Nt = int(Dtau / Dt)
    Ntau = int(T / Dtau)
    NtNtau = Ntau * Nt
    
    theta_true = np.zeros(NtNtau)
    theta_true[0] = 1.0  # 初始状态
    y_full = np.zeros(NtNtau)  # 全量观测（小步）
    
    # 检查函数参数个数
    f_sig = inspect.signature(f_func)
    h_sig = inspect.signature(h_func)
    
    # 初始化观测值 - 根据函数参数个数调用
    if len(h_sig.parameters) == 1:
        y_full[0] = h_func(theta_true[0]) * Dt
    else:
        y_full[0] = h_func(theta_true[0], 0) * Dt

    for t in range(1, NtNtau):
        current_time = t * Dt
        
        # 获取当前时间的状态噪声
        if callable(sigma_theta_func):
            current_sigma_theta = sigma_theta_func(current_time)
        else:
            current_sigma_theta = sigma_theta_func
            
        # 检查函数参数个数并相应调用
        if len(f_sig.parameters) == 1:
            drift_theta = f_func(theta_true[t-1])
        else:
            drift_theta = f_func(theta_true[t-1], current_time)
            
        theta_true[t] = theta_true[t-1] + drift_theta * Dt + current_sigma_theta * np.sqrt(Dt) * np.random.randn()
        
        if len(h_sig.parameters) == 1:
            drift_y = h_func(theta_true[t-1])
        else:
            drift_y = h_func(theta_true[t-1], current_time)
            
        y_full[t] = y_full[t-1] + drift_y * Dt + sigma_S * np.sqrt(Dt) * np.random.randn()
    
    y_tau = y_full[::Nt]  # 仅保留每个观测间隔的观测值
    return theta_true, y_full, y_tau

def solve_1d_dst(b, N, ds, dt):
    i = np.arange(1, N+1)
    lambda_i = -4 * np.sin(i * np.pi / (2 * (N + 1))) ** 2
    A_diag = 1 - (dt / (2 * ds ** 2)) * lambda_i
    b_hat = dst(b, type=2, norm=None)
    u_hat = b_hat / A_diag
    u = idst(u_hat, type=2, norm=None)
    return u

def run_qiem_filter_dst_1d(s, n_total, n_internal, L_1d, K_1d, y_tau, f_func, h_func, Ds, Dt, Dtau, sigma_theta_func):
    s_internal = s[1:-1]
    initial_theta = 1.0
    sigma0 = np.exp(-50 * (s_internal - initial_theta) ** 2)
    U = sigma0 / np.sum(sigma0)
    
    Nt = int(Dtau / Dt)
    Ntau = len(y_tau)
    NtNtau = Ntau * Nt
    theta_est = np.zeros(NtNtau)
    theta_est[0] = np.sum(U * s_internal)
    
    I = eye(n_internal, format='csc')
    
    # 检查函数参数个数
    f_sig = inspect.signature(f_func)
    h_sig = inspect.signature(h_func)
    
    for k in range(Ntau):
        current_tau = k * Dtau
        
        # 观测更新：仅在每个观测间隔（k>0）执行
        if k > 0:
            delta_y = y_tau[k] - y_tau[k-1]
            
            # 根据函数参数个数调用
            if len(h_sig.parameters) == 1:
                exponent = delta_y * h_func(s_internal)
            else:
                exponent = delta_y * h_func(s_internal, current_tau)
                
            exponent = np.clip(exponent, -30, 30)
            obs_likelihood = np.exp(exponent)
            
            U *= obs_likelihood
            U = np.maximum(U, 1e-30)
            U /= np.sum(U)
            theta_est[k * Nt] = np.sum(U * s_internal)
        
        # 预测小步：每个观测间隔内的Nt个小步
        for i in range(Nt):
            t_idx = k * Nt + i
            if t_idx >= NtNtau - 1:
                break
            
            inner_time = current_tau + i * Dt
            
            # 获取当前时间的状态噪声
            if callable(sigma_theta_func):
                current_sigma_theta = sigma_theta_func(inner_time)
            else:
                current_sigma_theta = sigma_theta_func
            
            # 根据函数参数个数调用
            if len(f_sig.parameters) == 1:
                p_vals = -f_func(s_internal)
            else:
                p_vals = -f_func(s_internal, inner_time)
                
            if len(h_sig.parameters) == 1:
                q_vals = -0.5 * (h_func(s_internal) ** 2)
            else:
                q_vals = -0.5 * (h_func(s_internal, inner_time) ** 2)
            
            # 确保p_vals和q_vals是正确形状的数组
            if np.isscalar(p_vals):
                p_vals = np.full(n_internal, p_vals)
            if np.isscalar(q_vals):
                q_vals = np.full(n_internal, q_vals)
            
            # 扩散项系数包含时变噪声
            diffusion_coeff = 0.5 * (current_sigma_theta ** 2)
            
            # 修改矩阵构建以包含扩散项
            P = spdiags(p_vals, 0, n_internal, n_internal, format='csc')
            PK = P @ K_1d
            PK = (1 / (2 * Ds)) * PK
            Q = spdiags(q_vals, 0, n_internal, n_internal, format='csc')
            
            # 添加扩散项（拉普拉斯算子）
            L_diffusion = diffusion_coeff * (1 / (Ds ** 2)) * L_1d
            
            A_f = PK + Q + L_diffusion
            
            b_vec = U + Dt * A_f.dot(U)
            U_new = solve_1d_dst(b_vec, n_internal, Ds, Dt)
            
            U = np.maximum(U_new, 1e-30)
            U /= np.sum(U)
            theta_est[t_idx + 1] = np.sum(U * s_internal)
    
    return theta_est

# ==============================
# 准隐式欧拉方法（匹配您提供的代码）
# ==============================
def run_qiem_filter_quasi_implicit(s, n_total, n_internal, L_1d, K_1d, y_tau, f_func, h_func, Ds, Dt, Dtau, sigma_theta, sigma_S):
    """准隐式欧拉方法实现，匹配您提供的独立代码"""
    s_flat = s[1:-1]  # 内部点
    
    # 初始概率密度
    sigma0 = np.exp(-10 * s_flat**2)
    U = sigma0 / np.sum(sigma0)
    
    Nt = int(Dtau / Dt)
    Ntau = len(y_tau)
    NtNtau = Ntau * Nt
    theta_est = np.zeros(NtNtau)
    theta_est[0] = np.sum(U * s_flat)
    
    # 系数矩阵
    I = eye(n_internal, format='csc')
    L = (1/Ds**2) * L_1d  # 拉普拉斯算子
    
    # 检查函数参数个数
    f_sig = inspect.signature(f_func)
    h_sig = inspect.signature(h_func)
    
    # q(s) 项 - 使用您提供的代码中的q_func
    def q_func(x):
        """论文中的q(s) = -(∇·f(s) + 0.5*||h(s)||²)"""
        # 计算f的导数
        eps = 1e-6
        if len(f_sig.parameters) == 1:
            df_dx = (f_func(x + eps) - f_func(x - eps)) / (2 * eps)
        else:
            # 对于双参数函数，我们使用固定的时间0
            df_dx = (f_func(x + eps, 0) - f_func(x - eps, 0)) / (2 * eps)
            
        if len(h_sig.parameters) == 1:
            h_val = h_func(x)
        else:
            h_val = h_func(x, 0)
            
        return -(df_dx + 0.5 * h_val**2)
    
    # p(s) = -f(s)
    if len(f_sig.parameters) == 1:
        p_vals = -f_func(s_flat)
    else:
        p_vals = -f_func(s_flat, 0)  # 使用固定时间0
    
    P_diag = spdiags(p_vals, 0, n_internal, n_internal, format='csc')
    
    q_vals = q_func(s_flat)
    Q_diag = spdiags(q_vals, 0, n_internal, n_internal, format='csc')
    
    # 一阶导数项
    K = (1/(2*Ds)) * K_1d
    
    for k in range(Ntau):
        # 观测更新
        if k > 0:
            delta_y = y_tau[k] - y_tau[k-1]
            
            # 根据函数参数个数调用
            if len(h_sig.parameters) == 1:
                obs_weight = np.exp(delta_y * h_func(s_flat))
            else:
                obs_weight = np.exp(delta_y * h_func(s_flat, k * Dtau))
                
            U = U * obs_weight
            U = np.maximum(U, 1e-12)
            U = U / np.sum(U)
        
        # 时间演化
        for i in range(Nt):
            t_idx = k * Nt + i
            
            if t_idx >= NtNtau - 1:
                break
                
            # QIEM格式
            # [I - (Δt/2)Δ]u_{n+1} = [I + Δt(p·∇ + q)]u_n
            
            # 左端矩阵
            A_left = I - (Dt/2) * L
            
            # 右端项
            explicit_part = P_diag @ K + Q_diag
            b = (I + Dt * explicit_part).dot(U)
            
            # 求解线性系统
            try:
                U_new = spsolve(A_left, b)
                U = np.maximum(U_new, 1e-12)
            except:
                # 备用方案: 显式欧拉
                U = U + Dt * (L.dot(U) + explicit_part.dot(U))
                U = np.maximum(U, 1e-12)
            
            # 归一化
            U = U / np.sum(U)
            
            # 状态估计
            theta_est[t_idx + 1] = np.sum(U * s_flat)
    
    return theta_est

def simulate_processes_qiem_quasi_implicit(f_func, h_func, sigma_theta, sigma_S, T, Dt, Dtau, seed):
    """数据生成函数，匹配您提供的独立代码"""
    np.random.seed(seed)
    Nt = int(Dtau / Dt)
    Ntau = int(T / Dtau)
    NtNtau = Ntau * Nt
    
    theta_true = np.zeros(NtNtau)
    y_obs = np.zeros(NtNtau)
    
    theta_true[0] = 0.0  # 初始状态，匹配您的代码
    y_obs[0] = 0.0       # 初始观测，匹配您的代码
    
    # 检查函数参数个数
    f_sig = inspect.signature(f_func)
    h_sig = inspect.signature(h_func)
    
    for t in range(1, NtNtau):
        current_time = t * Dt
        
        # 状态方程: dx = f(x)dt + dv
        # 根据函数参数个数调用
        if len(f_sig.parameters) == 1:
            f_val = f_func(theta_true[t-1])
        else:
            f_val = f_func(theta_true[t-1], current_time)
            
        theta_true[t] = theta_true[t-1] + f_val * Dt + sigma_theta * np.sqrt(Dt) * np.random.randn()
        
        # 观测方程: dy = h(x)dt + dw
        # 根据函数参数个数调用
        if len(h_sig.parameters) == 1:
            h_val = h_func(theta_true[t])
        else:
            h_val = h_func(theta_true[t], current_time)
            
        y_obs[t] = y_obs[t-1] + h_val * Dt + sigma_S * np.sqrt(Dt) * np.random.randn()
    
    y_tau = y_obs[::Nt]
    
    return theta_true, y_obs, y_tau

class EKF_QIEM:
    def __init__(self, dt, Dtau, Nt, sigma_theta, sigma_S, f_func):
        self.dt = dt                  # 小时间步
        self.Dtau = Dtau              # 观测间隔
        self.Nt = Nt                  # 每个观测间隔的小步数
        self.f_func = f_func          # 状态方程函数
        self.sigma_theta = sigma_theta# 状态噪声标准差
        self.sigma_S = sigma_S        # 观测噪声标准差
        
        # 过程噪声协方差 (Q = sigma_theta² * dt)
        self.Q = sigma_theta ** 2 * dt
        # 观测噪声协方差 (R = sigma_S² * Dtau)
        self.R = sigma_S ** 2 * Dtau

    def run(self, y_tau, NtNtau):
        """运行EKF"""
        # 初始化状态和协方差
        theta_est = np.zeros(NtNtau)
        P_est = np.zeros(NtNtau)
        
        theta_est[0] = 1.0  # 初始状态（与真实状态一致）
        P_est[0] = 0.1      # 初始协方差
        
        Ntau = len(y_tau)
        
        # 检查函数参数个数
        f_sig = inspect.signature(self.f_func)
        
        for k in range(Ntau):
            current_tau = k * self.Dtau
            t_start = k * self.Nt  # 当前观测间隔的起始时间步
            
            # 观测更新（每个观测间隔一次）
            if k > 0:
                delta_y = y_tau[k] - y_tau[k-1]  # 观测增量
                
                # 状态预测（基于上一间隔的最终状态）
                theta_pred = theta_est[t_start - 1]
                P_pred = P_est[t_start - 1]
                
                # 观测模型线性化（dh/dθ = 1）
                H = 1.0
                
                # 观测预测（z_pred = h(theta_pred) * Dtau）
                z_pred = (theta_pred - 0.5 * self.sigma_S ** 2) * self.Dtau
                
                # 创新和创新协方差
                innovation = delta_y - z_pred
                S = H * P_pred * H + self.R
                
                # 卡尔曼增益
                K = P_pred * H / (S + 1e-10)  # 避免除以零
                
                # 状态和协方差更新
                theta_est[t_start] = theta_pred + K * innovation
                P_est[t_start] = (1 - K * H) * P_pred
            else:
                # 第一个观测间隔，直接使用初始状态
                theta_est[t_start] = theta_est[0]
                P_est[t_start] = P_est[0]
            
            # 间隔内的小步预测（无观测更新）
            for i in range(1, self.Nt):
                t_idx = t_start + i
                if t_idx >= NtNtau:
                    break
                
                inner_time = current_tau + i * self.dt
                
                # 状态方程线性化（df/dθ = 0，因为f_func不依赖theta）
                F = 0.0
                
                # 状态预测
                # 根据函数参数个数调用
                if len(f_sig.parameters) == 1:
                    f_val = self.f_func(theta_est[t_idx - 1])
                else:
                    f_val = self.f_func(theta_est[t_idx - 1], inner_time)
                    
                theta_est[t_idx] = theta_est[t_idx - 1] + f_val * self.dt
                
                # 协方差预测
                P_est[t_idx] = F * P_est[t_idx - 1] * F + self.Q
        
        return theta_est

class ParticleFilter_QIEM:
    def __init__(self, f_func, h_func, dt, Dtau, sigma_theta, sigma_S, n_particles=100):
        self.f_func = f_func
        self.h_func = h_func
        self.dt = dt
        self.Dtau = Dtau
        self.Nt = int(Dtau / dt)
        self.sigma_theta = sigma_theta
        self.sigma_S = sigma_S
        self.n_particles = n_particles
        self.particles = None
        self.weights = None
        self.rng = np.random.RandomState()  # 使用独立的随机数生成器

    def initialize(self, x0=1.0, std=0.5):
        # 与参考代码一致的初始化
        self.particles = x0 + std * self.rng.randn(self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self, t):
        # 检查函数参数个数
        f_sig = inspect.signature(self.f_func)
        
        for i in range(self.n_particles):
            if len(f_sig.parameters) == 1:
                f_val = self.f_func(self.particles[i])
            else:
                f_val = self.f_func(self.particles[i], t)
                
            self.particles[i] += f_val * self.dt + self.sigma_theta * np.sqrt(self.dt) * self.rng.randn()

    def update(self, y_prev, y_curr):
        delta_y = y_curr - y_prev
        
        # 检查函数参数个数
        h_sig = inspect.signature(self.h_func)
        
        # 使用参考代码的似然计算方法
        weight_sum = 0.0
        for i in range(self.n_particles):
            if len(h_sig.parameters) == 1:
                expected_obs = self.h_func(self.particles[i]) * self.Dtau
            else:
                expected_obs = self.h_func(self.particles[i], 0) * self.Dtau
                
            log_likelihood = -0.5 * (delta_y - expected_obs)**2 / (self.sigma_S**2 * self.Dtau + 1e-10)
            log_likelihood = np.clip(log_likelihood, -30, 30)  # 重要：clip防止数值溢出
            self.weights[i] *= np.exp(log_likelihood)
            weight_sum += self.weights[i]
        
        # 归一化
        if weight_sum > 1e-300:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

    def resample(self):
        # 使用有效粒子数判断是否重采样
        Neff = 1.0 / np.sum(self.weights**2)
        if Neff < self.n_particles / 3:
            indices = self.systematic_resample(self.weights)
            self.particles = self.particles[indices]
            # 添加扰动，与参考代码一致
            jitter_std = 0.01 * self.sigma_theta * np.sqrt(self.dt)
            self.particles += jitter_std * self.rng.randn(self.n_particles)
            self.weights = np.ones(self.n_particles) / self.n_particles

    def systematic_resample(self, weights):
        N = len(weights)
        weights = np.maximum(weights, 1e-300)
        weights /= np.sum(weights)
        
        # 使用独立的随机数生成器
        positions = (self.rng.random() + np.arange(N)) / N
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0
        
        indices = np.zeros(N, dtype=int)
        j = 0
        for i in range(N):
            while positions[i] > cumulative_sum[j] and j < N-1:
                j += 1
            indices[i] = j
        return indices

    def get_estimate(self):
        return np.average(self.particles, weights=self.weights)

    def run(self, y_tau, NtNtau, seed):
        # 设置随机种子
        self.rng = np.random.RandomState(seed)
        
        est = np.zeros(NtNtau)
        self.initialize()
        est[0] = self.get_estimate()
        
        Ntau = len(y_tau)
        resample_count = 0  # 跟踪重采样次数
        
        for k in range(Ntau):
            current_tau = k * self.Dtau
            t_start = k * self.Nt
            
            if k > 0:
                self.update(y_tau[k-1], y_tau[k])
                self.resample()
            
            est[t_start] = self.get_estimate()
            
            # 间隔内的预测
            for i in range(1, self.Nt):
                t_idx = t_start + i
                if t_idx >= NtNtau:
                    break
                
                inner_time = current_tau + i * self.dt
                self.predict(inner_time)
                est[t_idx] = self.get_estimate()
        
        print(f"Particle Filter resampled {resample_count} times")
        return est

# ==============================
# Streamlit Main Interface
# ==============================
st.set_page_config(page_title="Filtering Methods Comparison", layout="wide")
st.title("Stochastic Filtering: Hermite Spectral vs Finite Difference")

# Method selection
method = st.radio(
    "Select Filtering Method",
    ["Hermite Spectral Method", "Finite Difference Method (QIEM)"]
)

# Initialize session state
if "results" not in st.session_state:
    st.session_state["results"] = {
        "true_state": None,
        "main_estimate": None,
        "pf_estimate": None,
        "ekf_estimate": None,
        "main_mse": None,
        "pf_mse": None,
        "ekf_mse": None,
        "main_time": None,
        "pf_time": None,
        "ekf_time": None,
        "time_axis": None,
        "seed": None,
        "y_tau": None,
        "y_obs": None,
        "NtNtau": None,
        "dt": None,  # Hermite用小写dt
        "Dt": None,  # QIEM用大写Dt
        "Dtau": None,
        "sigma_theta": None,
        "sigma_S": None,
        "Nt": None,
        "x0": None
    }

# 初始化粒子滤波运行状态
if "pf_run" not in st.session_state:
    st.session_state["pf_run"] = False
if "ekf_run" not in st.session_state:
    st.session_state["ekf_run"] = False

# 初始化求解器状态
if "solver_type" not in st.session_state:
    st.session_state["solver_type"] = "Time-Varying"

# Sidebar parameters
with st.sidebar:
    st.header("Parameters")
    
    if method == "Hermite Spectral Method":
        # Numerical parameters
        N = st.number_input("N (Spectral Basis Order)", value=12, min_value=1, max_value=50, step=1)
        M_quad = st.number_input("M_quad (Quadrature Points)", value=100, min_value=20, max_value=200, step=10)
        LT = st.number_input("LT (Total Steps)", value=200, min_value=50, max_value=500, step=50)
        dt = st.number_input("dt (Time Step)", value=0.02, min_value=0.001, max_value=0.1, step=0.001)
        x0 = st.number_input("x0 (Initial State)", value=3.0, min_value=-5.0, max_value=5.0, step=0.1)
        y0 = st.number_input("y0 (Initial Observation)", value=0.0, min_value=-5.0, max_value=5.0, step=0.1)
        scale = st.number_input("scale (Basis Scale)", value=0.3, min_value=0.1, max_value=2.0, step=0.1)
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=1000, step=1)
        
        # Function parameters
        st.subheader("State Equation")
        f_str = st.text_input(
            "Drift Function f(x,t)",
            value="0.2*x + 0.5*np.sin(3*t)"
        )
        sigma_x_str = st.text_input(
            "Diffusion Coefficient σ_x(t)",
            value="0.5*(1.0 + 0.5*np.sin(t))"
        )
        
        st.subheader("Observation Equation")
        h_str = st.text_input(
            "Observation Function h(x,t)",
            value="x**2 + 1"
        )
        sigma_y_str = st.text_input(
            "Observation Noise σ_y(t)",
            value="0.5*(1.0 + 0.2*np.cos(t*0.5))"
        )
        
        params = {
            "N": N, "M_quad": M_quad, "LT": LT, "dt": dt,
            "x0": x0, "y0": y0, "scale": scale, "seed": seed,
            "f_str": f_str, "sigma_x_str": sigma_x_str,
            "h_str": h_str, "sigma_y_str": sigma_y_str
        }
    
    else:  # Finite Difference Method (QIEM)
        # 求解器选择
        st.subheader("Solver Selection")
        solver_type = st.selectbox(
            "QIEM Solver Type",
            ["Time-Varying", "Time-Invariant"],
            help="Time-Varying: Original Streamlit implementation. Time-Invariant: Matches the standalone code for cubic sensor model."
        )
        
        # 检查求解器类型是否变化，如果变化则更新默认值
        if solver_type != st.session_state.get("solver_type"):
            st.session_state["solver_type"] = solver_type
        
        # Numerical parameters
        SEED = st.number_input("Random Seed", value=13, min_value=0, max_value=1000, step=1)
        T = st.number_input("T (Total Time)", value=20.0, min_value=5.0, max_value=50.0, step=1.0)
        Dt = st.number_input("Dt (Small Time Step)", value=0.001, min_value=0.0001, max_value=1.0, step=0.0001,format="%.3f")
        Dtau = st.number_input("Dtau (Observation Interval)", value=0.01, min_value=0.001, max_value=0.01, step=0.001,format="%.3f")
        Ds = st.number_input("Ds (Spatial Step)", value=0.5, min_value=0.01, max_value=1.0, step=0.01,format="%.3f")
        R = st.number_input("R (Spatial Range)", value=3.5, min_value=1.0, max_value=50.0, step=0.5)
        
        # 计算衍生参数
        Nt = int(Dtau / Dt)
        Ntau = int(T / Dtau)
        NtNtau = Ntau * Nt
        
        st.write(f"Derived parameters: Nt = {Nt}, Ntau = {Ntau}, Total steps = {NtNtau}")
        
        # 根据求解器类型设置不同的默认参数和函数
        if solver_type == "Time-Varying":
            # Time-Varying 的默认参数
            default_f_str = "0.1 + 0.3*np.sin(t)"
            default_h_str = "theta - 0.5 * sigma_S ** 2"
            default_sigma_theta_str = "0.70"
            default_sigma_S = "0.50"
        else:  # Time-Invariant
            # Time-Invariant 的默认参数 - 与独立代码一致
            default_f_str = "np.cos(theta)"  # 修正为 np.cos(theta)
            default_h_str = "theta**3"
            default_sigma_theta_str = "1.0"      # 修正为 1.0 以匹配独立代码
            default_sigma_S = "1.0"          # 修正为 1.0 以匹配独立代码
        
        # 状态方程和观测方程的漂移项输入
        st.subheader("State Equation")
        f_str = st.text_input(
            "Drift Function f(theta,t)",
            value=default_f_str,
            key="f_str_qiem"
        )
        
        st.subheader("Observation Equation") 
        h_str = st.text_input(
            "Observation Function h(theta)",
            value=default_h_str,
            key="h_str_qiem"
        )
        
        # 模型参数
        st.subheader("Model Parameters")
        sigma_theta_str = st.text_input("σ_theta (State Noise) as function of t", value=default_sigma_theta_str)
        sigma_S = float(st.text_input("σ_S (Observation Noise)", value=default_sigma_S))
        
        # 稳定性检查
        try:
            # 尝试评估f函数来获取最大p值
            f_func_test = safe_eval_func(f_str, ["theta", "t"])
            # 在空间域上采样检查
            theta_test = np.linspace(-R, R, 100)
            t_test = 0
            p_vals = -np.array([f_func_test(theta, t_test) for theta in theta_test])
            p_max = np.max(np.abs(p_vals))
            
            h_func_test = safe_eval_func(h_str, ["theta"])
            q_vals = -0.5 * np.array([h_func_test(theta) ** 2 for theta in theta_test])
            q_max = np.max(q_vals)
            
            is_stable = check_stability_condition(p_max, q_max, Ds, Dt)
            if not is_stable:
                st.warning("Stability condition not satisfied! Results may be unreliable.")
        except:
            st.info("Cannot perform stability check with user-defined functions.")
        
        params = {
            "SEED": SEED, "T": T, "Dt": Dt, "Dtau": Dtau, 
            "Ds": Ds, "R": R, "sigma_theta_str": sigma_theta_str,
            "sigma_S": sigma_S, "Nt": Nt, "Ntau": Ntau,
            "NtNtau": NtNtau, "f_str": f_str, "h_str": h_str,
            "solver_type": solver_type
        }

# Main content
col1, col2 = st.columns([3, 1])
with col1:
    st.header("Filtering Results")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("State", fontsize=12)
    ax.set_title("True State vs Estimated States", fontsize=14)
    ax.grid(True, alpha=0.3)
    placeholder = st.empty()
    placeholder.pyplot(fig)

with col2:
    st.header("Actions")
    run_btn = st.button("Run Simulation & Filtering")
    
    # 粒子滤波设置 - 默认100个粒子，用户可以调整
    st.subheader("Particle Filter")
    pf_n_particles = st.number_input("Number of Particles", value=100, min_value=1, max_value=500, step=1, key="pf_n_particles")
    run_pf_btn = st.button("Run Particle Filter")
    
    # EKF设置
    st.subheader("Extended Kalman Filter")
    run_ekf_btn = st.button("Run Extended Kalman Filter")
    
    st.header("Performance Metrics")
    metrics_placeholder = st.empty()
    # 初始化空的指标显示
    metrics_placeholder.markdown("""
    <div style="background-color:#f0f0f0;padding:10px;border-radius:5px">
        <p>Please run the main filtering first</p>
    </div>
    """, unsafe_allow_html=True)

# 初始化metrics_html变量
metrics_html = ""

# Run main filtering
if run_btn:
    with st.spinner("Running simulation and filtering..."):
        if method == "Hermite Spectral Method":
            # Parse functions
            f = safe_eval_func(params["f_str"], ["x", "t"])
            sigma_x_func = safe_eval_func(params["sigma_x_str"], ["t"])
            h = safe_eval_func(params["h_str"], ["x", "t"])
            sigma_y_func = safe_eval_func(params["sigma_y_str"], ["t"])
            
            if None in [f, sigma_x_func, h, sigma_y_func]:
                st.error("Please check function expressions")
                st.stop()
            
            # Initialize model
            model = StochasticModelHermite(f, h, sigma_x_func, sigma_y_func)
            basis = HermiteBasis(params["N"], params["scale"])
            op = GalerkinOperator(model, basis, M_quad=params["M_quad"])
            filt_adaptive = AdaptiveMemorylessFilter(model, op, params["dt"])
            
            # Simulate data
            x_true, y_obs = simulate_state_observation_hermite(
                model, params["LT"], params["dt"], 
                params["x0"], params["y0"], params["seed"]
            )
            time_axis = np.linspace(0, params["LT"] * params["dt"], params["LT"] + 1)
            
            # Run main filter (Adaptive Hermite)
            start_time = time.time()
            mu0 = 1.0
            X_init, W_init = op.get_nodes_weights(mu=mu0)
            rho0 = np.exp(-0.5 * ((X_init - mu0) / (params["scale"]/2))**2)
            rho0_norm = np.sum(rho0 * W_init)
            if rho0_norm > 0:
                rho0 /= rho0_norm
            else:
                rho0 = W_init / np.sum(W_init**2)
            
            est_main, _, _ = filt_adaptive.run(x_true, y_obs, rho0, mu0=mu0)
            main_time = time.time() - start_time
            main_mse = mean_squared_error(x_true[100:], est_main[100:])
            main_rmse = np.sqrt(main_mse)
            main_mae = np.mean(np.abs(x_true[100:] - est_main[100:]))
            main_r2 = r2_score(x_true[100:], est_main[100:])
            
            # Store results（Hermite用小写dt）
            st.session_state["results"] = {
                "true_state": x_true,
                "main_estimate": est_main,
                "pf_estimate": None,
                "ekf_estimate": None,
                "main_mse": main_mse,
                "main_rmse": main_rmse,
                "main_mae": main_mae,
                "main_r2": main_r2,
                "pf_mse": None,
                "ekf_mse": None,
                "main_time": main_time,
                "pf_time": None,
                "ekf_time": None,
                "time_axis": time_axis,
                "seed": params["seed"],
                "y_obs": y_obs,
                "x0": params["x0"],
                "dt": params["dt"],  # 小写dt
                "Dtau": None,  # Hermite无Dtau
                "sigma_theta": None,
                "sigma_S": None,
                "NtNtau": len(x_true),
                "Nt": None
            }
            # 重置粒子滤波和EKF的运行状态
            st.session_state["pf_run"] = False
            st.session_state["ekf_run"] = False
        
        else:  # Finite Difference Method (QIEM)
            # 使用用户定义的模型函数
            f_func = safe_eval_func(params["f_str"], ["theta", "t"])
            h_func = safe_eval_func(params["h_str"], ["theta"])
            
            # 解析时变状态噪声函数
            sigma_theta_func = safe_eval_func(params["sigma_theta_str"], ["t"])
            if sigma_theta_func is None:
                # 如果解析失败，尝试作为常数处理
                try:
                    sigma_theta_const = float(params["sigma_theta_str"])
                    sigma_theta_func = lambda t: sigma_theta_const
                except:
                    st.error("Please check sigma_theta expression")
                    st.stop()
            
            if None in [f_func, h_func]:
                st.error("Please check function expressions for QIEM method")
                st.stop()
            
            # Initialize
            set_random_seed_qiem(params["SEED"])
            s, n_total, n_internal, L_1d, K_1d = initialize_domain_qiem(
                params["Ds"], params["R"]
            )
            
            # 根据求解器类型选择数据生成方法
            if params["solver_type"] == "Time-Invariant":
                # 使用匹配独立代码的数据生成
                theta_true, y_full, y_tau = simulate_processes_qiem_quasi_implicit(
                    f_func, h_func, sigma_theta_func(0), params["sigma_S"],
                    params["T"], params["Dt"], params["Dtau"], params["SEED"]
                )
            else:
                # 使用修改后的数据生成，支持时变噪声
                theta_true, y_full, y_tau = simulate_processes_qiem(
                    f_func, h_func, sigma_theta_func, params["sigma_S"],
                    params["T"], params["Dt"], params["Dtau"], params["SEED"]
                )
            
            time_axis = np.arange(len(theta_true)) * params["Dt"]
            
            # Run main filter (QIEM)
            start_time = time.time()
            
            if params["solver_type"] == "Time-Invariant":
                # 使用准隐式欧拉方法
                est_main = run_qiem_filter_quasi_implicit(
                    s, n_total, n_internal, L_1d, K_1d,
                    y_tau, f_func, h_func, params["Ds"], params["Dt"], params["Dtau"],
                    sigma_theta_func(0), params["sigma_S"]
                )
            else:
                # 使用修改后的DST方法，支持时变噪声
                est_main = run_qiem_filter_dst_1d(
                    s, n_total, n_internal, L_1d, K_1d,
                    y_tau, f_func, h_func, params["Ds"], params["Dt"], params["Dtau"],
                    sigma_theta_func
                )
            
            main_time = time.time() - start_time
            main_mse = mean_squared_error(theta_true, est_main)
            main_rmse = np.sqrt(main_mse)
            main_mae = np.mean(np.abs(theta_true - est_main))
            main_r2 = r2_score(theta_true, est_main)
            
            # Store results（QIEM用大写Dt）
            st.session_state["results"] = {
                "true_state": theta_true,
                "main_estimate": est_main,
                "pf_estimate": None,
                "ekf_estimate": None,
                "main_mse": main_mse,
                "main_rmse": main_rmse,
                "main_mae": main_mae,
                "main_r2": main_r2,
                "pf_mse": None,
                "ekf_mse": None,
                "main_time": main_time,
                "pf_time": None,
                "ekf_time": None,
                "time_axis": time_axis,
                "seed": params["SEED"],
                "y_tau": y_tau,
                "y_obs": None,
                "NtNtau": params["NtNtau"],
                "dt": None,  # QIEM无小写dt
                "Dt": params["Dt"],  # 大写Dt
                "Dtau": params["Dtau"],
                "sigma_theta": sigma_theta_func(0),  # 存储常数值而不是函数
                "sigma_S": params["sigma_S"],
                "Nt": params["Nt"],
                "x0": None
            }
            # 重置粒子滤波和EKF的运行状态
            st.session_state["pf_run"] = False
            st.session_state["ekf_run"] = False
    
    # Update plot with main results
    res = st.session_state["results"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(res["time_axis"], res["true_state"], 'k-', linewidth=2, label='True State', alpha=0.8)
    ax.plot(res["time_axis"], res["main_estimate"], 'r-', linewidth=2, 
            label=f'{method.split(" ")[0]} Estimate (MSE={res["main_mse"]:.4f}, Time={res["main_time"]:.2f}s)', 
            alpha=0.8)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("State", fontsize=12)
    ax.set_title("True State vs Estimated States", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    placeholder.pyplot(fig)
    
    # Update metrics
    metrics_html = f"""
    <div style="background-color:#f0f0f0;padding:10px;border-radius:5px">
        <h4>{method}</h4>
        <p>MSE: {res["main_mse"]:.4f}</p>
        <p>RMSE: {res["main_rmse"]:.4f}</p>
        <p>MAE: {res["main_mae"]:.4f}</p>
        <p>R²: {res["main_r2"]:.4f}</p>
        <p>Runtime: {res["main_time"]:.2f}s</p>
    </div>
    """
    metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)

# Run Particle Filter if selected
if run_pf_btn and st.session_state["results"]["true_state"] is not None:
    # 确保metrics_html已初始化
    if not metrics_html:
        metrics_html = ""
        
    with st.spinner(f"Running Particle Filter with {pf_n_particles} particles..."):
        res = st.session_state["results"]
        
        if method == "Hermite Spectral Method":
            # Parse functions again
            f = safe_eval_func(params["f_str"], ["x", "t"])
            sigma_x_func = safe_eval_func(params["sigma_x_str"], ["t"])
            h = safe_eval_func(params["h_str"], ["x", "t"])
            sigma_y_func = safe_eval_func(params["sigma_y_str"], ["t"])
            model = StochasticModelHermite(f, h, sigma_x_func, sigma_y_func)
            
            # Run PF（使用Hermite的小写dt）
            pf = ParticleFilter_Hermite(model, res["dt"], n_particles=pf_n_particles)
            start_time = time.time()
            est_pf = pf.run(res["y_obs"], res["x0"], res["seed"])
            pf_time = time.time() - start_time
            pf_mse = mean_squared_error(res["true_state"][100:], est_pf[100:])
            pf_rmse = np.sqrt(pf_mse)
            pf_mae = np.mean(np.abs(res["true_state"][100:] - est_pf[100:]))
            pf_r2 = r2_score(res["true_state"][100:], est_pf[100:])
        
        else:
            # 使用用户定义的模型函数
            f_func = safe_eval_func(params["f_str"], ["theta", "t"])
            h_func = safe_eval_func(params["h_str"], ["theta"])
            
            # Run PF（使用QIEM的大写Dt和Dtau）
            pf = ParticleFilter_QIEM(
                f_func, h_func, res["Dt"], res["Dtau"], 
                res["sigma_theta"], res["sigma_S"],
                n_particles=pf_n_particles
            )
            start_time = time.time()
            est_pf = pf.run(res["y_tau"], res["NtNtau"], res["seed"])
            pf_time = time.time() - start_time
            pf_mse = mean_squared_error(res["true_state"], est_pf)
            pf_rmse = np.sqrt(pf_mse)
            pf_mae = np.mean(np.abs(res["true_state"] - est_pf))
            pf_r2 = r2_score(res["true_state"], est_pf)
        
        # Update session state
        st.session_state["results"]["pf_estimate"] = est_pf
        st.session_state["results"]["pf_mse"] = pf_mse
        st.session_state["results"]["pf_rmse"] = pf_rmse
        st.session_state["results"]["pf_mae"] = pf_mae
        st.session_state["results"]["pf_r2"] = pf_r2
        st.session_state["results"]["pf_time"] = pf_time
        st.session_state["pf_run"] = True
        
        # Update plot
        res = st.session_state["results"]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(res["time_axis"], res["true_state"], 'k-', linewidth=2, label='True State', alpha=0.8)
        ax.plot(res["time_axis"], res["main_estimate"], 'r-', linewidth=2, 
                label=f'{method.split(" ")[0]} Estimate (MSE={res["main_mse"]:.4f}, Time={res["main_time"]:.2f}s)', 
                alpha=0.8)
        ax.plot(res["time_axis"], res["pf_estimate"], 'g--', linewidth=2, 
                label=f'Particle Filter (N={pf_n_particles}) (MSE={res["pf_mse"]:.4f}, Time={res["pf_time"]:.2f}s)', 
                alpha=0.8)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("State", fontsize=12)
        ax.set_title("True State vs Estimated States", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        placeholder.pyplot(fig)
        
        # Update metrics
        metrics_html = f"""
        <div style="background-color:#f0f0f0;padding:10px;border-radius:5px">
            <h4>{method}</h4>
            <p>MSE: {res["main_mse"]:.4f}</p>
            <p>RMSE: {res["main_rmse"]:.4f}</p>
            <p>MAE: {res["main_mae"]:.4f}</p>
            <p>R²: {res["main_r2"]:.4f}</p>
            <p>Runtime: {res["main_time"]:.2f}s</p>
        </div>
        <div style="background-color:#e8f5e9;padding:10px;border-radius:5px;margin-top:10px">
            <h4>Particle Filter ({pf_n_particles} particles)</h4>
            <p>MSE: {res["pf_mse"]:.4f}</p>
            <p>RMSE: {res["pf_rmse"]:.4f}</p>
            <p>MAE: {res["pf_mae"]:.4f}</p>
            <p>R²: {res["pf_r2"]:.4f}</p>
            <p>Runtime: {res["pf_time"]:.2f}s</p>
        </div>
        """
        metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
elif run_pf_btn:
    st.warning("Please run the main filtering first before running Particle Filter")

# Run EKF if selected
if run_ekf_btn and st.session_state["results"]["true_state"] is not None:
    # 确保metrics_html已初始化
    if not metrics_html:
        metrics_html = ""
        
    with st.spinner("Running Extended Kalman Filter..."):
        res = st.session_state["results"]
        
        if method == "Hermite Spectral Method":
            # Parse functions again
            f = safe_eval_func(params["f_str"], ["x", "t"])
            sigma_x_func = safe_eval_func(params["sigma_x_str"], ["t"])
            h = safe_eval_func(params["h_str"], ["x", "t"])
            sigma_y_func = safe_eval_func(params["sigma_y_str"], ["t"])
            model = StochasticModelHermite(f, h, sigma_x_func, sigma_y_func)
            
            # Run EKF（使用Hermite的小写dt）
            ekf = EKF_Hermite(model, res["dt"])
            start_time = time.time()
            est_ekf = ekf.run(res["y_obs"], res["x0"], np.array([[0.1]]))
            ekf_time = time.time() - start_time
            ekf_mse = mean_squared_error(res["true_state"][100:], est_ekf[100:])
            ekf_rmse = np.sqrt(ekf_mse)
            ekf_mae = np.mean(np.abs(res["true_state"][100:] - est_ekf[100:]))
            ekf_r2 = r2_score(res["true_state"][100:], est_ekf[100:])
        
        else:
            # 使用用户定义的模型函数
            f_func = safe_eval_func(params["f_str"], ["theta", "t"])
            
            # Run EKF（使用QIEM的大写Dt和Dtau）
            ekf = EKF_QIEM(
                res["Dt"], res["Dtau"], res["Nt"],
                res["sigma_theta"], res["sigma_S"], f_func
            )
            start_time = time.time()
            est_ekf = ekf.run(res["y_tau"], res["NtNtau"])
            ekf_time = time.time() - start_time
            ekf_mse = mean_squared_error(res["true_state"], est_ekf)
            ekf_rmse = np.sqrt(ekf_mse)
            ekf_mae = np.mean(np.abs(res["true_state"] - est_ekf))
            ekf_r2 = r2_score(res["true_state"], est_ekf)
        
        # Update session state
        st.session_state["results"]["ekf_estimate"] = est_ekf
        st.session_state["results"]["ekf_mse"] = ekf_mse
        st.session_state["results"]["ekf_rmse"] = ekf_rmse
        st.session_state["results"]["ekf_mae"] = ekf_mae
        st.session_state["results"]["ekf_r2"] = ekf_r2
        st.session_state["results"]["ekf_time"] = ekf_time
        st.session_state["ekf_run"] = True
        
        # Update plot
        res = st.session_state["results"]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(res["time_axis"], res["true_state"], 'k-', linewidth=2, label='True State', alpha=0.8)
        ax.plot(res["time_axis"], res["main_estimate"], 'r-', linewidth=2, 
                label=f'{method.split(" ")[0]} Estimate (MSE={res["main_mse"]:.4f}, Time={res["main_time"]:.2f}s)', 
                alpha=0.8)
        
        if st.session_state["pf_run"]:
            ax.plot(res["time_axis"], res["pf_estimate"], 'g--', linewidth=2, 
                    label=f'Particle Filter (N={pf_n_particles}) (MSE={res["pf_mse"]:.4f}, Time={res["pf_time"]:.2f}s)', 
                    alpha=0.8)
        
        ax.plot(res["time_axis"], res["ekf_estimate"], 'b:', linewidth=2, 
                label=f'EKF Estimate (MSE={res["ekf_mse"]:.4f}, Time={res["ekf_time"]:.2f}s)', 
                alpha=0.8)
        
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("State", fontsize=12)
        ax.set_title("True State vs Estimated States", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        placeholder.pyplot(fig)
        
        # Update metrics
        metrics_html = f"""
        <div style="background-color:#f0f0f0;padding:10px;border-radius:5px">
            <h4>{method}</h4>
            <p>MSE: {res["main_mse"]:.4f}</p>
            <p>RMSE: {res["main_rmse"]:.4f}</p>
            <p>MAE: {res["main_mae"]:.4f}</p>
            <p>R²: {res["main_r2"]:.4f}</p>
            <p>Runtime: {res["main_time"]:.2f}s</p>
        </div>
        """
        
        if st.session_state["pf_run"]:
            metrics_html += f"""
            <div style="background-color:#e8f5e9;padding:10px;border-radius:5px;margin-top:10px">
                <h4>Particle Filter ({pf_n_particles} particles)</h4>
                <p>MSE: {res["pf_mse"]:.4f}</p>
                <p>RMSE: {res["pf_rmse"]:.4f}</p>
                <p>MAE: {res["pf_mae"]:.4f}</p>
                <p>R²: {res["pf_r2"]:.4f}</p>
                <p>Runtime: {res["pf_time"]:.2f}s</p>
            </div>
            """
        
        metrics_html += f"""
        <div style="background-color:#e3f2fd;padding:10px;border-radius:5px;margin-top:10px">
            <h4>Extended Kalman Filter</h4>
            <p>MSE: {res["ekf_mse"]:.4f}</p>
            <p>RMSE: {res["ekf_rmse"]:.4f}</p>
            <p>MAE: {res["ekf_mae"]:.4f}</p>
            <p>R²: {res["ekf_r2"]:.4f}</p>
            <p>Runtime: {res["ekf_time"]:.2f}s</p>
        </div>
        """
        metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
elif run_ekf_btn:
    st.warning("Please run the main filtering first before running EKF")

# 显示当前状态下的图表（如果已经运行了某些滤波器）
if st.session_state["results"]["true_state"] is not None:
    res = st.session_state["results"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(res["time_axis"], res["true_state"], 'k-', linewidth=2, label='True State', alpha=0.8)
    ax.plot(res["time_axis"], res["main_estimate"], 'r-', linewidth=2, 
            label=f'{method.split(" ")[0]} Estimate (MSE={res["main_mse"]:.4f}, Time={res["main_time"]:.2f}s)', 
            alpha=0.8)
    
    if st.session_state["pf_run"]:
        ax.plot(res["time_axis"], res["pf_estimate"], 'g--', linewidth=2, 
                label=f'Particle Filter (N={pf_n_particles}) (MSE={res["pf_mse"]:.4f}, Time={res["pf_time"]:.2f}s)', 
                alpha=0.8)
    
    if st.session_state["ekf_run"]:
        ax.plot(res["time_axis"], res["ekf_estimate"], 'b:', linewidth=2, 
                label=f'EKF Estimate (MSE={res["ekf_mse"]:.4f}, Time={res["ekf_time"]:.2f}s)', 
                alpha=0.8)
    
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("State", fontsize=12)
    ax.set_title("True State vs Estimated States", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    placeholder.pyplot(fig)

# Add download button for results
if st.session_state["results"]["true_state"] is not None:
    res = st.session_state["results"]
    results_data = np.column_stack([
        res["time_axis"],
        res["true_state"],
        res["main_estimate"],
        res["pf_estimate"] if res["pf_estimate"] is not None else np.full_like(res["time_axis"], np.nan),
        res["ekf_estimate"] if res["ekf_estimate"] is not None else np.full_like(res["time_axis"], np.nan)
    ])
    
    headers = "Time,True_State,Main_Estimate"
    if res["pf_estimate"] is not None:
        headers += ",PF_Estimate"
    if res["ekf_estimate"] is not None:
        headers += ",EKF_Estimate"
    
    np.savetxt("filtering_results.csv", results_data, delimiter=",", header=headers, comments="")
    st.download_button(
        label="Download Results (CSV)",
        data=open("filtering_results.csv", "rb"),
        file_name="filtering_results.csv",
        mime="text/csv"
    )