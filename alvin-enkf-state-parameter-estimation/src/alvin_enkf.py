#!/usr/bin/env python3
"""
alvin_enkf.py

Simple workflow:
1) Load CSV (timestamp, depth_m, optional pkt_zvel or vz_est, rho_kg_m3)
2) Optional quick offline fit (least squares) to get initial params
3) Run Ensemble Kalman Filter (EnKF) online to estimate state (z,v) and optionally parameters

Dependencies:
    pip install numpy scipy pandas matplotlib

Author: your friend (fast & practical)
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import sys

# ---------------------------
# USER CONFIG
# ---------------------------
CSV = r"C:\Users\suhan\Downloads\alvin_all.csv"
TIME_COL = "timestamp"
DEPTH_COL = "depth_m"
THR_COL = "pkt_zvel"      # primary thruster cmd column name
THR_FALLBACK = "vz_est"  # fallback (not ideal — if absent, u=0)
RHO_COL = "rho_kg_m3"

M0 = 20400.0   # kg, Alvin dry mass (fixed)
G = 9.80665

DO_OFFLINE_FIT = False     # run cheap offline fit first (recommended)
ENKF_RUN = True           # run EnKF after offline fit
AUGMENT_PARAMS = True     # augment state with parameters (Delta_m, logCd, k_thr, Aref)

# EnKF hyperparams
N_ENSEMBLE = 120
PROCESS_NOISE_STATE = np.array([1e-3, 1e-4])   # for z (m) and v (m/s)
PROCESS_NOISE_PARAM = np.array([1e-1, 1e-3, 1e-1, 1e-1])  # small random walk for params
OBS_NOISE_Z = 0.2  # meters
DT_DEFAULT = 0.1

# ---------------------------
# UTILITIES
# ---------------------------
def parse_time_series(df):
    # robust timestamp parsing: detect epoch numeric or datetime string
    if TIME_COL in df.columns:
        if np.issubdtype(df[TIME_COL].dtype, np.number):
            t = df[TIME_COL].to_numpy(dtype=float)
            t = t - t.min()
        else:
            ts = pd.to_datetime(df[TIME_COL], errors="coerce")
            if ts.isna().all():
                raise ValueError("timestamp parsing failed")
            t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    elif "timestamp_str" in df.columns:
        ts = pd.to_datetime(df["timestamp_str"], errors="coerce")
        t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    else:
        raise ValueError("No timestamp column found")
    return t

def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    t = parse_time_series(df)
    if DEPTH_COL not in df.columns:
        raise ValueError(f"{DEPTH_COL} not found in CSV")
    z = pd.to_numeric(df[DEPTH_COL], errors="coerce").to_numpy(dtype=float)
    # thruster
    if THR_COL in df.columns:
        u = pd.to_numeric(df[THR_COL], errors="coerce").to_numpy(dtype=float)
    elif THR_FALLBACK in df.columns:
        u = pd.to_numeric(df[THR_FALLBACK], errors="coerce").to_numpy(dtype=float)
    else:
        u = np.zeros_like(z)
    # normalize thruster values to [-1, 1] range
    maxu = np.nanmax(np.abs(u)) + 1e-6
    u = u / maxu

    # density
    if RHO_COL in df.columns:
        rho = pd.to_numeric(df[RHO_COL], errors="coerce").to_numpy(dtype=float)
        rho = np.nan_to_num(rho, nan=1027.0)
    else:
        rho = np.full_like(z, 1027.0)
    # mask invalids
    mask = ~np.isnan(t) & ~np.isnan(z)
    t = t[mask]; z = z[mask]; u = u[mask]; rho = rho[mask]
    return t, z, u, rho

# ---------------------------
# ALVIN DYNAMICS (for integrator / EnKF)
# ---------------------------
def dyn_rhs(state, t, params, u_func, rho):
    z, v = state
    Delta_m, logCd, Aref, k_thr = params

    # clamp parameters
    Delta_m = np.clip(Delta_m, -3000, 3000)
    Cd = np.exp(np.clip(logCd, np.log(0.1), np.log(2.0)))
    Aref = np.clip(Aref, 2.0, 10.0)
    k_thr = np.clip(k_thr, -200, 200)

    # clamp velocity
    v = np.clip(v, -1.5, 1.5)  # Alvin rarely goes >1 m/s vertically

    u = float(u_func(t))

    # SAFE drag
    F_drag = 0.5 * rho * Cd * Aref * v * abs(v)
    F_drag = np.clip(F_drag, -5e4, 5e4)  # max ±50 kN drag

    thrust = k_thr * u

    dvdt = (-Delta_m * G - F_drag + thrust) / M0
    dvdt = np.clip(dvdt, -1.0, 1.0)  # limit acceleration

    dzdt = v
    return [dzdt, dvdt]

def rk4_step(state, params, u_func, rho, dt):
    s = np.asarray(state, dtype=float)

    def f(s_, tau):
        return np.array(dyn_rhs(s_, tau, params, u_func, rho), dtype=float)

    k1 = f(s, 0)
    k2 = f(s + 0.5*dt*k1, 0.5*dt)
    k3 = f(s + 0.5*dt*k2, 0.5*dt)
    k4 = f(s + dt*k3, dt)

    s_new = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # numeric safety
    if np.any(~np.isfinite(s_new)):
        s_new = np.nan_to_num(s_new, nan=0.0, posinf=10.0, neginf=-10.0)

    return s_new

# ---------------------------
# Quick offline fit (cheap) to get initial params
# We fit Delta_m and logCd (and optionally Aref, k_thr) by minimizing squared z-errors
# Use coarse integrator and downsample to speed up
# ---------------------------
def offline_fit(t, z, u, rho, initial_guess=None):
    print("Running quick offline fit (least-squares)...")
    # downsample for speed
    step = max(1, int(len(t)/2000))
    t_ds = t[::step]; z_ds = z[::step]; u_ds = u[::step]; rho_ds = rho[::step]

    # interpolation function for u
    def make_u_fn(t_full, u_full):
        def ufunc(tq):
            return np.interp(tq, t_full, u_full)
        return ufunc

    ufunc = make_u_fn(t_ds, u_ds)

    # objective: simulate with params and compute z residuals
    def obj(x):
        # params: Delta_m, logCd, Aref, k_thr
        Delta_m, logCd, Aref, k_thr = x
        params = (Delta_m, logCd, Aref, k_thr)
        # simple forward integrate with RK4 at t_ds
        s = np.array([z_ds[0], 0.0])
        zpred = np.zeros_like(t_ds)
        zpred[0] = s[0]
        for i in range(1, len(t_ds)):
            dt = t_ds[i] - t_ds[i-1]
            # integrate from t[i-1] -> t[i] with nsub steps
            nsub = max(1, int(np.clip(dt/0.1, 1, 10)))
            dt_sub = dt / nsub
            for k in range(nsub):
                t_local = t_ds[i-1] + k*dt_sub
                s = rk4_step(s, params, ufunc, rho_ds[i-1], dt_sub)
            zpred[i] = s[0]
        return np.mean((zpred - z_ds)**2)

    # initial guess
    if initial_guess is None:
        # conservative starting values
        x0 = np.array([0.0, np.log(0.7), 6.0, 0.0])
    else:
        x0 = np.array(initial_guess)

    bounds = [(-5000, 5000), (np.log(0.1), np.log(5.0)), (1.0, 20.0), (-2000, 2000)]
    res = minimize(obj, x0, bounds=bounds, method='L-BFGS-B', options={'maxiter':200})
    if not res.success:
        print("Offline fit did not fully converge, returning initial guess.")
        return x0
    print("Offline fit done. params:", res.x)
    return res.x

# ---------------------------
# EnKF Implementation (augmented state)
# ---------------------------
class EnKF:
    def __init__(self, n_ens, augment_params=True):
        self.n = n_ens
        self.augment = augment_params

    def init_ensemble(self, z0, v0, param0, state_scales, param_scales):
        # state_scales: std dev for z and v initial ensemble spread
        # param_scales: std dev for parameters
        if self.augment:
            # augmented vector: [z, v, Delta_m, logCd, k_thr, Aref]
            self.dim = 2 + 4
        else:
            self.dim = 2
        self.E = np.zeros((self.n, self.dim))
        # initialize
        for i in range(self.n):
            z_s = z0 + np.random.normal(0, state_scales[0])
            v_s = v0 + np.random.normal(0, state_scales[1])
            if self.augment:
                Delta_m0, logCd0, Aref0, k_thr0 = param0
                p = np.array([
                    Delta_m0 + np.random.normal(0, param_scales[0]),
                    logCd0 + np.random.normal(0, param_scales[1]),
                    k_thr0 + np.random.normal(0, param_scales[2]),
                    Aref0 + np.random.normal(0, param_scales[3]),
                ])
                self.E[i, :] = np.hstack([z_s, v_s, p])
            else:
                self.E[i, :] = np.hstack([z_s, v_s])
        self.time = None

    def predict_step(self, dt, ufunc, rho_local):
        rho_local = float(np.clip(rho_local, 1000, 1050))
        # propagate each ensemble member through dynamics for dt
        for i in range(self.n):
            if self.augment:
                z_i, v_i, Delta_m_i, logCd_i, k_thr_i, Aref_i = self.E[i, :]
                params = (Delta_m_i, logCd_i, Aref_i, k_thr_i)
            else:
                z_i, v_i = self.E[i, :]
                # use fixed default params if no augmentation
                params = (self.fixed_params['Delta_m'], self.fixed_params['logCd'],
                          self.fixed_params['Aref'], self.fixed_params['k_thr'])
            # integrate small steps for stability
            nsteps = max(1, int(np.ceil(dt/0.1)))
            dt_sub = dt / nsteps
            s = np.array([z_i, v_i])
            for k in range(nsteps):
                t_local = (self.time or 0.0) + k*dt_sub
                s = rk4_step(s, params, ufunc, rho_local, dt_sub)
            # write back
            # final NaN cleanup safety
            if np.any(~np.isfinite(s)):
                s = np.nan_to_num(s, nan=0.0, posinf=10.0, neginf=-10.0)
            if self.augment:
                # parameters random-walk
                self.E[i, 0:2] = s
                self.E[i, 2] += np.random.normal(0, PROCESS_NOISE_PARAM[0])
                self.E[i, 3] += np.random.normal(0, PROCESS_NOISE_PARAM[1])
                self.E[i, 4] += np.random.normal(0, PROCESS_NOISE_PARAM[2])
                self.E[i, 5] += np.random.normal(0, PROCESS_NOISE_PARAM[3])
            else:
                self.E[i, 0:2] = s
        self.time = (self.time or 0.0) + dt

    def analysis_step(self, z_meas, R_z):
        # observation operator H maps augmented state to z (depth)
        # H = [1, 0, 0, 0, 0, 0] for augmented; else [1,0]
        n = self.n
        # ensemble matrix shape (dim, n)
        X = self.E.T  # dim x n
        X = np.nan_to_num(X, nan=0.0, posinf=1e3, neginf=-1e3)
        x_mean = np.mean(X, axis=1, keepdims=True)  # dim x 1
        Xc = X - x_mean  # dim x n
        # observation ensemble
        if self.augment:
            H = np.zeros((1, self.dim))
            H[0,0] = 1.0
        else:
            H = np.array([[1.0, 0.0]])
        Y = H @ X  # 1 x n
        Y = np.nan_to_num(Y, nan=0.0, posinf=1e3, neginf=-1e3)
        y_mean = np.mean(Y, axis=1, keepdims=True)
        Yc = Y - y_mean
        # covariance
        PfHT = (Xc @ Yc.T) / (n-1)  # dim x 1
        S = (Yc @ Yc.T) / (n-1) + R_z  # 1 x 1
        K = PfHT @ np.linalg.inv(S)  # dim x 1
        # update ensemble members with perturbed observations
        for i in range(n):
            y_pert = z_meas + np.random.normal(0, math.sqrt(R_z))
            innovation = y_pert - Y[0, i]
            X[:, i:i+1] = X[:, i:i+1] + K * innovation
        # write back
        self.E = X.T

    def get_posterior_mean(self):
        return np.mean(self.E, axis=0)

# ---------------------------
# MAIN FLOW
# ---------------------------
def main():
    print("Loading CSV...")
    t, z, u, rho = load_csv(CSV)
    print(f"Loaded {len(t)} samples; duration {t[-1]-t[0]:.1f}s")

    # prepare u function (interpolant)
    def ufunc(tq): return float(np.interp(tq, t, u))

    # offline fit for initial guess
    if DO_OFFLINE_FIT:
        x_opt = offline_fit(t, z, u, rho, initial_guess=None)
        Delta_m0, logCd0, Aref0, k_thr0 = x_opt
    else:
        # GOOD PHYSICAL DEFAULTS (stable)
        Delta_m0 = 500          # slightly heavy vehicle
        logCd0   = math.log(0.7)
        Aref0    = 5.0
        k_thr0   = 0.0          # until thruster is actually trusted

    print("Initial params (Delta_m, logCd, Aref, k_thr):", Delta_m0, logCd0, Aref0, k_thr0)

    if not ENKF_RUN:
        return

    # EnKF init
    enkf = EnKF(N_ENSEMBLE, augment_params=AUGMENT_PARAMS)
    z0 = z[0]
    v0 = 0.0
    param0 = (Delta_m0, logCd0, Aref0, k_thr0)
    enkf.init_ensemble(z0, v0, param0, state_scales=[0.1, 0.05], param_scales=[50.0, 0.1, 100.0, 1.0])

    # optional save
    z_est = np.zeros_like(z)
    v_est = np.zeros_like(z)
    params_est = np.zeros((len(z), 4)) if AUGMENT_PARAMS else None

    # run through time and assimilate at measurement points
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        # predict
        enkf.predict_step(dt, ufunc, rho[i-1])
        # analysis using measurement z[i]
        enkf.analysis_step(z[i], R_z=OBS_NOISE_Z**2)
        # record posterior mean
        pm = enkf.get_posterior_mean()
        z_est[i] = pm[0]
        v_est[i] = pm[1]
        if AUGMENT_PARAMS:
            params_est[i, :] = pm[2:6]

    # plot results
    plt.figure(figsize=(10,6))
    plt.plot(t, z, label='obs z', linewidth=0.7)
    plt.plot(t, z_est, label='enkf z_est', linewidth=1.0)
    plt.gca().invert_yaxis()
    plt.legend(); plt.xlabel('time (s)'); plt.ylabel('depth (m)')
    plt.title('EnKF depth tracking')
    plt.grid(True)
    plt.show()

    if AUGMENT_PARAMS:
        plt.figure(figsize=(10,6))
        plt.subplot(2,2,1); plt.plot(t, params_est[:,0]); plt.title("Delta_m (kg)"); plt.grid(True)
        plt.subplot(2,2,2); plt.plot(t, params_est[:,1]); plt.title("logCd"); plt.grid(True)
        plt.subplot(2,2,3); plt.plot(t, params_est[:,2]); plt.title("k_thr"); plt.grid(True)
        plt.subplot(2,2,4); plt.plot(t, params_est[:,3]); plt.title("Aref"); plt.grid(True)
        plt.tight_layout(); plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
