from scipy.integrate import solve_ivp
from r8hermt import r8herm_interpolation
import numpy as np
import time

temp_path = "/home/zkg/h1_scan/temp"
spline_data = np.load(f"{temp_path}/spline_data.npz")

bp_spline = spline_data['bp_spline']
br_spline = spline_data['br_spline']
bz_spline = spline_data['bz_spline']
r_grid = spline_data['r_grid']
z_grid = spline_data['z_grid']
phiarc_grid = spline_data['phiarc_grid']
phi_arcrange = phiarc_grid[:,0,0] # phi range in first dimesion
r_range = r_grid[0,:,0]
z_range = z_grid[0,0,:]
delta_phi = phiarc_grid[1,0,0] - phiarc_grid[0,0,0]
print("Spline data loaded")

phi_arcrange = phiarc_grid[:,0,0] # phi range in first dimesion
r_range = r_grid[0,:,0]
z_range = z_grid[0,0,:]
delta_phi = phiarc_grid[1,0,0] - phiarc_grid[0,0,0]

coil_currents = {
    'tf_1': 50, 'tf_2': 50, 'tf_3': 50, 'tf_4': 50,
    'tf_5': 50, 'tf_6': 50, 'pf': 5, 'hw': 5,
    'ovf': -80, 'ivf': -40
}
extcur = np.array(list(coil_currents.values())) 

bp_spline_extcur = bp_spline * extcur[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] 
bp_spline_extcur = np.sum(bp_spline_extcur, axis=0)
br_spline_extcur = br_spline * extcur[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
br_spline_extcur = np.sum(br_spline_extcur, axis=0)
bz_spline_extcur = bz_spline * extcur[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
bz_spline_extcur = np.sum(bz_spline_extcur, axis=0)

time_start = time.time()
bp_test = r8herm_interpolation(0, 1.25, 0, phi_arcrange, r_range, z_range, bp_spline_extcur)
time_end = time.time()
print(f"Interpolation done in {time_end - time_start:.3f} s")

def trace_single_fieldlines(R_start, Z_start, phi_start, phi_arcrange, r_range, z_range, nsteps, bp_spline_extcur, br_spline_extcur, bz_spline_extcur):
    R = np.zeros(nsteps)
    Z = np.zeros(nsteps)
    phi = np.zeros(nsteps)
    R[0] = R_start
    Z[0] = Z_start
    phi[0] = phi_start

    def filedline(phi, rz):
        time_start2 = time.time()
        R, Z = rz
        bp_tem = r8herm_interpolation(phi, R, Z, phi_arcrange, r_range, z_range, bp_spline_extcur)
        br_tem = r8herm_interpolation(phi, R, Z, phi_arcrange, r_range, z_range, br_spline_extcur)
        bz_tem = r8herm_interpolation(phi, R, Z, phi_arcrange, r_range, z_range, bz_spline_extcur)
        time_end2 = time.time()
        print(f"Interpolation done in {time_end2 - time_start2:.3f} s")
        return [br_tem/bp_tem, bz_tem/bp_tem]
    for i in range(nsteps):
        time_start = time.time()
        R_temep = R[i]
        Z_temep = Z[i]
        phi_temep = phi[i]
        sol = solve_ivp(filedline, [phi_temep, phi_temep + delta_phi], [R_temep, Z_temep], method='LSODA', rtol=1e-6)
        R[i+1] = sol.y[0][-1]
        Z[i+1] = sol.y[1][-1]
        phi[i+1] = phi[i] + delta_phi
        time_end = time.time()
        print(f"R = {R[i+1]:.3f}, Z = {Z[i+1]:.3f}, phi = {phi[i+1]:.3f}")
        print(f"Step {i+1} done in {time_end - time_start:.3f} s")
    line = np.vstack((R, Z, phi))
    return line

print("Begin fieldline tracing")
nsteps = 100
R_start = 1.25
Z_start = 0
phi_start = 0
time_start = time.time()
test_line = trace_single_fieldlines(R_start, Z_start, phi_start, phi_arcrange, r_range, z_range, nsteps, bp_spline_extcur, br_spline_extcur, bz_spline_extcur)
print("Fieldline tracing done")

