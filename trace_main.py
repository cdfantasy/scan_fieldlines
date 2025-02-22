import numpy as np
from scipy.integrate import solve_ivp
from r8hermt import r8herm_interpolation
import numpy as np
import time

################# Load spline data #################
temp_path = "/home/zkg/h1_scan/temp"
spline_data = np.load(f"{temp_path}/spline_data.npz")

bp_spline = spline_data['bp_spline']
bp_spline = np.asfortranarray(bp_spline)
br_spline = spline_data['br_spline']
br_spline = np.asfortranarray(br_spline)
bz_spline = spline_data['bz_spline']
bz_spline = np.asfortranarray(bz_spline)
r_range = spline_data['r_range']
r_range = np.asfortranarray(r_range)
z_range = spline_data['z_range']
z_range = np.asfortranarray(z_range)
phi_radrange = spline_data['phi_radrange']
phi_radrange = np.asfortranarray(phi_radrange)
delta_phiarc = np.deg2rad(phi_radrange[1] - phi_radrange[0])
delta_phirad = phi_radrange[1] - phi_radrange[0]
delta_phi = delta_phirad
print("Spline data loaded")

################# Test fieldline tracing #################
# Define coil currents
coil_currents = {
    'tf_1': 50, 'tf_2': 50, 'tf_3': 50, 'tf_4': 50,
    'tf_5': 50, 'tf_6': 50, 'pf': 5, 'hw': 5,
    'ovf': -80, 'ivf': -40
}
# Calculate external field
extcur = np.array(list(coil_currents.values())) 
bp_spline_extcur = bp_spline * extcur[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis] 
bp_spline_extcur = np.sum(bp_spline_extcur, axis=0)
br_spline_extcur = br_spline * extcur[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
br_spline_extcur = np.sum(br_spline_extcur, axis=0)
bz_spline_extcur = bz_spline * extcur[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
bz_spline_extcur = np.sum(bz_spline_extcur, axis=0)


def trace_single_fieldlines(R_start, Z_start, phi_start, phi_radrange, r_range, z_range, nsteps, bp_spline_extcur, br_spline_extcur, bz_spline_extcur):
    R = np.zeros(nsteps)
    Z = np.zeros(nsteps)
    phi = np.zeros(nsteps)
    R[0] = R_start
    Z[0] = Z_start
    phi[0] = phi_start

    def filedline(phi, rz):
        R, Z = rz
        bp_tem = r8herm_interpolation(phi, R, Z, phi_radrange, r_range, z_range, bp_spline_extcur)
        br_tem = r8herm_interpolation(phi, R, Z, phi_radrange, r_range, z_range, br_spline_extcur)
        bz_tem = r8herm_interpolation(phi, R, Z, phi_radrange, r_range, z_range, bz_spline_extcur)
        return [br_tem/bp_tem, bz_tem/bp_tem]

    for i in range(nsteps-1):
        R_temep = R[i]
        Z_temep = Z[i]
        phi_temep = phi[i]%phi_radrange[-1]
        sol = solve_ivp(filedline, [phi_temep, phi_temep + delta_phi], [R_temep, Z_temep], method='LSODA', rtol=1e-12)
        R[i+1] = sol.y[0][-1]
        Z[i+1] = sol.y[1][-1]
        phi[i+1] = phi[i] + phi_radrange[1]
 
        if R[i+1] < r_range[0] or R[i+1] > r_range[-1] or Z[i+1] < z_range[0] or Z[i+1] > z_range[-1]:
            print(f"Fieldline out of range at step {i+1}")
            break
        
        # time_end = time.time()
        # print(f"R = {R[i+1]:.3f}, Z = {Z[i+1]:.3f}, phi = {phi[i+1]:.3f}")
        # print(f"Step {i+1} done in {time_end - time_start:.3f} s")

    line = np.vstack((R, Z, phi))
    return line

print("Begin fieldline tracing")
nsteps = 360
R_start = 1.25
Z_start = 0
phi_start = 0
time_start = time.time()
test_line = trace_single_fieldlines(R_start, Z_start, phi_start, phi_radrange, r_range, z_range, nsteps, bp_spline_extcur, br_spline_extcur, bz_spline_extcur)
time_end = time.time()
print(f"Fieldline tracing done in {time_end - time_start:.5f} s")

import matplotlib.pyplot as plt
#plot R-phi 2D plot
plt.figure()
plt.plot(test_line[1], test_line[0])
plt.xlabel('phi')
plt.ylabel('Z')
plt.title('Z-phi plot')
plt.grid()
plt.show()

