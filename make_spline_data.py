import matplotlib.pyplot as plt
import numpy as np
from simsopt.field import MGrid
from r8hermt import r8herm_spline
import time

# Close all previous plots
plt.close('all')

# Define current values for coils
coil_currents = {
    'tf_1': 50, 'tf_2': 50, 'tf_3': 50, 'tf_4': 50,
    'tf_5': 50, 'tf_6': 50, 'pf': 5, 'hw': 5,
    'ovf': -80, 'ivf': -40
}
extcur = np.array(list(coil_currents.values()))

# File paths
input_path = "/home/zkg/h1_scan/input"
mgrid_filename = 'mgrid_h1_free.nc'
mgrid_path = f"{input_path}/{mgrid_filename}"
temp_path = "/home/zkg/h1_scan/temp"

# Load MGrid data
mgrid_data = MGrid.from_file(mgrid_path)
coil_list = mgrid_data.coil_names
num_coils = len(coil_list)
r_max, r_min, z_max, z_min = mgrid_data.rmax, mgrid_data.rmin, mgrid_data.zmax, mgrid_data.zmin
nr, nz, nphi, nfp = mgrid_data.nr, mgrid_data.nz, mgrid_data.nphi, mgrid_data.nfp + 1

# Calculate grid parameters
grid_unit = (r_max - r_min) / (nr - 1)
r_range = np.arange(r_min / grid_unit, r_max / grid_unit + 1, dtype=int)*grid_unit
z_range = np.arange(z_min / grid_unit, z_max/ grid_unit + 1, dtype=int)*grid_unit
phi_radrange = np.linspace(0, 360/nfp,nphi,endpoint=False)
phi_arcrange = np.deg2rad(phi_radrange)
delta_phi = np.deg2rad(360 / nfp / (nphi))
# phirad_grid,r_grid, z_grid = np.meshgrid(phi_radrange,r_range, z_range,indexing='ij')

bp_raw = mgrid_data.bp_arr.transpose((0,1,3,2))*1000 # phi,r,z.set current units = 1kA
br_raw = mgrid_data.br_arr.transpose((0,1,3,2))*1000
bz_raw = mgrid_data.bz_arr.transpose((0,1,3,2))*1000
bp_raw[bp_raw == 0] = 1e-10
br_raw[br_raw == 0] = 1e-10
bz_raw[bz_raw == 0] = 1e-10

# dR = (br_raw/bp_raw)*r_grid*delta_phi
# dZ = (bz_raw/bp_raw)*r_grid*delta_phi

dR_spline = np.zeros((num_coils, 8, nphi, nr, nz))
dZ_spline = np.zeros((num_coils, 8, nphi, nr, nz))
bp_spline = np.zeros((num_coils, 8, nphi, nr, nz))
br_spline = np.zeros((num_coils, 8, nphi, nr, nz))
bz_spline = np.zeros((num_coils, 8, nphi, nr, nz))

print(f"Begin spline interpolation")
time_start = time.time()
for i in range(num_coils):
    bp_spline[i,:,:,:,:] = r8herm_spline(phi_radrange, r_range, z_range, bp_raw[i,:,:,:])
    br_spline[i,:,:,:,:] = r8herm_spline(phi_radrange, r_range, z_range, br_raw[i,:,:,:])
    bz_spline[i,:,:,:,:] = r8herm_spline(phi_radrange, r_range, z_range, bz_raw[i,:,:,:])
time_end = time.time()

print(f"Spline interpolation done in {time_end - time_start:.3f} seconds")
# Save spline data
np.savez(f"{temp_path}/spline_data.npz", bp_spline=bp_spline, br_spline=br_spline, bz_spline=bz_spline,r_range=r_range,z_range=z_range,phi_radrange=phi_radrange)