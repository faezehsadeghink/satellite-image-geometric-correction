# Generated from the companion Jupyter notebook.

# %% [markdown]
# # QuickBird-2 RPC/RFM Geometric Accuracy Assessment
# 
# > Portfolio notebook. Heavy embedded outputs were removed; re-run with the included data to regenerate results.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer, Proj

# %%
def scale_data(data_arr):
    mid_val = (data_arr.max() + data_arr.min()) / 2
    range_val = (data_arr.max() - data_arr.min()) / 2
    return mid_val, range_val

def apply_normalization(data_arr, mid_val, range_val):
    return (data_arr - mid_val) / range_val

def reverse_normalization(normalized_arr, mid_val, range_val):
    return normalized_arr * range_val + mid_val

def compute_rmse(obs_col, obs_row, calc_col, calc_row):
    diff_col = calc_col - obs_col
    diff_row = calc_row - obs_row
    dist = np.sqrt(diff_col**2 + diff_row**2)
    return np.sqrt(np.sum(dist**2) / (len(dist) - 1))

# %%
gcp_df = pd.read_csv("../../data/quickbird/GCPsData.txt", sep='\\s+', header=None,
                     names=["col", "row", "east", "north", "elev"])
gcp_df.head(10)

# %%
proj_utm = Proj(proj="utm", zone=10, datum="WGS84")
proj_geo = Proj(proj="latlong", datum="WGS84")
coord_trans = Transformer.from_proj(proj_utm, proj_geo)

gcp_df["lon"], gcp_df["lat"] = coord_trans.transform(gcp_df["east"].values, gcp_df["north"].values)
gcp_df["height"] = gcp_df["elev"]
gcp_df.head(10)

# %%
col_raw = gcp_df['col'].to_numpy()
row_raw = gcp_df['row'].to_numpy()
lon_raw = gcp_df['lon'].to_numpy()
lat_raw = gcp_df['lat'].to_numpy()
hgt_raw = gcp_df['height'].to_numpy()

col_mid, col_rng = scale_data(col_raw)
row_mid, row_rng = scale_data(row_raw)
lon_mid, lon_rng = scale_data(lon_raw)
lat_mid, lat_rng = scale_data(lat_raw)
hgt_mid, hgt_rng = scale_data(hgt_raw)

col_n = apply_normalization(col_raw, col_mid, col_rng)
row_n = apply_normalization(row_raw, row_mid, row_rng)
lon_n = apply_normalization(lon_raw, lon_mid, lon_rng)
lat_n = apply_normalization(lat_raw, lat_mid, lat_rng)
hgt_n = apply_normalization(hgt_raw, hgt_mid, hgt_rng)

# %%
np.random.seed(89)

n_control = 60
n_check = 24

idx_all = np.arange(col_n.shape[0])
np.random.shuffle(idx_all)

ctrl_idx = idx_all[:n_control]
chk_idx = idx_all[n_control:]

col_ctrl, row_ctrl = col_n[ctrl_idx], row_n[ctrl_idx]
lon_ctrl, lat_ctrl, hgt_ctrl = lon_n[ctrl_idx], lat_n[ctrl_idx], hgt_n[ctrl_idx]

col_chk, row_chk = col_n[chk_idx], row_n[chk_idx]
lon_chk, lat_chk, hgt_chk = lon_n[chk_idx], lat_n[chk_idx], hgt_n[chk_idx]

# %%
idx_all

# %%
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#f8f9fa')
ax.set_facecolor('#ffffff')

ax.scatter(gcp_df["east"][ctrl_idx], gcp_df["north"][ctrl_idx],
           s=150, c='#2E86AB', edgecolor='#1a1a2e', linewidth=1.5,
           marker='o', label='GCPs (Control Points)', alpha=0.85, zorder=3)

ax.scatter(gcp_df["east"][chk_idx], gcp_df["north"][chk_idx],
           s=150, c='#E94560', edgecolor='#1a1a2e', linewidth=1.5,
           marker='^', label='ICPs (Check Points)', alpha=0.85, zorder=3)

ax.set_xlabel("Easting (m)", fontsize=13, fontweight='bold', color='#333333')
ax.set_ylabel("Northing (m)", fontsize=13, fontweight='bold', color='#333333')
ax.set_title("Distribution of Ground Control Points and Independent Check Points", 
             fontsize=15, fontweight='bold', color='#1a1a2e', pad=15)

ax.grid(True, linestyle=':', alpha=0.6, color='#cccccc', zorder=1)
ax.tick_params(axis='both', labelsize=11, colors='#333333')

for spine in ax.spines.values():
    spine.set_edgecolor('#888888')
    spine.set_linewidth(1.2)

legend = ax.legend(fontsize=11, loc='upper right', frameon=True, 
                   fancybox=True, shadow=True, framealpha=0.95)
legend.get_frame().set_edgecolor('#888888')

plt.tight_layout()
plt.show()

# %%
def create_polynomial_basis(lon, lat, height):
    """
    Create polynomial basis terms for Rational Function Model (RFM).
    
    This function generates 20 polynomial terms from longitude, latitude, and height
    coordinates for use in the RFM transformation.
    
    Parameters:
    -----------
    lon : array-like
        Longitude coordinates
    lat : array-like  
        Latitude coordinates
    height : array-like
        Height/elevation coordinates
        
    Returns:
    --------
    numpy.ndarray
        Array of shape (20, n_points) containing polynomial terms
    """
    n_points = len(lon)
    
    # Initialize array for 20 polynomial terms
    poly_terms = np.zeros((20, n_points))
    
    # Term 0: Constant term
    poly_terms[0, :] = 1.0
    
    # Terms 1-3: First-order terms (linear)
    poly_terms[1, :] = lon      # λ
    poly_terms[2, :] = lat      # φ  
    poly_terms[3, :] = height   # h
    
    # Terms 4-6: Second-order terms (cross products)
    poly_terms[4, :] = lon * lat      # λφ
    poly_terms[5, :] = lon * height   # λh
    poly_terms[6, :] = lat * height   # φh
    
    # Terms 7-9: Second-order terms (squared)
    poly_terms[7, :] = lon**2     # λ²
    poly_terms[8, :] = lat**2     # φ²
    poly_terms[9, :] = height**2  # h²
    
    # Term 10: Third-order term (triple product)
    poly_terms[10, :] = lon * lat * height  # λφh
    
    # Terms 11-13: Third-order terms (lon³, lon*lat², lon*height²)
    poly_terms[11, :] = lon**3           # λ³
    poly_terms[12, :] = lon * lat**2     # λφ²
    poly_terms[13, :] = lon * height**2   # λh²
    
    # Terms 14-16: Third-order terms (lon²*lat, lat³, lat*height²)
    poly_terms[14, :] = lon**2 * lat     # λ²φ
    poly_terms[15, :] = lat**3           # φ³
    poly_terms[16, :] = lat * height**2  # φh²
    
    # Terms 17-19: Third-order terms (lon²*height, lat²*height, height³)
    poly_terms[17, :] = lon**2 * height  # λ²h
    poly_terms[18, :] = lat**2 * height   # φ²h
    poly_terms[19, :] = height**3         # h³
    
    return poly_terms

def rational_func_model(c_ctrl, r_ctrl, lam_ctrl, phi_ctrl, h_ctrl, lam_eval, phi_eval, h_eval):
    num_obs = c_ctrl.shape[0]
    obs_vec = np.concatenate((c_ctrl, r_ctrl))
    
    poly = create_polynomial_basis(lam_ctrl, phi_ctrl, h_ctrl)
    
    design_mat = np.zeros((num_obs*2, 78))
    design_mat[:num_obs, :20] = poly.T
    design_mat[:num_obs, 20:39] = (-c_ctrl * poly)[1:].T
    design_mat[num_obs:, 39:59] = poly.T
    design_mat[num_obs:, 59:] = (-r_ctrl * poly)[1:].T
    
    delta_params = 1
    params_prev = 0
    weight_mat = np.eye(design_mat.shape[0])
    
    while np.linalg.norm(delta_params) >= 1e-3:
        AtWW = np.dot(design_mat.T, weight_mat**2)
        params = np.dot(np.linalg.inv(np.dot(AtWW, design_mat)), np.dot(AtWW, obs_vec))
        
        delta_params = params - params_prev
        params_prev = params
        
        denom_c = np.insert(params[20:39], 0, 1)
        denom_r = np.insert(params[59:], 0, 1)
        
        weight_mat[:num_obs, :num_obs] = np.diag(1 / np.sum(denom_c * poly.T, axis=1))
        weight_mat[num_obs:, num_obs:] = np.diag(1 / np.sum(denom_r * poly.T, axis=1))
    
    numer_c = params[:20]
    denom_c = np.insert(params[20:39], 0, 1)
    numer_r = params[39:59]
    denom_r = np.insert(params[59:], 0, 1)
    
    poly_eval = create_polynomial_basis(lam_eval, phi_eval, h_eval)
    
    c_result = np.sum(numer_c * poly_eval.T, axis=1) / np.sum(denom_c * poly_eval.T, axis=1)
    r_result = np.sum(numer_r * poly_eval.T, axis=1) / np.sum(denom_r * poly_eval.T, axis=1)
    
    return c_result, r_result

# %%
col_chk_rfm, row_chk_rfm = rational_func_model(col_ctrl, row_ctrl, lon_ctrl, lat_ctrl, hgt_ctrl, 
                                                lon_chk, lat_chk, hgt_chk)
col_ctrl_rfm, row_ctrl_rfm = rational_func_model(col_ctrl, row_ctrl, lon_ctrl, lat_ctrl, hgt_ctrl, 
                                                  lon_ctrl, lat_ctrl, hgt_ctrl)

rmse_chk = compute_rmse(reverse_normalization(col_chk, col_mid, col_rng), 
                        reverse_normalization(row_chk, row_mid, row_rng),
                        reverse_normalization(col_chk_rfm, col_mid, col_rng), 
                        reverse_normalization(row_chk_rfm, row_mid, row_rng))

rmse_ctrl = compute_rmse(reverse_normalization(col_ctrl, col_mid, col_rng), 
                         reverse_normalization(row_ctrl, row_mid, row_rng),
                         reverse_normalization(col_ctrl_rfm, col_mid, col_rng), 
                         reverse_normalization(row_ctrl_rfm, row_mid, row_rng))

print("RFM GCP RMSE:", rmse_ctrl)
print("RFM ICP RMSE:", rmse_chk)

# %%
import re

rpc_file = "../../data/quickbird/RPC.RPB"
rpc_params = {}

regex_single = re.compile(r"(\w+)\s*=\s*([\d\.\-E\+]+);")
regex_list_begin = re.compile(r"(\w+)\s*=\s*\(")
regex_list_vals = re.compile(r"([\d\.\-E\+]+),?")

with open(rpc_file, "r") as f:
    content = f.readlines()

active_key = None
for ln in content:
    ln = ln.strip()
    
    m = regex_single.match(ln)
    if m:
        rpc_params[m.group(1)] = float(m.group(2))
        continue
    
    m = regex_list_begin.match(ln)
    if m:
        active_key = m.group(1)
        rpc_params[active_key] = []
        continue
    
    if active_key:
        vals = regex_list_vals.findall(ln)
        if vals:
            rpc_params[active_key].extend([float(v) for v in vals])
        if ln.endswith(");"):
            active_key = None

col_n_rpc = apply_normalization(col_raw, rpc_params['sampOffset'], rpc_params['sampScale'])
row_n_rpc = apply_normalization(row_raw, rpc_params['lineOffset'], rpc_params['lineScale'])
lon_n_rpc = apply_normalization(lon_raw, rpc_params['longOffset'], rpc_params['longScale'])
lat_n_rpc = apply_normalization(lat_raw, rpc_params['latOffset'], rpc_params['latScale'])
hgt_n_rpc = apply_normalization(hgt_raw, rpc_params['heightOffset'], rpc_params['heightScale'])

# %%
poly_rpc = create_polynomial_basis(lon_n_rpc, lat_n_rpc, hgt_n_rpc)

# %%
col_rpc_n = np.sum((rpc_params['sampNumCoef'] * poly_rpc.T), axis=1) / np.sum((rpc_params['sampDenCoef'] * poly_rpc.T), axis=1)
row_rpc_n = np.sum((rpc_params['lineNumCoef'] * poly_rpc.T), axis=1) / np.sum((rpc_params['lineDenCoef'] * poly_rpc.T), axis=1)

# %%
col_rpc_dn = reverse_normalization(col_rpc_n, rpc_params['sampOffset'], rpc_params['sampScale'])
row_rpc_dn = reverse_normalization(row_rpc_n, rpc_params['lineOffset'], rpc_params['lineScale'])

# %%
poly_rpc = create_polynomial_basis(lon_n_rpc, lat_n_rpc, hgt_n_rpc)

# %%
err_col = col_rpc_dn - col_raw
err_row = row_rpc_dn - row_raw

# %%
np.random.seed(12)
sel_ctrl = np.random.choice(ctrl_idx, 6, replace=True)
sel_chk = np.random.choice(chk_idx, 4, replace=True)

obs_refine = np.concatenate((err_col[sel_ctrl], err_row[sel_ctrl]))

design_refine = np.zeros((obs_refine.shape[0], 6))
design_refine[:6, :3] = np.column_stack((np.ones(6), col_raw[sel_ctrl], row_raw[sel_ctrl]))
design_refine[6:, 3:] = np.column_stack((np.ones(obs_refine.shape[0] - 6), col_raw[sel_ctrl], row_raw[sel_ctrl]))

coef_refine = np.dot(np.linalg.inv(np.dot(design_refine.T, design_refine)), np.dot(design_refine.T, obs_refine))

err_col_pred = coef_refine[0] + coef_refine[1]*col_rpc_dn[sel_chk] + coef_refine[2]*row_rpc_dn[sel_chk]
err_row_pred = coef_refine[3] + coef_refine[4]*col_rpc_dn[sel_chk] + coef_refine[5]*row_rpc_dn[sel_chk]

rmse_refine = compute_rmse(err_col[sel_chk], err_row[sel_chk], err_col_pred, err_row_pred)
print("RFM dxdy Refinment RMSE:", rmse_refine)

# %%
err_col_all = coef_refine[0] + coef_refine[1]*col_rpc_dn[chk_idx] + coef_refine[2]*row_rpc_dn[chk_idx]
err_row_all = coef_refine[3] + coef_refine[4]*col_rpc_dn[chk_idx] + coef_refine[5]*row_rpc_dn[chk_idx]

col_refined = col_rpc_dn[chk_idx] - err_col_all
row_refined = row_rpc_dn[chk_idx] - err_row_all

# %%
col_rpc_n = np.sum((rpc_params['sampNumCoef'] * poly_rpc.T), axis=1) / np.sum((rpc_params['sampDenCoef'] * poly_rpc.T), axis=1)
row_rpc_n = np.sum((rpc_params['lineNumCoef'] * poly_rpc.T), axis=1) / np.sum((rpc_params['lineDenCoef'] * poly_rpc.T), axis=1)

col_rpc_dn = reverse_normalization(col_rpc_n, rpc_params['sampOffset'], rpc_params['sampScale'])
row_rpc_dn = reverse_normalization(row_rpc_n, rpc_params['lineOffset'], rpc_params['lineScale'])

rmse_rpc = compute_rmse(col_raw, row_raw, col_rpc_dn, row_rpc_dn)
print("RFM GCP RMSE:", rmse_rpc)

err_col = col_rpc_dn - col_raw
err_row = row_rpc_dn - row_raw

# %%
rmse_final = compute_rmse(col_refined, row_refined, col_raw[chk_idx], row_raw[chk_idx])
print("RFM Refinment RMSE:", rmse_final)
