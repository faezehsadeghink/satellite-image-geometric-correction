# Generated from the companion Jupyter notebook.

# %% [markdown]
# # SPOT Image Geometric Rectification
# 
# > Portfolio notebook. Heavy embedded outputs were removed; re-run with the included data to regenerate results.

# %%
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# %%
image_data = cv2.imread("../../data/spot/Imag_03_Spot.jpg")
plt.figure(figsize=(20,12))
plt.imshow(image_data, cmap='gray')

colorbar = plt.colorbar(orientation="vertical", fraction=0.08, pad=0.06)
colorbar.set_label("Pixel Intensity", fontsize=14, weight='bold')

plt.title("Spot Image", fontsize=15, color="black", weight='bold')
plt.axis('off')
plt.show()

# %%
plt.figure(figsize=(20,12))

plt.hist(image_data.ravel(), bins=256, range=[0, 256], color='black', edgecolor='black', alpha=0.8)
plt.title('Grayscale Spot-Image Histogram', fontsize=16, fontweight='bold', color='black')
plt.xlabel('Pixel Intensity', fontsize=14, color='black')
plt.ylabel('Frequency', fontsize=14, color='black')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.xlim(0, 256)
plt.show()

# %%
points = pd.read_csv("../../data/spot/Image_03_Spot_Coordinates.txt" , header = None , delim_whitespace = True , names = ["No.","x","y","X","Y","Z"])
points = points.sort_values("No.")
points = points.reset_index(drop = True)
points

# %%
plt.figure(figsize=(20, 12))
plt.imshow(image_data, cmap='gray')

colorbar = plt.colorbar(orientation="vertical")
colorbar.set_label("Pixel Intensity", fontsize=14, weight='bold')


plt.scatter(points.iloc[:, 1], points.iloc[:, 2], c="cyan", marker="o", s=150, label="Points", edgecolor="black", linewidths=2)


for i in range(points.shape[0]):
    x_coord = points.iloc[i, 1]
    y_coord = points.iloc[i, 2]
    plt.text(x_coord + 80, y_coord - 30 , str(points.iloc[i, 0]), fontsize=12, color="green", weight='bold', 
             bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.3', alpha=0.5))
    

plt.title("Spot Image with Ground Control Points (GCPs)", fontsize=15, color="black", weight='bold')


plt.legend(loc="upper right", fontsize=12)

plt.axis('off')  
plt.show()

# %%

ICP_indices = [7, 8, 11, 35, 19, 21, 22, 24]

ICP_points = points[points["No."].isin(ICP_indices)]
GCP_points = points[~points["No."].isin(ICP_indices)]

# %%
plt.figure(figsize=(20, 12))
plt.imshow(image_data, cmap='gray')

colorbar = plt.colorbar(orientation="vertical")
colorbar.set_label("Pixel Intensity", fontsize=14, weight='bold')


plt.scatter(ICP_points.iloc[:, 1], ICP_points.iloc[:, 2], c="red", marker="o", s=150, label="ICP", edgecolor="black", linewidths=2)

plt.scatter(GCP_points.iloc[:, 1], GCP_points.iloc[:, 2], c="blue", marker="o", s=150, label="GCP", edgecolor="black", linewidths=2)


for i in range(points.shape[0]):
    x_coord = points.iloc[i, 1]
    y_coord = points.iloc[i, 2]
    plt.text(x_coord + 80, y_coord - 10, str(points.iloc[i, 0]), fontsize=12, color="green", weight='bold', 
             bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.3', alpha=0.5))


plt.title("Spot Image with Ground Control Points (GCPs) and Independent Contol Points(ICPs)", fontsize=15, color="black", weight='bold')


plt.legend(loc="upper right", fontsize=12)

plt.axis('on')  
plt.show()

# %%
def RMSE(observed_x, observed_y, calculated_x, calculated_y):

    dr_x = calculated_x - observed_x
    dr_y = calculated_y - observed_y

    residuals = np.sqrt(dr_x ** 2 + dr_y ** 2)
    

    rmse = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 1))
    
    return rmse, dr_x, dr_y

# %%
def affine_backward(x_GCP, y_GCP, X_GCP, Y_GCP, X_coord, Y_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((X_GCP, Y_GCP))

    A = np.zeros((2*n, 6))
    for i in range(2*n):
        if i < n:
            A[i, :] = [x_GCP[i], y_GCP[i], 1, 0, 0, 0]
        else:
            j = i - n
            A[i, :] = [0, 0, 0, x_GCP[j], y_GCP[j], 1]

    X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))

    a1, a2, a3, b1, b2, b3 = X
    ZARAYRB = X
    
    x_coord_calc = a1 * X_coord + a2 * Y_coord + a3
    y_coord_calc = b1 * X_coord + b2 * Y_coord + b3

    return x_coord_calc, y_coord_calc,ZARAYRB

# %%
def affine(x_GCP, y_GCP, X_GCP, Y_GCP, X_coord, Y_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))
    
    A = np.zeros((2*n, 6))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], 1, 0, 0, 0]
        else:
            j = i - n
            A[i, :] = [0, 0, 0, X_GCP[j], Y_GCP[j], 1]

    X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))
    
    a1, a2, a3, b1, b2, b3 = X
    
    x_coord_calc = a1 * X_coord + a2 * Y_coord + a3
    y_coord_calc = b1 * X_coord + b2 * Y_coord + b3

    return x_coord_calc, y_coord_calc

# %%
def conformal(x_GCP, y_GCP, X_GCP, Y_GCP, X_coord, Y_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))
    
    A = np.zeros((2*n, 4))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], 1, 0]
        else:
            j = i - n
            A[i, :] = [Y_GCP[j], -X_GCP[j], 0, 1]

    N = np.dot(A.T, A)
    S = np.dot(A.T, L)
    X = np.dot(np.linalg.inv(N), S)

    a, b, c, d = X

    x_coord_calc = a * X_coord + b * Y_coord + c
    y_coord_calc = -b * X_coord + a * Y_coord + d

    return x_coord_calc, y_coord_calc

# %%
def polynomialD2(x_GCP, y_GCP, X_GCP, Y_GCP, X_coord, Y_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))

    A = np.zeros((2*n, 12))
    for i in range(2*n):
        if i < n:
            A[i, :] = [1, Y_GCP[i], X_GCP[i], Y_GCP[i]**2, X_GCP[i]*Y_GCP[i], X_GCP[i]**2 , 0, 0, 0, 0, 0, 0]
        else:
            j = i - n
            A[i, :] = [0, 0, 0, 0, 0, 0, 1, Y_GCP[j], X_GCP[j], Y_GCP[j]**2, X_GCP[j]*Y_GCP[j], X_GCP[j]**2]

    N = np.dot(np.transpose(A), A)
    S = np.dot(np.transpose(A), L)

    X = np.dot(np.linalg.inv(N), S)
    
    a00, a01, a10, a02, a11, a20, b00, b01, b10, b02, b11, b20 = X

    x_coord_calc = a00 + a01*Y_coord + a10*X_coord + a02*(Y_coord**2) + a11*X_coord*Y_coord + a20*(X_coord**2)
    y_coord_calc = b00 + b01*Y_coord + b10*X_coord + b02*(Y_coord**2) + b11*X_coord*Y_coord + b20*(X_coord**2)

    return x_coord_calc, y_coord_calc

# %%
def polynomialD3(x_GCP, y_GCP, X_GCP, Y_GCP, X_coord, Y_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))

    
    A = np.zeros((n*2, 20))
    for i in range(n*2):
        if i < n:
            
            A[i, :] = [
                1, Y_GCP[i], X_GCP[i], Y_GCP[i]**2, X_GCP[i]*Y_GCP[i], X_GCP[i]**2,
                Y_GCP[i]**3, (X_GCP[i]**2)*Y_GCP[i], X_GCP[i]*(Y_GCP[i]**2), X_GCP[i]**3,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        else:
            j = i - n
            
            A[i, :] = [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, Y_GCP[j], X_GCP[j], Y_GCP[j]**2, X_GCP[j]*Y_GCP[j], X_GCP[j]**2,
                Y_GCP[j]**3, (X_GCP[j]**2)*Y_GCP[j], X_GCP[j]*(Y_GCP[j]**2), X_GCP[j]**3
            ]

    
    coord_terms = []
    for j in range(X_coord.shape[0]):
        coord_terms.append([
            1, Y_coord[j], X_coord[j], Y_coord[j]**2, X_coord[j] * Y_coord[j], X_coord[j]**2,
            Y_coord[j]**3, (X_coord[j]**2) * Y_coord[j], X_coord[j] * (Y_coord[j]**2), X_coord[j]**3
        ])
    coord_terms = np.array(coord_terms)

    base_terms = [(0, 10), (1, 11), (2, 12), (3, 13), (4, 14), (5, 15)]
    additional_terms = [(6, 16), (7, 17), (8, 18), (9, 19)]

    x_coord_calc = []
    y_coord_calc = []
    for i in range(4):
        for pairs in combinations(additional_terms, i+1):
            selected_pairs = base_terms.copy()
            selected_pairs += pairs

        
            selected_indices = [index for pair in selected_pairs for index in pair]
            selected_indices.sort()

            
            Anew = A[:, selected_indices]
            Anew = np.array(Anew)

            Anew = Anew.astype(np.longdouble)
            L = L.astype(np.longdouble)

        
            N = np.dot(np.transpose(Anew), Anew)
            S = np.dot(np.transpose(Anew), L)
            N = N.astype(np.float64)
            S = S.astype(np.float64)

            X = np.dot(np.linalg.inv(N), S)

            filtered_selected_indices = [item for item in selected_indices if item < 10]
            print(filtered_selected_indices)
            filtered_ICP_points = np.tile(coord_terms[:,filtered_selected_indices], (1, 2))
            
            xy_calc = filtered_ICP_points*X
            # print(xy_calc[:, :int(X.shape[0]/2)])

            
            x_coord_calc.append(np.sum(xy_calc[:, :int(X.shape[0]/2)], axis=1))
            y_coord_calc.append(np.sum(xy_calc[:, int(X.shape[0]/2):], axis=1))

    return x_coord_calc, y_coord_calc

# %%
def projective(x_GCP, y_GCP, X_GCP, Y_GCP, X_coord, Y_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))

    A = np.zeros((2*n, 8))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], 1, 0, 0, 0, -x_GCP[i]*X_GCP[i], -x_GCP[i]*Y_GCP[i]]
        else:
            j = i - n
            A[i, :] = [0, 0, 0, X_GCP[j], Y_GCP[j], 1, -y_GCP[j]*X_GCP[j], -y_GCP[j]*Y_GCP[j]]

    X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))

    a1, a2, a3, b1, b2, b3, c1, c2 = X
    
    x_coord_calc = (a1*X_coord + a2*Y_coord + a3)/(c1*X_coord+c2*Y_coord+1)
    y_coord_calc = (b1*X_coord + b2*Y_coord + b3)/(c1*X_coord+c2*Y_coord+1)

    return x_coord_calc, y_coord_calc

# %%
def plot_dr_vectors(img, ICP_points, GCP_points, points, dr_x, dr_y, scale_factor=1):
    plt.figure(figsize=(20, 12))
    plt.imshow(img, cmap='gray')

    colorbar = plt.colorbar(orientation="vertical")
    colorbar.set_label("Pixel Intensity", fontsize=14, weight='bold')

   
    plt.scatter(ICP_points.iloc[:, 1], ICP_points.iloc[:, 2], c="green", marker="X", s=150, label="ICP", edgecolor="black", linewidths=2)
    
    plt.scatter(GCP_points.iloc[:, 1], GCP_points.iloc[:, 2], c="yellow", marker="o", s=150, label="GCP", edgecolor="black", linewidths=2)

    for i in range(points.shape[0]):
        x_coord = points.iloc[i, 1]
        y_coord = points.iloc[i, 2]
        plt.text(x_coord + 80, y_coord - 10, str(points.iloc[i, 0]), fontsize=12, color="red", weight='bold', 
                bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3', alpha=0.5))

    plt.title("Spot Image with Ground Control Points (GCPs), Independent Contol Points(ICPs) and Residual Vectors (Calculated - Actual ICP)", fontsize=15, color="black", weight='bold')

    
    plt.legend(loc="upper right", fontsize=12)

    plt.quiver(ICP_points.iloc[:, 1], ICP_points.iloc[:, 2], dr_x * scale_factor, dr_y * scale_factor, angles='xy', scale_units='xy', scale=1, color='r')
    plt.xlabel("X (ICP)")
    plt.ylabel("Y (ICP)")
    plt.grid()
    plt.axis('off')
    plt.show()

# %%
x_GCP = GCP_points['x'].to_numpy()
y_GCP = GCP_points['y'].to_numpy()
X_GCP = GCP_points['X'].to_numpy()
Y_GCP = GCP_points['Y'].to_numpy()

x_ICP = ICP_points['x'].to_numpy()
y_ICP = ICP_points['y'].to_numpy()
X_ICP = ICP_points['X'].to_numpy()
Y_ICP = ICP_points['Y'].to_numpy()

# %%
x_corners = np.array([0, 0, image_data.shape[0], image_data.shape[0]])
y_corners = np.array([0, image_data.shape[1], image_data.shape[1], 0])

X_corners, Y_corners , ZARAYRB = affine_backward(x_GCP, y_GCP, X_GCP, Y_GCP, x_corners, y_corners)
X_corners = np.round(X_corners)
Y_corners = np.round(Y_corners)

corners_coord = np.column_stack((X_corners,Y_corners))
print("CornersCordinates:",corners_coord)
print("Coefficient :",ZARAYRB)

# %%
X_indices = np.arange(X_corners.min() - 10, X_corners.max() + 10, 10)
Y_indices = np.arange(Y_corners.min() - 10, Y_corners.max() + 10, 10)

# %%
plt.figure(figsize=(20, 12))
corners_closed = np.vstack([corners_coord, corners_coord[0]]) 
plt.plot(corners_closed[:, 0], corners_closed[:, 1], 'b-', linewidth=2, label='Image Boundary')
plt.scatter(corners_coord[:, 0], corners_coord[:, 1], color='red', s=80, label='Corner Points')

plt.plot([X_indices.min(),X_indices.min(),X_indices.max(),X_indices.max(),X_indices.min()],[Y_indices.min(),Y_indices.max(),Y_indices.max(),Y_indices.min(),Y_indices.min()],'b-')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rectangle from Corner Points with 10-Meter Grid')

# %%
x_ICP_affine, y_ICP_affine = affine(x_GCP, y_GCP, X_GCP, Y_GCP, X_ICP, Y_ICP)
x_GCP_affine, y_GCP_affine = affine(x_GCP, y_GCP, X_GCP, Y_GCP, X_GCP, Y_GCP)

RMSE_affine, dr_x_affine, dr_y_affine = RMSE(x_ICP, y_ICP, x_ICP_affine, y_ICP_affine)
RMSE_GCP_affine, dr_x_affine_GCP, dr_y_affine_GCP = RMSE(x_GCP, y_GCP, x_GCP_affine, y_GCP_affine)

# plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_affine, dr_y_affine, scale_factor=20)

print("Affine GCP RMSE:", RMSE_GCP_affine)
print("Affine ICP RMSE:", RMSE_affine)

# %%
x_ICP_conformal, y_ICP_conformal = conformal(x_GCP, y_GCP, X_GCP, Y_GCP, X_ICP, Y_ICP)
x_GCP_conformal, y_GCP_conformal = conformal(x_GCP, y_GCP, X_GCP, Y_GCP, X_GCP, Y_GCP)

RMSE_conformal, dr_x_conformal, dr_y_conformal =  RMSE(x_ICP, y_ICP, x_ICP_conformal, y_ICP_conformal)
RMSE_GCP_conformal, dr_x_conformal, dr_y_conformal =  RMSE(x_GCP, y_GCP, x_GCP_conformal, y_GCP_conformal)

# plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_conformal, dr_y_conformal, scale_factor=0.5)

print("GCP RMSE", RMSE_GCP_conformal)
print("ICP RMSE", RMSE_conformal)

# %%
x_ICP_polynomialD2, y_ICP_polynomialD2 = polynomialD2(x_GCP, y_GCP, X_GCP, Y_GCP, X_ICP, Y_ICP)
x_GCP_polynomialD2, y_GCP_polynomialD2 = polynomialD2(x_GCP, y_GCP, X_GCP, Y_GCP, X_GCP, Y_GCP)

RMSE_polynomialD2, dr_x_polynomialD2, dr_y_polynomialD2 = RMSE(x_ICP, y_ICP, x_ICP_polynomialD2, y_ICP_polynomialD2)
RMSE_GCP_polynomialD2, dr_x_polynomialD2_GCP, dr_y_polynomialD2_GCP = RMSE(x_GCP, y_GCP, x_GCP_polynomialD2, y_GCP_polynomialD2)

# plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_polynomialD2, dr_y_polynomialD2, scale_factor=200)

print("Polynomial Degree 2 GCP RMSE:", RMSE_GCP_polynomialD2)
print("Polynomial Degree 2 ICP RMSE:", RMSE_polynomialD2)

# %%
x_ICP_polynomialD3, y_ICP_polynomialD3 = polynomialD3(x_GCP, y_GCP, X_GCP, Y_GCP, X_ICP, Y_ICP)
x_GCP_polynomialD3, y_GCP_polynomialD3 = polynomialD3(x_GCP, y_GCP, X_GCP, Y_GCP, X_GCP, Y_GCP)

for i in range(len(x_ICP_polynomialD3)):
    RMSE_polynomialD3, dr_x_polynomialD3, dr_y_polynomialD3 = RMSE(x_ICP, y_ICP, x_ICP_polynomialD3[i], y_ICP_polynomialD3[i])
    RMSE_GCP_polynomialD3, dr_x_polynomialD3_GCP, dr_y_polynomialD3_GCP = RMSE(x_GCP, y_GCP, x_GCP_polynomialD3[i], y_GCP_polynomialD3[i])

    if RMSE_polynomialD3 < 5:
        scale_factor = 200
    elif 5 <= RMSE_polynomialD3 < 100:
        scale_factor = 30
    else:
        scale_factor = 1

    # plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_polynomialD3, dr_y_polynomialD3, scale_factor)

    print(f"Polynomial Degree 3 GCP RMSE (Instance {i}):", RMSE_GCP_polynomialD3)
    print(f"Polynomial Degree 3 ICP RMSE (Instance {i}):", RMSE_polynomialD3)
    print()

# %%
x_ICP_projective, y_ICP_projective = projective(x_GCP, y_GCP, X_GCP, Y_GCP, X_ICP, Y_ICP)
x_GCP_projective, y_GCP_projective = projective(x_GCP, y_GCP, X_GCP, Y_GCP, X_GCP, Y_GCP)

RMSE_projective, dr_x_projective, dr_y_projective = RMSE(x_ICP, y_ICP, x_ICP_projective, y_ICP_projective)
RMSE_GCP_projective, dr_x_projective_GCP, dr_y_projective_GCP = RMSE(x_GCP, y_GCP, x_GCP_projective, y_GCP_projective)

# plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_projective, dr_y_projective, scale_factor=30)

print("Projective GCP RMSE:", RMSE_GCP_projective)
print("Projective ICP RMSE:", RMSE_projective)

# %%
def RMSE_OnlyX(observed_x, calculated_x):

    dr_x = calculated_x - observed_x

    residuals = np.sqrt(dr_x ** 2)

    rmse = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 1))
    
    return rmse, dr_x

# %%
def RMSE_OnlyY(observed_y, calculated_y):

    dr_y = calculated_y - observed_y

    residuals = np.sqrt(dr_y ** 2)
    

    rmse = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 1))
    
    return rmse, dr_y

# %%
def polynomialD2_OnlyX(x_GCP, X_GCP, Y_GCP, X_coord, Y_coord):
    n = x_GCP.shape[0]
    L = x_GCP

    A = np.zeros((n, 6))
    for i in range(n):
            A[i, :] = [1, Y_GCP[i], X_GCP[i], Y_GCP[i]**2, X_GCP[i]*Y_GCP[i], X_GCP[i]**2]

    N = np.dot(np.transpose(A), A)
    S = np.dot(np.transpose(A), L)

    X = np.dot(np.linalg.inv(N), S)
    
    a00, a01, a10, a02, a11, a20 = X

    x_coord_calc = a00 + a01*Y_coord + a10*X_coord + a02*(Y_coord**2) + a11*X_coord*Y_coord + a20*(X_coord**2)

    return x_coord_calc

# %%
x_ICP_polynomialD2_OnlyX = polynomialD2_OnlyX(x_GCP, X_GCP, Y_GCP, X_ICP, Y_ICP)
x_GCP_polynomialD2_OnlyX = polynomialD2_OnlyX(x_GCP, X_GCP, Y_GCP, X_GCP, Y_GCP)

RMSE_polynomialD2_OnlyX, dr_x_polynomialD2_OnlyX = RMSE_OnlyX(x_ICP, x_ICP_polynomialD2_OnlyX)
RMSE_GCP_polynomialD2_OnlyX, dr_x_polynomialD2_GCP_OnlyX = RMSE_OnlyX(x_GCP, x_GCP_polynomialD2_OnlyX)

# plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_polynomialD2, dr_y_polynomialD2, scale_factor=200)

print("Polynomial Degree 2_OnlyX GCP RMSE:", RMSE_GCP_polynomialD2_OnlyX)
print("Polynomial Degree 2_OnlyX ICP RMSE:", RMSE_polynomialD2_OnlyX)

# %%
def projective_OnlyY(y_GCP, X_GCP, Y_GCP, X_coord, Y_coord):
    n = x_GCP.shape[0]
    L = y_GCP

    A = np.zeros((n, 5))
    for i in range(n):
            A[i, :] = [X_GCP[i], Y_GCP[i], 1, -y_GCP[i]*X_GCP[i], -y_GCP[i]*Y_GCP[i]]

    X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))

    b1, b2, b3, c1, c2 = X
    
    y_coord_calc = (b1*X_coord + b2*Y_coord + b3)/(c1*X_coord+c2*Y_coord+1)

    return y_coord_calc

# %%
y_ICP_projective_OnlyY = projective_OnlyY(y_GCP, X_GCP, Y_GCP, X_ICP, Y_ICP)
y_GCP_projective_OnlyY = projective_OnlyY(y_GCP, X_GCP, Y_GCP, X_GCP, Y_GCP)

RMSE_projective, dr_y_projective = RMSE_OnlyY(y_ICP, y_ICP_projective_OnlyY)
RMSE_GCP_projective, dr_y_projective_GCP = RMSE_OnlyY(y_GCP, y_GCP_projective_OnlyY)

# plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_projective, dr_y_projective, scale_factor=30)

print("Projective_onlyY GCP RMSE:", RMSE_GCP_projective)
print("Projective_onlyY ICP RMSE:", RMSE_projective)

# %%
RMSE_combined_GCP, dr_x_combined, dr_y_combined = RMSE(x_GCP, y_GCP, x_GCP_polynomialD2_OnlyX, y_GCP_projective_OnlyY)
RMSE_combined_ICP, dr_x_combined, dr_y_combined = RMSE(x_ICP, y_ICP, x_ICP_polynomialD2_OnlyX, y_ICP_projective_OnlyY)

plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_combined, dr_y_combined, scale_factor=30)

print("Projective GCP RMSE:", RMSE_combined_GCP)
print("Projective ICP RMSE:", RMSE_combined_ICP)

# %%

X_Ground, Y_Ground = np.meshgrid(X_indices, Y_indices)

X_Ground = X_Ground.ravel()
Y_Ground = Y_Ground.ravel()

# %%
x_img, y_img = polynomialD2(x_GCP, y_GCP, X_GCP, Y_GCP, X_Ground, Y_Ground)

# %%
# Round the coordinates and convert to integer type in one step
x_rounded = np.round(x_img).astype(int)
y_rounded = np.round(y_img).astype(int)

valid_mask = (
    (y_rounded >= 0) & (y_rounded < image_data.shape[0]) & (x_rounded >= 0) & (x_rounded < image_data.shape[1])
)


img_ground = np.zeros(x_img.shape[0], dtype=image_data.dtype)

img_ground[valid_mask] = image_data[y_rounded[valid_mask], x_rounded[valid_mask]][:,0]

# %%

output_matrix = np.zeros((Y_indices.size, X_indices.size), dtype=img_ground.dtype)

# Find the corresponding indices of X_Ground and Y_Ground in X_vector and Y_vector
x_indices = np.searchsorted(Y_indices, Y_Ground)
y_indices = np.searchsorted(X_indices, X_Ground)

# Use valid_mask to place img_ground values in the output matrix at the correct positions
output_matrix[x_indices[valid_mask], y_indices[valid_mask]] = img_ground[valid_mask]

# %%
# output_matrix = cv2.flip(output_matrix, 0) 
# cv2.imwrite("NearesrNeighbor.jpg", output_matrix)

plt.figure(figsize=(20, 12))
plt.imshow(output_matrix, origin='lower', cmap='gray')  
plt.colorbar(label='Pixel Value')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Output Matrix as Image')
plt.show()

# %%
def bilinear_interpolation_vectorized(image_data, x_img, y_img):
    image_data = np.vstack([image_data, image_data[-1:, :]])
    image_data = np.hstack([image_data, image_data[:, -1:]])

    x_int = np.floor(x_img).astype(int)
    y_int = np.floor(y_img).astype(int)

    x_frac = x_img - x_int
    y_frac = y_img - y_int


    L00 = image_data[x_int, y_int, 0]
    L10 = image_data[x_int + 1, y_int, 0]
    L01 = image_data[x_int, y_int + 1, 0]
    L11 = image_data[x_int + 1, y_int + 1, 0]

    DN = (L00 * (1 - x_frac) * (1 - y_frac) +
          L10 * x_frac * (1 - y_frac) +
          L01 * (1 - x_frac) * y_frac +
          L11 * x_frac * y_frac)

    return np.round(DN).astype(int)

valid_mask = (y_img >= 0) & (y_img < image_data.shape[0]) & (x_img >= 0) & (x_img < image_data.shape[1])

result = bilinear_interpolation_vectorized(image_data, y_img[valid_mask], x_img[valid_mask])

img_ground = np.zeros(x_img.shape[0], dtype=image_data.dtype)
img_ground[valid_mask] = result

# %%

output_matrix = np.zeros((Y_indices.size, X_indices.size), dtype=img_ground.dtype)

x_indices = np.searchsorted(Y_indices, Y_Ground)
y_indices = np.searchsorted(X_indices, X_Ground)


output_matrix[x_indices[valid_mask], y_indices[valid_mask]] = img_ground[valid_mask]

# %%
# output_matrix = cv2.flip(output_matrix, 0) 
# cv2.imwrite("BilinearInterpolarion.jpg", output_matrix)

plt.figure(figsize=(20, 12))
plt.imshow(output_matrix, origin='lower', cmap='gray')  
plt.colorbar(label='Pixel Value')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Output Matrix as Image')
plt.show()

# %%
plt.figure(figsize=(20, 12))
plt.imshow(output_matrix, origin='lower', cmap='gray') 


colorbar = plt.colorbar(orientation="vertical")
colorbar.set_label("Pixel Intensity", fontsize=14, weight='bold')

plt.scatter(np.searchsorted(X_indices, GCP_points.iloc[:, 3]), np.searchsorted(Y_indices, GCP_points.iloc[:, 4]), c="yellow", marker="o", s=150, label="GCP", edgecolor="black", linewidths=2)


for i in range(GCP_points.shape[0]):
    x_coord = np.searchsorted(X_indices, GCP_points.iloc[i, 3])
    y_coord = np.searchsorted(Y_indices, GCP_points.iloc[i, 4])
    plt.text(x_coord + 80, y_coord - 10, str(GCP_points.iloc[i, 0]), fontsize=12, color="red", weight='bold', 
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3', alpha=0.5))

plt.title("Spot Image with Ground Control Points (GCPs) and Independent Contol Points(ICPs)", fontsize=15, color="black", weight='bold')


plt.legend(loc="upper right", fontsize=12)

plt.show()

# %%
def MovingAverage_PW(dr_x, dr_y, X_GCP, Y_GCP, X_ICP, Y_ICP):
    n = dr_x.shape[0]
    L = np.concatenate((dr_x, dr_y))
    
    A = np.zeros((2*n, 6))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], 1, 0, 0, 0]
        else:
            j = i - n
            A[i, :] = [0, 0, 0, X_GCP[j], Y_GCP[j], 1]

    X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))
    
    a1, a2, a3, b1, b2, b3 = X
    
    dr_x_calc = a1 * X_ICP + a2 * Y_ICP + a3
    dr_y_calc = b1 * X_ICP + b2 * Y_ICP + b3
    l1=np.shape(L)
    a1=np.shape(A)
    return dr_x_calc, dr_y_calc,l1,a1

# %%
maskPW_GCP = GCP_points["No."].isin([13,10,40]).reset_index(drop=True)

# %%
maskPW_GCP

# %%
maskPW_ICP = ICP_points["No."].isin([11]).reset_index(drop=True)

# %%
maskPW_ICP

# %%
dr_x_polynomialD2_GCP[[6,8,29]]

# %%
MAPW = MovingAverage_PW(dr_x_polynomialD2_GCP[[6,8,29]], dr_y_polynomialD2_GCP[[6,8,29]], points[points["No."].isin([13,10,40])]["X"].to_numpy(), points[points["No."].isin([13,10,40])]["Y"].to_numpy(), points[points["No."].isin([11])]["X"].to_numpy(), points[points["No."].isin([11])]["Y"].to_numpy())

# %%
MAPW

# %%
x_ICP_polynomialD2_new = x_ICP_polynomialD2.copy()
y_ICP_polynomialD2_new = y_ICP_polynomialD2.copy()

# %%
x_ICP_polynomialD2_new[2] = x_ICP_polynomialD2[2] - MAPW[0]
y_ICP_polynomialD2_new[2]  = y_ICP_polynomialD2[2] - MAPW[1]

# %%
RMSE_polynomialD2_new, dr_x_polynomialD2_new, dr_y_polynomialD2_new = RMSE(x_ICP, y_ICP, x_ICP_polynomialD2_new, y_ICP_polynomialD2_new)

# %%
print(dr_y_polynomialD2_new)
print(dr_x_polynomialD2_new)

# %%
float(RMSE_polynomialD2_new)

# %%
plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_y_polynomialD2_new, dr_x_polynomialD2_new, scale_factor=30)
