# Generated from the companion Jupyter notebook.

# %% [markdown]
# # DEM-Assisted SPOT Orthorectification
# 
# > Portfolio notebook. Heavy embedded outputs were removed; re-run with the included data to regenerate results.

# %%
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import rasterio

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
def calculate_RMSE(observed_x, observed_y, calculated_x, calculated_y):
    # Calculate residuals in x and y directions
    dr_x = calculated_x - observed_x
    dr_y = calculated_y - observed_y

    # Calculate Euclidean distance for each residual
    residuals = np.sqrt(dr_x ** 2 + dr_y ** 2)
    
    # Calculate RMSE
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
    Coefficient = X
    
    x_coord_calc = a1 * X_coord + a2 * Y_coord + a3
    y_coord_calc = b1 * X_coord + b2 * Y_coord + b3

    return x_coord_calc, y_coord_calc , Coefficient

# %%
def plot_dr_vectors(img, ICP_points, GCP_points, points, dr_x, dr_y, scale_factor=1):
    plt.figure(figsize=(20, 12))
    plt.imshow(img, cmap='gray')

    # Adding colorbar with improved orientation and aspect ratio
    colorbar = plt.colorbar(orientation="vertical")
    colorbar.set_label("Pixel Intensity", fontsize=14, weight='bold')

    # Plot ICP points with label
    plt.scatter(ICP_points.iloc[:, 1], ICP_points.iloc[:, 2], c="green", marker="X", s=150, label="ICP", edgecolor="black", linewidths=2)
    # Plot GCP points with label
    plt.scatter(GCP_points.iloc[:, 1], GCP_points.iloc[:, 2], c="yellow", marker="o", s=150, label="GCP", edgecolor="black", linewidths=2)

    # Annotate each point
    for i in range(points.shape[0]):
        x_coord = points.iloc[i, 1]
        y_coord = points.iloc[i, 2]
        plt.text(x_coord + 80, y_coord - 10, str(points.iloc[i, 0]), fontsize=12, color="red", weight='bold', 
                bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3', alpha=0.5))

    # Adding a title, legend, and colorbar for reference
    plt.title("Spot Image with Ground Control Points (GCPs), Independent Contol Points(ICPs) and Residual Vectors (Calculated - Actual ICP)", fontsize=15, color="black", weight='bold')

    # Display legend
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
Z_GCP = GCP_points['Z'].to_numpy()

x_ICP = ICP_points['x'].to_numpy()
y_ICP = ICP_points['y'].to_numpy()
X_ICP = ICP_points['X'].to_numpy()
Y_ICP = ICP_points['Y'].to_numpy()
Z_ICP = ICP_points['Z'].to_numpy()

# %%
x_corners = np.array([0, 0, image_data.shape[0], image_data.shape[0]])
y_corners = np.array([0, image_data.shape[1], image_data.shape[1], 0])

X_corners, Y_corners , Coefficient = affine_backward(x_GCP, y_GCP, X_GCP, Y_GCP, x_corners, y_corners)
X_corners = np.round(X_corners)
Y_corners = np.round(Y_corners)

corners_coord = np.column_stack((X_corners,Y_corners))
print("CornersCordinates:",corners_coord)
print("Coefficient :",Coefficient)

# %%
X_indices = np.arange(X_corners.min() - 10, X_corners.max() + 10, 10)
Y_indices = np.arange(Y_corners.min() - 10, Y_corners.max() + 10, 10)

# %%
X_indices

# %%
# Close the rectangle by adding the first point at the end
rectangle_x = np.append(X_corners, X_corners[0])
rectangle_y = np.append(Y_corners, Y_corners[0])

# Plot the rectangle
plt.figure(figsize=(20, 12))
plt.plot(rectangle_x, rectangle_y, 'b-')  # Blue line for the rectangle
plt.scatter(rectangle_x[:-1], rectangle_y[:-1], color='red')  # Red points for the corners
plt.plot([X_indices.min(),X_indices.min(),X_indices.max(),X_indices.max(),X_indices.min()],[Y_indices.min(),Y_indices.max(),Y_indices.max(),Y_indices.min(),Y_indices.min()],'b-')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rectangle from Corner Points with 10-Meter Grid')

# %%
# Use meshgrid to create all coordinate pairs
X_Ground, Y_Ground = np.meshgrid(X_indices, Y_indices)

# Flatten the arrays if you need them in 1D
X_Ground = X_Ground.ravel()
Y_Ground = Y_Ground.ravel()

# %%
with rasterio.open("../../data/dem/Dem_Merge.tif") as dem:
    dem_data = dem.read(1)
    transform = dem.transform
    rows, cols = ~transform * (X_Ground, Y_Ground)
    print(dem.crs)
    print(dem.bounds)

# %%
plt.figure(figsize=(20,12))
plt.imshow(dem_data, cmap='gray')

colorbar = plt.colorbar(orientation="vertical", fraction=0.03, pad=0.04)
colorbar.set_label("Pixel Intensity", fontsize=14, weight='bold')

plt.title("Spot Image", fontsize=15, color="black", weight='bold')
plt.axis('off')
plt.show()

# %%
# Vectorized bilinear interpolation function
def bilinear_interpolation_vectorized(image_data, x_img, y_img):
    image_data = np.vstack([image_data, image_data[-1:, :]])
    image_data = np.hstack([image_data, image_data[:, -1:]])

    x_int = np.floor(x_img).astype(int)
    y_int = np.floor(y_img).astype(int)

    # Fractional parts
    x_frac = x_img - x_int
    y_frac = y_img - y_int

    # Gather four neighboring pixel values for all points in a vectorized manner
    L00 = image_data[x_int, y_int]
    L10 = image_data[x_int + 1, y_int]
    L01 = image_data[x_int, y_int + 1]
    L11 = image_data[x_int + 1, y_int + 1]

    # Calculate interpolated values using bilinear interpolation formula
    DN = (L00 * (1 - x_frac) * (1 - y_frac) +
          L10 * x_frac * (1 - y_frac) +
          L01 * (1 - x_frac) * y_frac +
          L11 * x_frac * y_frac)

    return np.round(DN).astype(int)

# %%
Z_Ground = bilinear_interpolation_vectorized(dem_data, cols, rows)

# %%
def affine3D(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_coord, Y_coord, Z_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))
    
    A = np.zeros((2*n, 8))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], Z_GCP[i], 1, 0, 0, 0, 0]
        else:
            j = i - n
            A[i, :] = [0, 0, 0, 0, X_GCP[j], Y_GCP[j], Z_GCP[j], 1]

    X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))
    
    a1, a2, a3, a4, b1, b2, b3, b4 = X
    
    x_coord_calc = a1 * X_coord + a2 * Y_coord + a3 * Z_coord + a4
    y_coord_calc = b1 * X_coord + b2 * Y_coord + b3 * Z_coord + b4

    return x_coord_calc, y_coord_calc

# %%
x_ICP_affine, y_ICP_affine = affine3D(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_ICP, Y_ICP, Z_ICP)
x_GCP_affine, y_GCP_affine = affine3D(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_GCP, Y_GCP, Z_GCP)

RMSE_affine, dr_x_affine, dr_y_affine = calculate_RMSE(x_ICP, y_ICP, x_ICP_affine, y_ICP_affine)
RMSE_GCP_affine, dr_x_affine_GCP, dr_y_affine_GCP = calculate_RMSE(x_GCP, y_GCP, x_GCP_affine, y_GCP_affine)

plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_affine, dr_y_affine, scale_factor=20)

print("Affine GCP RMSE:", RMSE_GCP_affine)
print("Affine ICP RMSE:", RMSE_affine)

# %%
def polynomial3DD2(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_coord, Y_coord, Z_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))

    A = np.zeros((2*n, 20))
    for i in range(2*n):
        if i < n:
            A[i, :] = [1, X_GCP[i], Y_GCP[i], Z_GCP[i], X_GCP[i]**2, Y_GCP[i]**2, Z_GCP[i]**2, X_GCP[i]*Y_GCP[i], X_GCP[i]*Z_GCP[i], Y_GCP[i]*Z_GCP[i]] + [0] * 10
        else:
            j = i - n
            A[i, :] = [0] * 10 + [1, X_GCP[j], Y_GCP[j], Z_GCP[j], X_GCP[j]**2, Y_GCP[j]**2, Z_GCP[j]**2, X_GCP[j]*Y_GCP[j], X_GCP[j]*Z_GCP[j], Y_GCP[j]*Z_GCP[j]]

    N = np.dot(np.transpose(A), A)
    S = np.dot(np.transpose(A), L)

    X = np.dot(np.linalg.inv(N), S)
    
    a00, a10, a01, a001, a20, a02, a002, a11, a101, a011, b00, b10, b01, b001, b20, b02, b002, b11, b101, b011 = X

    x_coord_calc = a00 + a10 * X_coord + a01 * Y_coord + a001 * Z_coord + a20 * (X_coord**2) + a02 * (Y_coord**2) + a002 * (Z_coord**2) + a11 * (X_coord * Y_coord) + a101 * (X_coord * Z_coord) + a011 * (Y_coord * Z_coord)
    y_coord_calc = b00 + b10 * X_coord + b01 * Y_coord + b001 * Z_coord + b20 * (X_coord**2) + b02 * (Y_coord**2) + b002 * (Z_coord**2) + b11 * (X_coord * Y_coord) + b101 * (X_coord * Z_coord) + b011 * (Y_coord * Z_coord)

    return x_coord_calc, y_coord_calc

# %%
x_ICP_polynomialD2, y_ICP_polynomialD2 = polynomial3DD2(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_ICP, Y_ICP, Z_ICP)
x_GCP_polynomialD2, y_GCP_polynomialD2 = polynomial3DD2(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_GCP, Y_GCP, Z_GCP)

RMSE_polynomialD2, dr_x_polynomialD2, dr_y_polynomialD2 = calculate_RMSE(x_ICP, y_ICP, x_ICP_polynomialD2, y_ICP_polynomialD2)
RMSE_GCP_polynomialD2, dr_x_polynomialD2_GCP, dr_y_polynomialD2_GCP = calculate_RMSE(x_GCP, y_GCP, x_GCP_polynomialD2, y_GCP_polynomialD2)

plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_polynomialD2, dr_y_polynomialD2, scale_factor=200)

print("Polynomial Degree 2 GCP RMSE:", RMSE_GCP_polynomialD2)
print("Polynomial Degree 2 ICP RMSE:", RMSE_polynomialD2)

# %%
def DLT(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_coord, Y_coord, Z_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))

    A = np.zeros((2*n, 11))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], Z_GCP[i], 1, 0, 0, 0, 0, -x_GCP[i]*X_GCP[i], -x_GCP[i]*Y_GCP[i], -x_GCP[i]*Z_GCP[i]]
        else:
            j = i - n
            A[i, :] = [0, 0, 0, 0, X_GCP[j], Y_GCP[j], Z_GCP[j], 1, -y_GCP[j]*X_GCP[j], -y_GCP[j]*Y_GCP[j], -y_GCP[j]*Z_GCP[j]]

    X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))

    a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3 = X
    
    x_coord_calc = (a1*X_coord + a2*Y_coord + a3*Z_coord+a4)/(c1*X_coord+c2*Y_coord+c3*Z_coord+1)
    y_coord_calc = (b1*X_coord + b2*Y_coord + b3*Z_coord+b4)/(c1*X_coord+c2*Y_coord+c3*Z_coord+1)

    return x_coord_calc, y_coord_calc

# %%
x_ICP_DLT, y_ICP_DLT = DLT(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_ICP, Y_ICP, Z_ICP)
x_GCP_DLT, y_GCP_DLT = DLT(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_GCP, Y_GCP, Z_GCP)

RMSE_DLT, dr_x_DLT, dr_y_DLT = calculate_RMSE(x_ICP, y_ICP, x_ICP_DLT, y_ICP_DLT)
RMSE_GCP_DLT, dr_x_DLT_GCP, dr_y_DLT_GCP = calculate_RMSE(x_GCP, y_GCP, x_GCP_DLT, y_GCP_DLT)

plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_DLT, dr_y_DLT, scale_factor=30)

print("Projective GCP RMSE:", RMSE_GCP_DLT)
print("Projective ICP RMSE:", RMSE_DLT)

# %%
def polynomialD3(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_coord, Y_coord, Z_coord):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))


    A = np.zeros((n*2, 40))
    for i in range(n*2):
        if i < n:
            A[i, :] = [
                1, Y_GCP[i], X_GCP[i], Z_GCP[i],
                Y_GCP[i]**2, X_GCP[i]*Y_GCP[i], X_GCP[i]*Z_GCP[i], Y_GCP[i]*Z_GCP[i],
                X_GCP[i]**2, Z_GCP[i]**2,
                Y_GCP[i]**3, (X_GCP[i]**2)*Y_GCP[i], (X_GCP[i]**2)*Z_GCP[i],
                X_GCP[i]*(Y_GCP[i]**2), Z_GCP[i]*(Y_GCP[i]**2),
                (Z_GCP[i]**2)*X_GCP[i], (Z_GCP[i]**2)*Y_GCP[i], X_GCP[i]*Y_GCP[i]*Z_GCP[i],
                X_GCP[i]**3, Z_GCP[i]**3,
                *([0] * 20)
            ]
        else:
            j = i - n
            A[i, :] = [
                *([0] * 20),

                1, Y_GCP[j], X_GCP[j], Z_GCP[j],
                Y_GCP[j]**2, X_GCP[j]*Y_GCP[j], X_GCP[j]*Z_GCP[j], Y_GCP[j]*Z_GCP[j],
                X_GCP[j]**2, Z_GCP[j]**2,
                Y_GCP[j]**3, (X_GCP[j]**2)*Y_GCP[j], (X_GCP[j]**2)*Z_GCP[j],
                X_GCP[j]*(Y_GCP[j]**2), Z_GCP[j]*(Y_GCP[j]**2),
                (Z_GCP[j]**2)*X_GCP[j], (Z_GCP[j]**2)*Y_GCP[j], X_GCP[j]*Y_GCP[j]*Z_GCP[j],
                X_GCP[j]**3, Z_GCP[j]**3
            ]


    coord_terms = []
    for j in range(X_coord.shape[0]):
        coord_terms.append([
            1, Y_coord[j], X_coord[j], Z_coord[j],
            Y_coord[j]**2, X_coord[j] * Y_coord[j], X_coord[j]*Z_coord[j], Y_coord[j]*Z_coord[j],
            X_coord[j]**2, Z_coord[j]**2,

            Y_coord[j]**3,
            (X_coord[j]**2) * Y_coord[j],
            (X_coord[j]**2) * Z_coord[j],
            X_coord[j] * (Y_coord[j]**2),
            Z_coord[j] * (Y_coord[j]**2),
            (Z_coord[j]**2)*X_coord[j],
            (Z_coord[j]**2)*Y_coord[j],
            Z_coord[j]*X_coord[j]*Y_coord[j],
            X_coord[j]**3,
            Z_coord[j]**3
        ])
    coord_terms = np.array(coord_terms)

    base_terms = [(0, 20), (1, 21), (2, 22), (3, 23), (4, 24), (5, 25)]
    additional_terms = [(6, 26), (7, 27), (8, 28), (9, 29)]

    x_coord_calc = []
    y_coord_calc = []

    for i in range(4):
        for pairs in combinations(additional_terms, i+1):

            selected_pairs = base_terms.copy()
            selected_pairs += pairs

            selected_indices = [index for pair in selected_pairs for index in pair]
            selected_indices.sort()

            Anew = A[:, selected_indices].astype(np.longdouble)
            Ld = L.astype(np.longdouble)

            N = np.dot(Anew.T, Anew)
            S = np.dot(Anew.T, Ld)

            N = N.astype(np.float64)
            S = S.astype(np.float64)

            X = np.dot(np.linalg.inv(N), S)

            filtered_selected_indices = [item for item in selected_indices if item < 20]
            print(filtered_selected_indices)

            filtered_ICP_points = np.tile(coord_terms[:, filtered_selected_indices], (1, 2))

            xy_calc = filtered_ICP_points * X

            x_coord_calc.append(np.sum(xy_calc[:, :int(X.shape[0] / 2)], axis=1))
            y_coord_calc.append(np.sum(xy_calc[:, int(X.shape[0] / 2):], axis=1))

    return x_coord_calc, y_coord_calc

# %%
x_ICP_polynomialD3, y_ICP_polynomialD3 = polynomialD3(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_ICP, Y_ICP, Z_ICP)
x_GCP_polynomialD3, y_GCP_polynomialD3 = polynomialD3(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_GCP, Y_GCP, Z_GCP)
for i in range(len(x_ICP_polynomialD3)):
    RMSE_polynomialD3, dr_x_polynomialD3, dr_y_polynomialD3 = calculate_RMSE(x_ICP, y_ICP,x_ICP_polynomialD3[i], y_ICP_polynomialD3[i])

    RMSE_GCP_polynomialD3, dr_x_polynomialD3_GCP, dr_y_polynomialD3_GCP = calculate_RMSE(x_GCP, y_GCP,x_GCP_polynomialD3[i], y_GCP_polynomialD3[i])

    plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_polynomialD3, dr_y_polynomialD3, scale_factor=30)
    print(f"Polynomial Degree 3 GCP RMSE (Instance {i}):", RMSE_GCP_polynomialD3)
    print(f"Polynomial Degree 3 ICP RMSE (Instance {i}):", RMSE_polynomialD3)
    print()

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
def polynomialD2_OnlyX(x_GCP, X_GCP, Y_GCP,Z_GCP, X_coord, Y_coord, Z_coord):
    n = x_GCP.shape[0]
    L = x_GCP

    A = np.zeros((n, 10))
    for i in range(n):
            A[i, :] = [1, Y_GCP[i], X_GCP[i],Z_GCP[i], Y_GCP[i]**2, X_GCP[i]*Y_GCP[i],X_GCP[i]*Z_GCP[i],Y_GCP[i]*Z_GCP[i],Z_GCP[i]**2, X_GCP[i]**2]

    N = np.dot(np.transpose(A), A)
    S = np.dot(np.transpose(A), L)

    X = np.dot(np.linalg.inv(N), S)
    
    a000, a010, a100, a001,a020, a110,a101,a011, a002,a200 = X

    x_coord_calc = a000 + a010*Y_coord + a100*X_coord +a001*Z_coord + a020*(Y_coord**2) + a110*X_coord*Y_coord +a101*X_coord*Z_coord+a011*Y_coord*Z_coord +a002*(Z_coord**2)+a200*(X_coord**2)

    return x_coord_calc

# %%
x_ICP_polynomialD2_OnlyX = polynomialD2_OnlyX(x_GCP, X_GCP, Y_GCP, Z_GCP, X_ICP, Y_ICP, Z_ICP)
x_GCP_polynomialD2_OnlyX = polynomialD2_OnlyX(x_GCP, X_GCP, Y_GCP, Z_GCP, X_GCP, Y_GCP, Z_GCP)

RMSE_polynomialD2_OnlyX, dr_x_polynomialD2_OnlyX = RMSE_OnlyX(x_ICP, x_ICP_polynomialD2_OnlyX)
RMSE_GCP_polynomialD2_OnlyX, dr_x_polynomialD2_GCP_OnlyX = RMSE_OnlyX(x_GCP, x_GCP_polynomialD2_OnlyX)

# plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_polynomialD2, dr_y_polynomialD2, scale_factor=200)

print("Polynomial Degree 2_OnlyX GCP RMSE:", RMSE_GCP_polynomialD2_OnlyX)
print("Polynomial Degree 2_OnlyX ICP RMSE:", RMSE_polynomialD2_OnlyX)

# %%
def DLT_OnlyY(y_GCP, X_GCP, Y_GCP,Z_GCP, X_coord, Y_coord,Z_coord):
    n = x_GCP.shape[0]
    L = y_GCP

    A = np.zeros((n,7))
    for i in range(n):
            A[i, :] = [X_GCP[i], Y_GCP[i],Z_GCP[i], 1, -y_GCP[i]*X_GCP[i], -y_GCP[i]*Y_GCP[i],-y_GCP[i]*Z_GCP[i]]

    X = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, L))

    b1, b2, b3,b4, c1, c2,c3 = X
    
    y_coord_calc = (b1*X_coord + b2*Y_coord + b3*Z_coord+b4)/(c1*X_coord+c2*Y_coord+c3*Z_coord+1)

    return y_coord_calc

# %%
y_ICP_projective_OnlyY = DLT_OnlyY(y_GCP, X_GCP, Y_GCP, Z_GCP, X_ICP, Y_ICP, Z_ICP)
y_GCP_projective_OnlyY = DLT_OnlyY(y_GCP, X_GCP, Y_GCP, Z_GCP, X_GCP, Y_GCP, Z_GCP)

RMSE_projective, dr_y_projective = RMSE_OnlyY(y_ICP, y_ICP_projective_OnlyY)
RMSE_GCP_projective, dr_y_projective_GCP = RMSE_OnlyY(y_GCP, y_GCP_projective_OnlyY)

# plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_projective, dr_y_projective, scale_factor=30)

print("Projective_onlyY GCP RMSE:", RMSE_GCP_projective)
print("Projective_onlyY ICP RMSE:", RMSE_projective)

# %%
RMSE_combined_GCP, dr_x_combined, dr_y_combined = calculate_RMSE(x_GCP, y_GCP, x_GCP_polynomialD2_OnlyX, y_GCP_projective_OnlyY)
RMSE_combined_ICP, dr_x_combined, dr_y_combined = calculate_RMSE(x_ICP, y_ICP, x_ICP_polynomialD2_OnlyX, y_ICP_projective_OnlyY)

plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_combined, dr_y_combined, scale_factor=30)

print("Projective GCP RMSE:", RMSE_combined_GCP)
print("Projective ICP RMSE:", RMSE_combined_ICP)

# %%
def normalize_minmax(data):
    shift = (np.max(data) + np.min(data)) / 2
    scale = (np.max(data) - np.min(data)) / 2 
    
    normalized_data = (data - shift) / scale
    return normalized_data, shift, scale

def denormalize_minmax(normalized_data, shift, scale):
    return normalized_data * scale + shift

# %%
X_GCP_norm, shift_X_GCP, scale_X_GCP = normalize_minmax(X_GCP)
Y_GCP_norm, shift_Y_GCP, scale_Y_GCP = normalize_minmax(Y_GCP)
Z_GCP_norm, shift_Z_GCP, scale_Z_GCP = normalize_minmax(Z_GCP)

# %%
x_GCP_norm, shift_x_GCP, scale_x_GCP = normalize_minmax(x_GCP)
y_GCP_norm, shift_y_GCP, scale_y_GCP = normalize_minmax(y_GCP)

# %%
X_ICP_norm = (X_ICP - shift_X_GCP)/scale_X_GCP
Y_ICP_norm = (Y_ICP - shift_Y_GCP)/scale_Y_GCP
Z_ICP_norm = (Z_ICP - shift_Z_GCP)/scale_Z_GCP

# %%
def denormalize(x, Shift, Scale):
            return (x*Scale)+Shift

# %%
def RFM_d1(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_coord, Y_coord, Z_coord, Shift_x, Shift_y, Scale_x, Scale_y):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))
    W = np.eye(2*n)

    A = np.zeros((n*2, 14))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], Z_GCP[i], 1, -x_GCP[i]*X_GCP[i], -x_GCP[i]*Y_GCP[i], -x_GCP[i]*Z_GCP[i],0,0,0,0,0,0,0]
        else:
            j = i - n
            A[i, :] = [0,0,0,0,0,0,0,X_GCP[j], Y_GCP[j], Z_GCP[j], 1, -y_GCP[j]*X_GCP[j], -y_GCP[j]*Y_GCP[j], -y_GCP[j]*Z_GCP[j]]

    RMSE0 = [1000000]
    while True:
        X = np.dot(np.linalg.inv(np.dot(np.dot(A.T, W**2),A)), np.dot(np.dot(A.T, W**2),L))
        a1 , a2, a3 , a4, b1, b2, b3, c1, c2, c3, c4, d1, d2, d3 = X

        x_coord_calc = (a1*X_coord + a2*Y_coord + a3*Z_coord+a4)/(b1*X_coord+b2*Y_coord+b3*Z_coord+1)
        y_coord_calc = (c1*X_coord + c2*Y_coord + c3*Z_coord+c4)/(d1*X_coord+d2*Y_coord+d3*Z_coord+1)

        RMSE,dr_x,dr_y = calculate_RMSE(x_ICP, y_ICP,denormalize(x_coord_calc,Shift_x,Scale_x), denormalize(y_coord_calc,Shift_y,Scale_y))
        RMSE0.append(RMSE)
        if np.abs(RMSE0[-1] - RMSE0[-2]) < 1e-10:
            print()
            return denormalize(x_coord_calc,Shift_x,Scale_x), denormalize(y_coord_calc,Shift_y,Scale_y), RMSE, dr_x, dr_y, X

        B = b1*X_GCP + b2*Y_GCP + b3*Z_GCP + 1
        D = d1*X_GCP + d2*Y_GCP + d3*Z_GCP + 1

        W_x = 1/B
        W_y = 1/D
        W = np.diag(np.concatenate((W_x, W_y)))

# %%
x_ICP_RFM_d1, y_ICP_RFM_d1 ,RMSE_RFM_d1, dr_x_RFM_d1, dr_y_RFM_d1, X_RFM_d1 = RFM_d1(x_GCP_norm, y_GCP_norm, X_GCP_norm, Y_GCP_norm, Z_GCP_norm, X_ICP_norm, Y_ICP_norm, Z_ICP_norm, shift_x_GCP, shift_y_GCP, scale_x_GCP, scale_y_GCP)
plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_RFM_d1, dr_y_RFM_d1, scale_factor=30)

# %%
RMSE_RFM_d1

# %%
# p2=p4  d2
def RFM_d2_1(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_coord, Y_coord, Z_coord, Shift_x, Shift_y, Scale_x, Scale_y):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))
    W = np.eye(2*n)

    A = np.zeros((n*2, 29))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], Z_GCP[i],X_GCP[i]*Y_GCP[i],X_GCP[i]*Z_GCP[i],Y_GCP[i]*Z_GCP[i],X_GCP[i]**2,Y_GCP[i]**2,Z_GCP[i]**2,1,
                        -x_GCP[i]*X_GCP[i], -x_GCP[i]*Y_GCP[i], -x_GCP[i]*Z_GCP[i],-x_GCP[i]*X_GCP[i]*Y_GCP[i],-x_GCP[i]*X_GCP[i]*Z_GCP[i],
                        -x_GCP[i]*Y_GCP[i]*Z_GCP[i],-x_GCP[i]*(X_GCP[i]**2),-x_GCP[i]*(Y_GCP[i]**2),-x_GCP[i]*(Z_GCP[i]**2),0,0,0,0,0,0,0,0,0,0]
        else:
            j = i - n
            A[i, :] = [0,0,0,0,0,0,0,0,0,0,
                        -y_GCP[j]*X_GCP[j], -y_GCP[j]*Y_GCP[j], -y_GCP[j]*Z_GCP[j],-y_GCP[j]*X_GCP[j]*Y_GCP[j],-y_GCP[j]*X_GCP[j]*Z_GCP[j],
                        -y_GCP[j]*Y_GCP[j]*Z_GCP[j],-y_GCP[j]*(X_GCP[j]**2),-y_GCP[j]*(Y_GCP[j]**2),-y_GCP[j]*(Z_GCP[j]**2),
                        X_GCP[j], Y_GCP[j], Z_GCP[j],X_GCP[j]*Y_GCP[j],X_GCP[j]*Z_GCP[j],Y_GCP[j]*Z_GCP[j],X_GCP[j]**2,Y_GCP[j]**2,Z_GCP[j]**2,1]

    RMSE0 = [1000000]
    while True:
        X = np.dot(np.linalg.inv(np.dot(np.dot(A.T, W**2),A)), np.dot(np.dot(A.T, W**2),L))
        a1 , a2, a3 , a4, a5, a6, a7, a8, a9, a10, c1, c2, c3, c4, c5, c6, c7, c8, c9 , b1, b2, b3, b4, b5, b6, b7, b8, b9, b10= X

        x_coord_calc = (a1*X_coord + a2*Y_coord + a3*Z_coord+a4*X_coord*Y_coord+ a5*X_coord*Z_coord + a6*Y_coord*Z_coord +a7*(X_coord**2)+a8*(Y_coord**2)+a9*(Z_coord**2)+a10)/(c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord + c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+1)
        y_coord_calc = (b1*X_coord + b2*Y_coord + b3*Z_coord+b4*X_coord*Y_coord+ b5*X_coord*Z_coord + b6*Y_coord*Z_coord +b7*(X_coord**2)+b8*(Y_coord**2)+b9*(Z_coord**2)+b10)/(c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord + c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+1)

        RMSE,dr_x,dr_y = calculate_RMSE(x_ICP, y_ICP,denormalize(x_coord_calc,Shift_x,Scale_x), denormalize(y_coord_calc,Shift_y,Scale_y))
        RMSE0.append(RMSE)
        if RMSE0[-1] - RMSE0[-2] < 1e-10:
            return denormalize(x_coord_calc,Shift_x,Scale_x), denormalize(y_coord_calc,Shift_y,Scale_y), RMSE, dr_x, dr_y

        B = c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord + c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+1
        D = c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord + c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+1

        W_x = 1/B
        W_y = 1/D
        W = np.diag(np.concatenate((W_x, W_y)))

# %%
x_ICP_RFM_d2_1, y_ICP_RFM_d2_1 ,RMSE_RFM_d2_1,dr_x_RFM_d2_1, dr_y_RFM_d2_1 = RFM_d2_1(x_GCP_norm, y_GCP_norm, X_GCP_norm, Y_GCP_norm, Z_GCP_norm, X_ICP_norm, Y_ICP_norm, Z_ICP_norm, shift_x_GCP, shift_y_GCP, scale_x_GCP, scale_y_GCP)
plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_RFM_d2_1, dr_y_RFM_d2_1, scale_factor=30)

# %%
RMSE_RFM_d2_1

# %%
# p2!=p4  d2
def RFM_d2_2(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_coord, Y_coord, Z_coord, Shift_x, Shift_y, Scale_x, Scale_y):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))
    W = np.eye(2*n)

    A = np.zeros((n*2, 38))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], Z_GCP[i],X_GCP[i]*Y_GCP[i],X_GCP[i]*Z_GCP[i],Y_GCP[i]*Z_GCP[i],X_GCP[i]**2,Y_GCP[i]**2,Z_GCP[i]**2,1,
                       0 ,0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0,-x_GCP[i]*X_GCP[i], -x_GCP[i]*Y_GCP[i], -x_GCP[i]*Z_GCP[i],-x_GCP[i]*X_GCP[i]*Y_GCP[i],-x_GCP[i]*X_GCP[i]*Z_GCP[i],
                        -x_GCP[i]*Y_GCP[i]*Z_GCP[i],-x_GCP[i]*(X_GCP[i]**2),-x_GCP[i]*(Y_GCP[i]**2),-x_GCP[i]*(Z_GCP[i]**2),0,0,0,0,0,0,0,0,0]
        else:
            j = i - n
            A[i, :] = [0,0,0,0,0,0,0,0,0,0,X_GCP[j], Y_GCP[j], Z_GCP[j],X_GCP[j]*Y_GCP[j],X_GCP[j]*Z_GCP[j],Y_GCP[j]*Z_GCP[j],X_GCP[j]**2,Y_GCP[j]**2,Z_GCP[j]**2, 1,
                       0, 0, 0, 0, 0, 0, 0, 0, 0,-y_GCP[j]*X_GCP[j], -y_GCP[j]*Y_GCP[j], -y_GCP[j]*Z_GCP[j],-y_GCP[j]*X_GCP[j]*Y_GCP[j],-y_GCP[j]*X_GCP[j]*Z_GCP[j],
                        -y_GCP[j]*Y_GCP[j]*Z_GCP[j],-y_GCP[j]*(X_GCP[j]**2),-y_GCP[j]*(Y_GCP[j]**2),-y_GCP[j]*(Z_GCP[j]**2) ]

    RMSE0 = [1000000]
    while True:
        X = np.dot(np.linalg.inv(np.dot(np.dot(A.T, W**2),A)), np.dot(np.dot(A.T, W**2),L))
        a1 , a2, a3 , a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, c1, c2, c3, c4, c5, c6, c7, c8, c9, d1, d2, d3 , d4 ,d5 ,d6 , d7, d8, d9 = X

        x_coord_calc = (a1*X_coord + a2*Y_coord + a3*Z_coord+a4*X_coord*Y_coord+ a5*X_coord*Z_coord + a6*Y_coord*Z_coord +a7*(X_coord**2)+a8*(Y_coord**2)+a9*(Z_coord**2)+a10)/(c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord + c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+1)
        y_coord_calc = (b1*X_coord + b2*Y_coord + b3*Z_coord+b4*X_coord*Y_coord+ b5*X_coord*Z_coord + b6*Y_coord*Z_coord +b7*(X_coord**2)+b8*(Y_coord**2)+b9*(Z_coord**2)+b10)/(d1*X_coord+d2*Y_coord+d3*Z_coord+ d4*X_coord*Y_coord+ d5*X_coord*Z_coord + d6*Y_coord*Z_coord +d7*(X_coord**2)+d8*(Y_coord**2)+d9*(Z_coord**2)+1)

        RMSE,dr_x,dr_y = calculate_RMSE(x_ICP, y_ICP,denormalize(x_coord_calc,Shift_x,Scale_x), denormalize(y_coord_calc,Shift_y,Scale_y))
        RMSE0.append(RMSE)
        if RMSE0[-1] - RMSE0[-2] < 1e-10:
            return denormalize(x_coord_calc,Shift_x,Scale_x), denormalize(y_coord_calc,Shift_y,Scale_y), RMSE, dr_x, dr_y
        
        B = c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord + c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+1
        D = d1*X_coord+d2*Y_coord+d3*Z_coord+ d4*X_coord*Y_coord+ d5*X_coord*Z_coord + d6*Y_coord*Z_coord +d7*(X_coord**2)+d8*(Y_coord**2)+d9*(Z_coord**2)+1

        W_x = 1/B
        W_y = 1/D
        W = np.diag(np.concatenate((W_x, W_y)))

# %%
x_ICP_RFM_d1, y_ICP_RFM_d1 ,RMSE_RFM_d2_2, dr_x_RFM_d2_2, dr_y_RFM_d2_2 = RFM_d2_2(x_GCP_norm, y_GCP_norm, X_GCP_norm, Y_GCP_norm, Z_GCP_norm, X_ICP_norm, Y_ICP_norm, Z_ICP_norm, shift_x_GCP, shift_y_GCP, scale_x_GCP, scale_y_GCP)
plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_RFM_d2_1, dr_y_RFM_d2_2, scale_factor=30)

# %%
RMSE_RFM_d2_2

# %%
# p2=p4  d23
def RFM_d3(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_coord, Y_coord, Z_coord, Shift_x, Shift_y, Scale_x, Scale_y):
    n = x_GCP.shape[0]
    L = np.concatenate((x_GCP, y_GCP))
    W = np.eye(2*n)

    A = np.zeros((n*2, 59))
    for i in range(2*n):
        if i < n:
            A[i, :] = [X_GCP[i], Y_GCP[i], Z_GCP[i],X_GCP[i]*Y_GCP[i],X_GCP[i]*Z_GCP[i],Y_GCP[i]*Z_GCP[i],X_GCP[i]**2,Y_GCP[i]**2,Z_GCP[i]**2,
                       (X_GCP[i]**2)*Y_GCP[i],(X_GCP[i]**2)*Z_GCP[i],(Y_GCP[i]**2)*X_GCP[i],(Y_GCP[i]**2)*Z_GCP[i],(Z_GCP[i]**2)*X_GCP[i],(Z_GCP[i]**2)*Y_GCP[i],
                       X_GCP[i]*Y_GCP[i]*Z_GCP[i],X_GCP[i]**3,Y_GCP[i]**3,Z_GCP[i]**3,1,
                        -x_GCP[i]*X_GCP[i], -x_GCP[i]*Y_GCP[i], -x_GCP[i]*Z_GCP[i],-x_GCP[i]*X_GCP[i]*Y_GCP[i],-x_GCP[i]*X_GCP[i]*Z_GCP[i],
                        -x_GCP[i]*Y_GCP[i]*Z_GCP[i],-x_GCP[i]*(X_GCP[i]**2),-x_GCP[i]*(Y_GCP[i]**2),-x_GCP[i]*(Z_GCP[i]**2),-x_GCP[i]*(X_GCP[i]**2)*Y_GCP[i],
                        -x_GCP[i]*(X_GCP[i]**2)*Z_GCP[i],-x_GCP[i]*(Y_GCP[i]**2)*X_GCP[i],-x_GCP[i]*(Y_GCP[i]**2)*Z_GCP[i],-x_GCP[i]*(Z_GCP[i]**2)*X_GCP[i],
                        -x_GCP[i]*(Z_GCP[i]**2)*Y_GCP[i],-x_GCP[i]*X_GCP[i]*Y_GCP[i]*Z_GCP[i],-x_GCP[i]*(X_GCP[i]**3),-x_GCP[i]*(Y_GCP[i]**3),
                        -x_GCP[i]*(Z_GCP[i]**3),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            j = i - n
            A[i, :] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        -y_GCP[j]*X_GCP[j], -y_GCP[j]*Y_GCP[j], -y_GCP[j]*Z_GCP[j],-y_GCP[j]*X_GCP[j]*Y_GCP[j],-y_GCP[j]*X_GCP[j]*Z_GCP[j],
                        -y_GCP[j]*Y_GCP[j]*Z_GCP[j],-y_GCP[j]*(X_GCP[j]**2),-y_GCP[j]*(Y_GCP[j]**2),-y_GCP[j]*(Z_GCP[j]**2),
                        -y_GCP[j]*(X_GCP[j]**2)*Y_GCP[j],-y_GCP[j]*(X_GCP[j]**2)*Z_GCP[j],-y_GCP[j]*(Y_GCP[j]**2)*X_GCP[j],-y_GCP[j]*(Y_GCP[j]**2)*Z_GCP[j],-y_GCP[j]*(Z_GCP[j]**2)*X_GCP[j],
                        -y_GCP[j]*(Z_GCP[j]**2)*Y_GCP[j],-y_GCP[j]*X_GCP[j]*Y_GCP[j]*Z_GCP[j],-y_GCP[j]*(X_GCP[j]**3),-y_GCP[j]*(Y_GCP[j]**3),-y_GCP[j]*(Z_GCP[j]**3),X_GCP[j], Y_GCP[j], Z_GCP[j],X_GCP[j]*Y_GCP[j],X_GCP[j]*Z_GCP[j],Y_GCP[j]*Z_GCP[j],X_GCP[j]**2,Y_GCP[j]**2,Z_GCP[j]**2,
                        (X_GCP[j]**2)*Y_GCP[j],(X_GCP[j]**2)*Z_GCP[j],(Y_GCP[j]**2)*X_GCP[j],(Y_GCP[j]**2)*Z_GCP[j],(Z_GCP[j]**2)*X_GCP[j],(Z_GCP[j]**2)*Y_GCP[j],
                         X_GCP[j]*Y_GCP[j]*Z_GCP[j],X_GCP[j]**3,Y_GCP[j]**3,Z_GCP[j]**3,1 ]

    RMSE0 = [1000000]
    while True:
        X = np.dot(np.linalg.inv(np.dot(np.dot(A.T, W**2),A)), np.dot(np.dot(A.T, W**2),L))
        a1 , a2, a3 , a4, a5, a6, a7, a8, a9, a10, a11,a12,a13,a14,a15,a16,a17,a18,a19,a20, c1, c2, c3, c4, c5, c6, c7, c8, c9 ,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20= X

        x_coord_calc = (a1*X_coord + a2*Y_coord + a3*Z_coord+a20*X_coord*Y_coord+ a5*X_coord*Z_coord + a6*Y_coord*Z_coord +a7*(X_coord**2)+a8*(Y_coord**2)+a9*(Z_coord**2)+a10*(X_coord**2)*Y_coord+ a11*(X_coord**2)*Z_coord)+ a12*(Y_coord**2)*X_coord+a13*(Y_coord**2)*Z_coord+a14*(Z_coord**2)*X_coord+a15*(Z_coord**2)*Y_coord+a16*X_coord*Y_coord*Z_coord+a17*(X_coord)**3+a18*(Y_coord**3)+a19*(Z_coord**3)+a4/(c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord +
                                                                                                                                                                                                                                                                                                                                                                                                                        c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+c10*(X_coord**2)*Y_coord+c11*(X_coord**2)*Z_coord+c12*(Y_coord**2)*X_coord+c13*(Y_coord**2)*Z_coord+c14*(Z_coord**2)*X_coord+c15*(Z_coord**2)*Y_coord +c16*X_coord*Y_coord*Z_coord+c17*(X_coord**3)+
                                                                                                                                                                                                                                                                                                                                                                                                                        c18*(Y_coord**3)+c19*(Z_coord**3)+1)
        y_coord_calc = (b1*X_coord + b2*Y_coord + b3*Z_coord+b20*X_coord*Y_coord+ b5*X_coord*Z_coord + b6*Y_coord*Z_coord +b7*(X_coord**2)+b8*(Y_coord**2)+b9*(Z_coord**2)+b10*(X_coord**2)*Y_coord+ b11*(X_coord**2)*Z_coord)+ b12*(Y_coord**2)*X_coord+b13*(Y_coord**2)*Z_coord+b14*(Z_coord**2)*X_coord+b15*(Z_coord**2)*Y_coord+b16*X_coord*Y_coord*Z_coord+b17*(X_coord)**3+b18*(Y_coord**3)+b19*(Z_coord**3)+b4/(c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord +
                                                                                                                                                                                                                                                                                                                                                                                                                        c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+c10*(X_coord**2)*Y_coord+c11*(X_coord**2)*Z_coord+c12*(Y_coord**2)*X_coord+c13*(Y_coord**2)*Z_coord+c14*(Z_coord**2)*X_coord+c15*(Z_coord**2)*Y_coord +c16*X_coord*Y_coord*Z_coord+c17*(X_coord**3)+
                                                                                                                                                                                                                                                                                                                                                                                                                        c18*(Y_coord**3)+c19*(Z_coord**3)+1)
        

        RMSE,dr_x,dr_y = calculate_RMSE(x_ICP, y_ICP,denormalize(x_coord_calc,Shift_x,Scale_x), denormalize(y_coord_calc,Shift_y,Scale_y))
        RMSE0.append(RMSE)
        if RMSE0[-1] - RMSE0[-2] < 1e-10:
            return denormalize(x_coord_calc,Shift_x,Scale_x), denormalize(y_coord_calc,Shift_y,Scale_y), RMSE, dr_x, dr_y

        B = c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord + c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+c10*(X_coord**2)*Y_coord+c11*(X_coord**2)*Z_coord+c12*(Y_coord**2)*X_coord+c13*(Y_coord**2)*Z_coord+c14*(Z_coord**2)*X_coord+c15*(Z_coord**2)*Y_coord +c16*X_coord*Y_coord*Z_coord+c17*(X_coord**3)+ c18*(Y_coord**3)+c19*(Z_coord**3)+1
        D = c1*X_coord+c2*Y_coord+c3*Z_coord+ c4*X_coord*Y_coord+ c5*X_coord*Z_coord + c6*Y_coord*Z_coord +c7*(X_coord**2)+c8*(Y_coord**2)+c9*(Z_coord**2)+c10*(X_coord**2)*Y_coord+c11*(X_coord**2)*Z_coord+c12*(Y_coord**2)*X_coord+c13*(Y_coord**2)*Z_coord+c14*(Z_coord**2)*X_coord+c15*(Z_coord**2)*Y_coord +c16*X_coord*Y_coord*Z_coord+c17*(X_coord**3)+ c18*(Y_coord**3)+c19*(Z_coord**3)+1
        W_x = 1/B
        W_y = 1/D
        W = np.diag(np.concatenate((W_x, W_y)))

# %%
x_ICP_RFM_d1, y_ICP_RFM_d1 ,RMSE_RFM_d3, dr_x_RFM_d3, dr_y_RFM_d3 = RFM_d3(x_GCP_norm, y_GCP_norm, X_GCP_norm, Y_GCP_norm, Z_GCP_norm, X_ICP_norm, Y_ICP_norm, Z_ICP_norm, shift_x_GCP, shift_y_GCP, scale_x_GCP, scale_y_GCP)
plot_dr_vectors(image_data, ICP_points, GCP_points, points, dr_x_RFM_d3, dr_y_RFM_d3, scale_factor=30)

# %%
RMSE_RFM_d3

# %%
# RMSE values for different methods (replace these values with your actual data)
icp_rmse = [RMSE_affine, RMSE_DLT,RMSE_combined_ICP ,RMSE_polynomialD2, RMSE_polynomialD3,RMSE_RFM_d1,RMSE_RFM_d2_1,RMSE_RFM_d2_2,RMSE_RFM_d3 ]  # ICP: Affine, DLT, etc.
#gcp_rmse = [RMSE_GCP_affine, RMSE_GCP_DLT,RMSE_combined_GCP, RMSE_GCP_polynomialD2, RMSE_GCP_polynomialD3]  # GCP: Affine, DLT, etc.

# Model labels (methods)
models = ['Affine 3D'  , 'DLT',  'Hybrid Model','Polynomial2D','Polynomial3D','RFM D1(P2!=p4)','RFM D2(P2!=p4)','RFM D2(P2=p4)','RFM D3(P2=p4)']  # Replace with your model names

# X-axis positions for each group
x = np.arange(len(models))

# Bar width
bar_width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot ICP RMSE bars
rects1 = ax.bar(x - bar_width / 2, icp_rmse, bar_width, label='ICP', color='skyblue', edgecolor='black', linewidth=1)

# Plot GCP RMSE bars
#rects2 = ax.bar(x + bar_width / 2, gcp_rmse, bar_width, label='GCP', color='salmon', edgecolor='black', linewidth=1)

# Add labels, title, and legend
ax.set_xlabel('Methods', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('RMSE Comparison for ICP and GCP Across Methods', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7)
ax.legend(fontsize=11)

# # Connect GCP bars with lines
# for i in range(len(models) - 1):
#     x1_gcp = x[i] + bar_width / 2  # GCP bar position (current)
#     x2_gcp = x[i + 1] + bar_width / 2  # GCP bar position (next model)
#     y1_gcp = gcp_rmse[i]  # GCP bar height (current)
#     y2_gcp = gcp_rmse[i + 1]  # GCP bar height (next model)
#     ax.plot([x1_gcp, x2_gcp], [y1_gcp, y2_gcp], color='darkred', linestyle='--', linewidth=1.5, label='_nolegend_')  # Add connecting line

# Connect ICP bars with lines
for i in range(len(models) - 1):
    x1_icp = x[i] - bar_width / 2  # ICP bar position (current)
    x2_icp = x[i + 1] - bar_width / 2  # ICP bar position (next model)
    y1_icp = icp_rmse[i]  # ICP bar height (current)
    y2_icp = icp_rmse[i + 1]  # ICP bar height (next model)
    ax.plot([x1_icp, x2_icp], [y1_icp, y2_icp], color='darkblue', linestyle='--', linewidth=1.5, label='_nolegend_')  # Add connecting line

# Show RMSE values on each bar (optional)
def add_value_labels(rects, offset=3):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset),  # Offset for text
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='black')

add_value_labels(rects1)
#add_value_labels(rects2)

# Improve layout
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines
plt.tight_layout()
plt.show()

# %%
x_img, y_img = polynomial3DD2(x_GCP, y_GCP, X_GCP, Y_GCP, Z_GCP, X_Ground, Y_Ground, Z_Ground)

# %%
X_Ground_norm = (X_Ground - shift_X_GCP)/scale_X_GCP
Y_Ground_norm = (Y_Ground - shift_Y_GCP)/scale_Y_GCP
Z_Ground_norm = (Z_Ground - shift_Z_GCP)/scale_Z_GCP

# %%
a1 , a2, a3 , a4, b1, b2, b3, c1, c2, c3, c4, d1, d2, d3 = X_RFM_d1

x_img = ((a1*X_Ground_norm + a2*Y_Ground_norm + a3*Z_Ground_norm+a4)/(b1*X_Ground_norm+b2*Y_Ground_norm+b3*Z_Ground_norm+1))*scale_x_GCP + shift_x_GCP
y_img = ((c1*X_Ground_norm + c2*Y_Ground_norm + c3*Z_Ground_norm+c4)/(d1*X_Ground_norm+d2*Y_Ground_norm+d3*Z_Ground_norm+1))*scale_y_GCP + shift_y_GCP

# %%
# Round the coordinates and convert to integer type in one step
x_rounded = np.round(x_img).astype(int)
y_rounded = np.round(y_img).astype(int)

# Create a mask for valid indices
valid_mask = (
    (y_rounded >= 0) & (y_rounded < image_data.shape[0]) & (x_rounded >= 0) & (x_rounded < image_data.shape[1])
)

# Initialize img_ground with zeros
img_ground = np.zeros(x_img.shape[0], dtype=image_data.dtype)

# Apply the mask and retrieve valid pixel values
# img_ground[valid_mask] = image_data[x_rounded[valid_mask], y_rounded[valid_mask], ][:,0]
img_ground[valid_mask] = image_data[y_rounded[valid_mask], x_rounded[valid_mask]][:,0]

# %%
# Initialize a matrix with the size (Y_vector.size, X_vector.size)
output_matrix = np.zeros((Y_indices.size, X_indices.size), dtype=img_ground.dtype)

# Find the corresponding indices of X_Ground and Y_Ground in X_vector and Y_vector
x_indices = np.searchsorted(Y_indices, Y_Ground)
y_indices = np.searchsorted(X_indices, X_Ground)

# Use valid_mask to place img_ground values in the output matrix at the correct positions
output_matrix[x_indices[valid_mask], y_indices[valid_mask]] = img_ground[valid_mask]

# %%
# output_matrix = cv2.flip(output_matrix, 0) 
# cv2.imwrite("NearestNeighbor.jpg", output_matrix)

plt.figure(figsize=(20, 12))
plt.imshow(output_matrix, origin='lower', cmap='gray')  # 'gray' is commonly used for single-channel images
plt.colorbar(label='Pixel Value')  # Add a colorbar for reference
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Output Matrix as Image')
plt.show()

# %%
# Vectorized bilinear interpolation function
def bilinear_interpolation_vectorized(image_data, x_img, y_img):
    image_data = np.vstack([image_data, image_data[-1:, :]])
    image_data = np.hstack([image_data, image_data[:, -1:]])

    x_int = np.floor(x_img).astype(int)
    y_int = np.floor(y_img).astype(int)

    # Fractional parts
    x_frac = x_img - x_int
    y_frac = y_img - y_int

    # Gather four neighboring pixel values for all points in a vectorized manner
    L00 = image_data[x_int, y_int, 0]
    L10 = image_data[x_int + 1, y_int, 0]
    L01 = image_data[x_int, y_int + 1, 0]
    L11 = image_data[x_int + 1, y_int + 1, 0]

    # Calculate interpolated values using bilinear interpolation formula
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
# Initialize a matrix with the size (Y_vector.size, X_vector.size)
output_matrix = np.zeros((Y_indices.size, X_indices.size), dtype=img_ground.dtype)

# Find the corresponding indices of X_Ground and Y_Ground in X_vector and Y_vector
x_indices = np.searchsorted(Y_indices, Y_Ground)
y_indices = np.searchsorted(X_indices, X_Ground)

# Use valid_mask to place img_ground values in the output matrix at the correct positions
output_matrix[x_indices[valid_mask], y_indices[valid_mask]] = img_ground[valid_mask]

# %%
# output_matrix = cv2.flip(output_matrix, 0) 
# cv2.imwrite("BilinearInterpolarion.jpg", output_matrix)

plt.figure(figsize=(20, 12))
plt.imshow(output_matrix, origin='lower', cmap='gray')  # 'gray' is commonly used for single-channel images
plt.colorbar(label='Pixel Value')  # Add a colorbar for reference
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Output Matrix as Image')
plt.show()
