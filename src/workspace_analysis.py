import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.ndimage import gaussian_filter
from skimage import measure
from spatialmath import SE3

from robot import create_so101, manipulability, return_jacobian

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
control_qlimit = [
    [-2.1, -3.1, -0.0, -1.375, -1.57, -0.15],
    [2.1, 0.0, 3.1, 1.475, 3.1, 1.5],
]


# -----------------------------------------------------------
# Sampling & FK
# -----------------------------------------------------------
def sample_joint_configs(num_samples=10000):
    qlim = np.array(control_qlimit)[:, :5]
    samples = np.random.uniform(qlim[0], qlim[1], (num_samples, 5))
    return samples


def compute_full_fk(q, robot):
    theta0 = q[0]
    q_arm = q[1:5]
    T_arm = robot.fkine(q_arm)
    Rz = SE3.Rz(theta0)
    T_full = Rz * T_arm
    return T_full.t


def compute_workspace(robot, joint_samples):
    positions = []
    for q in joint_samples:
        positions.append(compute_full_fk(q, robot))
    return np.array(positions)


# -----------------------------------------------------------
# Workspace Visualization
# -----------------------------------------------------------
def voxelize_workspace(positions, grid_resolution=80):
    x_min, y_min, z_min = positions.min(axis=0)
    x_max, y_max, z_max = positions.max(axis=0)
    x_lin = np.linspace(x_min, x_max, grid_resolution)
    y_lin = np.linspace(y_min, y_max, grid_resolution)
    z_lin = np.linspace(z_min, z_max, grid_resolution)
    grid = np.zeros((grid_resolution, grid_resolution, grid_resolution))
    for pos in positions:
        xi = np.searchsorted(x_lin, pos[0]) - 1
        yi = np.searchsorted(y_lin, pos[1]) - 1
        zi = np.searchsorted(z_lin, pos[2]) - 1
        if (
            0 <= xi < grid_resolution
            and 0 <= yi < grid_resolution
            and 0 <= zi < grid_resolution
        ):
            grid[xi, yi, zi] += 1
    grid = gaussian_filter(grid, sigma=1.5)
    return grid, (x_lin, y_lin, z_lin)


def extract_surface(grid, grid_axes, level):
    x_lin, y_lin, z_lin = grid_axes
    verts, faces, _, values = measure.marching_cubes(grid, level=level)
    scale = [
        (x_lin[-1] - x_lin[0]) / len(x_lin),
        (y_lin[-1] - y_lin[0]) / len(y_lin),
        (z_lin[-1] - z_lin[0]) / len(z_lin),
    ]
    verts[:, 0] = x_lin[0] + verts[:, 0] * scale[0]
    verts[:, 1] = y_lin[0] + verts[:, 1] * scale[1]
    verts[:, 2] = z_lin[0] + verts[:, 2] * scale[2]
    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
    mesh = pv.PolyData(verts, faces_flat)
    return mesh


def visualize_surface(mesh, title, cmap="turquoise"):
    p = pv.Plotter()
    p.add_mesh(mesh, color=cmap, opacity=0.8, smooth_shading=True)
    p.add_axes()
    p.show_grid()
    p.add_title(title)
    p.show()


# -----------------------------------------------------------
# Manipulability Scatter (PyVista)
# -----------------------------------------------------------
def visualize_manipulability_scatter(robot, joint_samples, positions):
    """Compute manipulability and show via PyVista (Cleaner, no Open3D dependency)."""
    print("Computing manipulability values...")
    manipulabilities = []

    # Keep your calculation logic
    for q in joint_samples:
        J = return_jacobian(q[1:5], robot)
        m, cond = manipulability(J)
        manipulabilities.append(m)
    manipulabilities = np.array(manipulabilities)

    print(
        f"Manipulability range: {manipulabilities.min():.3e} → {manipulabilities.max():.3e}"
    )

    # Create PyVista Point Cloud
    pcd = pv.PolyData(positions)

    # Attach the scalar values directly to the mesh
    pcd["Manipulability"] = manipulabilities

    # Visualization
    plotter = pv.Plotter()
    plotter.add_mesh(
        pcd,
        scalars="Manipulability",
        cmap="viridis",
        point_size=5,
        render_points_as_spheres=True,  # Makes it look like particles rather than pixels
        scalar_bar_args={"title": "Manipulability Index"},
    )
    plotter.add_axes()
    plotter.add_text("Workspace Manipulability", position="upper_left")
    plotter.show()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    print("Creating robot...")
    robot = create_so101()

    print("Sampling joint configurations...")
    joint_samples = sample_joint_configs(num_samples=200000)  # 1000000

    print("Computing reachable workspace...")
    positions = compute_workspace(robot, joint_samples)

    print("Generating reachable workspace surface...")
    grid_reach, grid_axes = voxelize_workspace(positions, grid_resolution=80)
    level_reach = np.percentile(grid_reach, 50)
    mesh_reach = extract_surface(grid_reach, grid_axes, level_reach)
    visualize_surface(mesh_reach, title="Reachable Workspace", cmap="lightblue")

    mesh_reach.save("reachable_workspace.obj")

    print(f"Total sampled points: {len(positions)}")

    # -------------------------------------------------------
    # Manipulability scatter (Open3D)
    # -------------------------------------------------------
    visualize_manipulability_scatter(robot, joint_samples, positions)


if __name__ == "__main__":
    main()
