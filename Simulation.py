import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Gravitational constant
G = 1

# Bodies (initial states)
bodies = {
    "body1": {"pos": np.array([1.0, 0.0, 0.0]),
              "vel": np.array([0.0, 0.5, 0.0]),
              "mass": 1.0},

    "body2": {"pos": np.array([-1.0, 0.0, 0.0]),
              "vel": np.array([0.0, -0.5, 0.0]),
              "mass": 1.0},

    "body3": {"pos": np.array([0.0, 1.0, 0.0]),
              "vel": np.array([-0.5, 0.0, 0.5]),
              "mass": 1.0},
}


def grav_force(body1, body2):
    r = body2["pos"] - body1["pos"]
    r_norm = np.linalg.norm(r)
    return G * (body1["mass"] * body2["mass"] * r) / r_norm**3

def acceleration(body, net_force):
    return net_force / body["mass"]

def compute_all_forces(bodies):
    net_forces = {}
    for name in bodies:
        net_forces[name] = np.array([0.0, 0.0, 0.0])
    
    names = list(bodies.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            b_i = names[i]
            b_j = names[j]
            f = grav_force(bodies[b_i], bodies[b_j])
            net_forces[b_i] += f
            net_forces[b_j] -= f
    return net_forces

def update_bodies(bodies, dt):
    net_forces = compute_all_forces(bodies)
    
    for name, body in bodies.items():
        acc = acceleration(body, net_forces[name])
        body["vel"] += acc * dt
        body["pos"] += body["vel"] * dt

# Simulation parameters
dt = 0.01  # time step
steps = 500 # number of steps

# For visualization
body_names = list(bodies.keys())
positions = {name: [] for name in body_names}

# Run simulation
for step in range(steps):
    update_bodies(bodies, dt)
    # Store positions for later plotting
    for name in body_names:
        positions[name].append(bodies[name]["pos"].copy())

# Convert positions to arrays for plotting
for name in body_names:
    positions[name] = np.array(positions[name])

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set background color to black
ax.set_facecolor('black')

# Initialize the 3D plot
lines = {}
for name in body_names:
    line, = ax.plot([], [], [], label=name, marker="o")
    lines[name] = line

# Set plot limits (adjust based on your simulation scale)
lim_axis = 1
ax.set_xlim(-lim_axis, lim_axis)
ax.set_ylim(-lim_axis, lim_axis)
ax.set_zlim(-lim_axis, lim_axis)

# Ensure square grid by setting equal scaling
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for x, y, z axes

# Set labels and colors
ax.grid(False)
ax.set_xlabel('X', color='white')
ax.set_ylabel('Y', color='white')
ax.set_zlabel('Z', color='white')

# Add white grid lines
ax.grid(color='white', linestyle='--', linewidth=0.5)

# Remove grey pane background
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Change axis colors to white for visibility
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')

# Remove numbers on axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Initialize the 3D plot with specified colors
body_colors = ['red', 'white', 'blue']
lines = {}
for i, name in enumerate(bodies.keys()):
    line, = ax.plot([], [], [], label=name, marker="o", linestyle='-', color=body_colors[i], markersize=1)
    lines[name] = line

# Animation update function
def update(frame):
    for name, line in lines.items():
        pos = positions[name]
        line.set_data(pos[:frame, 0], pos[:frame, 1])
        line.set_3d_properties(pos[:frame, 2])
    return lines.values()

# Create the animation
ani = FuncAnimation(fig, update, frames=steps, interval=20, blit=False)

# Save the animation as a video or display it
ani.save('3body_simulation.mp4', fps=30, writer='ffmpeg')  # Save to file
plt.show()