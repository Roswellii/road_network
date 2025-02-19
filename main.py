import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Switch to the TkAgg backend
import matplotlib.pyplot as plt

# Function to generate a trajectory
def generate_trajectory(start_point, end_point, num_points):
    return np.linspace(start_point, end_point, num_points)

# Generating the first trajectory from (0,0) to (10,10)
trajectory_1 = generate_trajectory(np.array([0, 0]), np.array([10, 10]), 10)

# Generating the second trajectory from (0,10) to (10,0), crossing the first one
trajectory_2 = generate_trajectory(np.array([0, 10]), np.array([10, 0]), 10)

# Plotting the trajectories
plt.figure(figsize=(6,6))
plt.plot(trajectory_1[:,0], trajectory_1[:,1], marker='o', label='Trajectory 1')
plt.plot(trajectory_2[:,0], trajectory_2[:,1], marker='x', label='Trajectory 2')

# Highlight the crossing point
crossing_point = ([5, 5])
plt.plot(crossing_point[0], crossing_point[1], 'ro', label='Crossing Point')

plt.xlim(-1, 11)
plt.ylim(-1, 11)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Two Crossing Trajectories')
plt.legend()
plt.grid(True)
plt.show()

# Print the coordinates of the trajectories
print("Trajectory 1 Points:\n", trajectory_1)
print("\nTrajectory 2 Points:\n", trajectory_2)