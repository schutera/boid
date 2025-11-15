import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Simulation parameters
width, height, depth = 250, 250, 250
num_boids = 100
visual_range = 75
speed_limit = 5
margin = 20
turn_factor = 2  # Stronger border repulsion
centering_factor = 0.005
avoid_factor_intra = 0.05  # Avoidance within same flock
avoid_factor_inter = 0.15  # Stronger avoidance between flocks
matching_factor_intra = 0.05  # Alignment within same flock
matching_factor_inter = 0.01  # Alignment between flocks
avoid_factor_predator = 0.5   # Strong avoidance of predators
enable_predators = False       # Toggle predators on/off
num_predators = 10
predator_speed = 2            # Speed of predator (lower = slower)
# Color parameters
color_flock1 = "#00c3ff"
color_flock2 = "#ffffff"
color_predator = "#9e405b"
min_distance = 20

class Boid:
    def __init__(self, flock=1):
        self.x = np.random.rand() * width
        self.y = np.random.rand() * height
        self.z = np.random.rand() * depth
        self.dx = np.random.rand() * 10 - 5
        self.dy = np.random.rand() * 10 - 5
        self.dz = np.random.rand() * 10 - 5
        self.flock = flock
        self.history = []

    def position(self):
        return np.array([self.x, self.y, self.z])

    def velocity(self):
        return np.array([self.dx, self.dy, self.dz])

# Half boids in flock 1, half in flock 2
# Half boids in flock 1, half in flock 2
boids = [Boid(flock=1) for _ in range(num_boids // 2)] + [Boid(flock=2) for _ in range(num_boids - num_boids // 2)]

# Predator class
class Predator:
    def __init__(self):
        self.x = np.random.rand() * width
        self.y = np.random.rand() * height
        self.z = np.random.rand() * depth
        self.dx = np.random.rand() * 10 - 5
        self.dy = np.random.rand() * 10 - 5
        self.dz = np.random.rand() * 10 - 5
        self.history = []

    def position(self):
        return np.array([self.x, self.y, self.z])

    def velocity(self):
        return np.array([self.dx, self.dy, self.dz])

if enable_predators:
    predators = [Predator() for _ in range(num_predators)]
else:
    predators = []

def distance(b1, b2):
    def distance_to_predator(boid, predator):
        return np.sqrt((boid.x - predator.x)**2 + (boid.y - predator.y)**2 + (boid.z - predator.z)**2)
    return np.sqrt((b1.x - b2.x)**2 + (b1.y - b2.y)**2 + (b1.z - b2.z)**2)

def fly_towards_center(boid):
    center_x, center_y, center_z, num_neighbors = 0, 0, 0, 0
    for other in boids:
        # Only consider boids of the same flock for cohesion
        if other.flock == boid.flock and distance(boid, other) < visual_range:
            center_x += other.x
            center_y += other.y
            center_z += other.z
            num_neighbors += 1
    if num_neighbors:
        center_x /= num_neighbors
        center_y /= num_neighbors
        center_z /= num_neighbors
        boid.dx += (center_x - boid.x) * centering_factor
        boid.dy += (center_y - boid.y) * centering_factor
        boid.dz += (center_z - boid.z) * centering_factor

def avoid_others(boid):
    move_x_intra, move_y_intra, move_z_intra = 0, 0, 0
    move_x_inter, move_y_inter, move_z_inter = 0, 0, 0
    for other in boids:
        if other is not boid and distance(boid, other) < min_distance:
            if other.flock == boid.flock:
                move_x_intra += boid.x - other.x
                move_y_intra += boid.y - other.y
                move_z_intra += boid.z - other.z
            else:
                move_x_inter += boid.x - other.x
                move_y_inter += boid.y - other.y
                move_z_inter += boid.z - other.z
    boid.dx += move_x_intra * avoid_factor_intra + move_x_inter * avoid_factor_inter
    boid.dy += move_y_intra * avoid_factor_intra + move_y_inter * avoid_factor_inter
    boid.dz += move_z_intra * avoid_factor_intra + move_z_inter * avoid_factor_inter

    # Avoid predators
    if enable_predators:
        for predator in predators:
            if distance(boid, predator) < visual_range:
                boid.dx += (boid.x - predator.x) * avoid_factor_predator
                boid.dy += (boid.y - predator.y) * avoid_factor_predator
                boid.dz += (boid.z - predator.z) * avoid_factor_predator

def match_velocity(boid):
    avg_dx, avg_dy, avg_dz, num = 0, 0, 0, 0
    for other in boids:
        if other.flock == boid.flock and distance(boid, other) < visual_range:
            avg_dx += other.dx
            avg_dy += other.dy
            avg_dz += other.dz
            num += 1
    if num > 0:
        avg_dx /= num
        avg_dy /= num
        avg_dz /= num
        boid.dx += (avg_dx - boid.dx) * matching_factor_intra
        boid.dy += (avg_dy - boid.dy) * matching_factor_intra
        boid.dz += (avg_dz - boid.dz) * matching_factor_intra

def limit_speed(boid):
    speed = np.sqrt(boid.dx**2 + boid.dy**2 + boid.dz**2)
    if speed > speed_limit:
        boid.dx = (boid.dx / speed) * speed_limit
        boid.dy = (boid.dy / speed) * speed_limit
        boid.dz = (boid.dz / speed) * speed_limit

def keep_within_bounds(boid):
    if boid.x < margin:
        boid.dx += turn_factor * (margin - boid.x) / margin
    if boid.x > width - margin:
        boid.dx -= turn_factor * (boid.x - (width - margin)) / margin
    if boid.y < margin:
        boid.dy += turn_factor * (margin - boid.y) / margin
    if boid.y > height - margin:
        boid.dy -= turn_factor * (boid.y - (height - margin)) / margin
    if boid.z < margin:
        boid.dz += turn_factor * (margin - boid.z) / margin
    if boid.z > depth - margin:
        boid.dz -= turn_factor * (boid.z - (depth - margin)) / margin

def update_boids():
    for boid in boids:
        fly_towards_center(boid)
        avoid_others(boid)
        match_velocity(boid)
        limit_speed(boid)
        keep_within_bounds(boid)
        boid.x += boid.dx
        boid.y += boid.dy
        boid.z += boid.dz
        # Store history for traces
        boid.history.append((boid.x, boid.y, boid.z))
        if len(boid.history) > 0:
            boid.history.pop(0)

    # Update predators
    if enable_predators:
        for predator in predators:
            # Chase nearest boid
            nearest = min(boids, key=lambda b: distance(b, predator))
            direction = np.array([nearest.x - predator.x, nearest.y - predator.y, nearest.z - predator.z])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                speed = predator_speed
            predator.dx, predator.dy, predator.dz = direction * speed
            predator.x += predator.dx
            predator.y += predator.dy
            predator.z += predator.dz
            # Store history for traces
            predator.history.append((predator.x, predator.y, predator.z))
            if len(predator.history) > 0:
                predator.history.pop(0)

def animate(frame):
    update_boids()
    ax.cla()
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, depth)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    xs1 = [b.x for b in boids if b.flock == 1]
    ys1 = [b.y for b in boids if b.flock == 1]
    zs1 = [b.z for b in boids if b.flock == 1]
    xs2 = [b.x for b in boids if b.flock == 2]
    ys2 = [b.y for b in boids if b.flock == 2]
    zs2 = [b.z for b in boids if b.flock == 2]
    ax.scatter(xs1, ys1, zs1, color=color_flock1, label="Flock 1")
    ax.scatter(xs2, ys2, zs2, color=color_flock2, label="Flock 2")
    # Draw traces for each boid
    for b in boids:
        if len(b.history) > 1:
            h = np.array(b.history)
            color = color_flock1 if b.flock == 1 else color_flock2
            ax.plot(h[:,0], h[:,1], h[:,2], color=color, alpha=0.5, linewidth=1)
    # Draw predators
    if enable_predators:
        xp = [p.x for p in predators]
        yp = [p.y for p in predators]
        zp = [p.z for p in predators]
        ax.scatter(xp, yp, zp, color=color_predator, s=80, marker='o', label="Predator")
        for p in predators:
            if len(p.history) > 1:
                h = np.array(p.history)
                ax.plot(h[:,0], h[:,1], h[:,2], color=color_predator, alpha=0.7, linewidth=2)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ani = FuncAnimation(fig, animate, interval=16) # 16ms  for ~60fps (what humans perceive)
plt.show()
