import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from dqn_agent import DQNAgent


## Simulation parameters
width, height, depth = 500, 500, 500 #250, 250, 250 for more action
num_boids = 10  # Number of boids per flock
num_flocks = 1  # Number of flocks (new parameter) - when more than 7 add colours
num_leaders = 10  # Number of leader boids (simulation param)
model_save_interval = 50000  # Steps between saving model (simulation param)
continue_training = True  # If True, load model weights from file
continue_model_path = "checkpoints/shared_leader_agent_step200000.pth"  # Path to load model from
leader_control_mode = "dqn"  # Options: "dqn", "trajectory"
leader_reward_type = "circular"  # Options: "quadrant", "circular"
from reward_circular import circularity_reward
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
enable_leader = True          # Toggle leader boid logic on/off
show_quadrant_visu = False     # Toggle quadrant visualization on/off
num_predators = 10
predator_speed = 2            # Speed of predator (lower = slower)
# Color parameters
flock_colors = ["#00c3ff", "#ffffff", "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]  # Extendable color list
color_predator = "#9e405b"
min_distance = 20

from leader_trajectory import LeaderTrajectory

# Trajectory instance for leader control
leader_trajectory = LeaderTrajectory(
    trajectory_type="circle",
    center=[width/2, height/2, depth/2],
    radius=min(width, height, depth) * 0.3,
    speed=0.02,
    axis="z"
)

class Boid:
    def __init__(self, flock=1, is_leader=False):
        self.x = np.random.rand() * width
        self.y = np.random.rand() * height
        self.z = np.random.rand() * depth
        self.dx = np.random.rand() * 10 - 5
        self.dy = np.random.rand() * 10 - 5
        self.dz = np.random.rand() * 10 - 5
        self.flock = flock
        self.is_leader = is_leader
        self.history = []

    def position(self):
        return np.array([self.x, self.y, self.z])

    def velocity(self):
        return np.array([self.dx, self.dy, self.dz])

# Initialize boids for any number of flocks
boids = []
leader_boids = []
for flock_id in range(1, num_flocks + 1):
    # Add leader boids for this flock
    for _ in range(num_leaders):
        leader = Boid(flock=flock_id, is_leader=True)
        boids.append(leader)
        leader_boids.append(leader)
    # The rest are regular boids
    boids.extend([Boid(flock=flock_id) for _ in range(num_boids-num_leaders)])

# DQN agents for each leader
quadrant_count = 8
quadrant_assignments = np.random.randint(1, quadrant_count+1, size=num_flocks)
state_dim = 6 * 5 + 1  # 5 neighbors, each with 6 features, plus quadrant indicator
action_dim = 7  # 6 directions + no move
# Shared DQN agent for all leaders
shared_leader_agent = DQNAgent(state_dim, action_dim)
# Load model if continuing training
if continue_training:
    if os.path.exists(continue_model_path):
        shared_leader_agent.load(continue_model_path)
        print(f"Loaded shared_leader_agent weights from {continue_model_path}")
    else:
        print(f"Model file {continue_model_path} not found. Starting fresh.")

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

def get_quadrant(x, y, z):
    qx = int(x > width/2)
    qy = int(y > height/2)
    qz = int(z > depth/2)
    return 1 + qx + 2*qy + 4*qz

def leader_reward(leader, quadrant):
    # Reward for each boid in the assigned quadrant
    flock_boids = [b for b in boids if b.flock == leader.flock]
    if leader_reward_type == "circular":
        # Use circular trajectory reward
        circle_reward = circularity_reward(flock_boids)
        # Flocking reward: negative average pairwise distance (closer = higher reward)
        cohesion = 0
        if len(flock_boids) > 1:
            dists = [distance(b1, b2) for i, b1 in enumerate(flock_boids) for b2 in flock_boids[i+1:]]
            avg_dist = np.mean(dists)
            max_possible = np.sqrt(width**2 + height**2 + depth**2)
            cohesion = 1.0 - min(avg_dist / max_possible, 1.0)
        total_reward = (2*circle_reward + cohesion) / 3
        return total_reward, circle_reward, None, cohesion
    else:
        # Quadrant reward (default)
        in_quadrant = sum(1 for b in flock_boids if get_quadrant(b.x, b.y, b.z) == quadrant)
        quadrant_reward = in_quadrant / len(flock_boids)
        q = quadrant - 1
        qx = q & 1
        qy = (q >> 1) & 1
        qz = (q >> 2) & 1
        target_x = width/4 if qx == 0 else 3*width/4
        target_y = height/4 if qy == 0 else 3*height/4
        target_z = depth/4 if qz == 0 else 3*depth/4
        move_toward = 0
        for b in flock_boids:
            to_target = np.array([target_x - b.x, target_y - b.y, target_z - b.z])
            velocity = np.array([b.dx, b.dy, b.dz])
            if np.linalg.norm(to_target) > 0 and np.linalg.norm(velocity) > 0:
                move_toward += np.dot(to_target, velocity) / (np.linalg.norm(to_target) * np.linalg.norm(velocity))
        move_toward_reward = (move_toward / len(flock_boids) + 1) / 2
        cohesion = 0
        if len(flock_boids) > 1:
            dists = [distance(b1, b2) for i, b1 in enumerate(flock_boids) for b2 in flock_boids[i+1:]]
            avg_dist = np.mean(dists)
            max_possible = np.sqrt(width**2 + height**2 + depth**2)
            cohesion = 1.0 - min(avg_dist / max_possible, 1.0)
        total_reward = (2*quadrant_reward + 2*move_toward_reward + cohesion) / 5
        return total_reward, quadrant_reward, move_toward_reward, cohesion
        flock_boids = [b for b in boids if b.flock == leader.flock]
        if leader_reward_type == "circular":
            # Use circular trajectory reward
            circle_reward = circular_trajectory_reward(flock_boids, center=[width/2, height/2, depth/2], radius=min(width, height, depth)*0.3, axis="z")
            # Flocking reward: negative average pairwise distance (closer = higher reward)
            cohesion = 0
            if len(flock_boids) > 1:
                dists = [distance(b1, b2) for i, b1 in enumerate(flock_boids) for b2 in flock_boids[i+1:]]
                avg_dist = np.mean(dists)
                max_possible = np.sqrt(width**2 + height**2 + depth**2)
                cohesion = 1.0 - min(avg_dist / max_possible, 1.0)
            total_reward = (2*circle_reward + cohesion) / 3
            return total_reward, circle_reward, None, cohesion
        else:
            # Quadrant reward (default)
            in_quadrant = sum(1 for b in flock_boids if get_quadrant(b.x, b.y, b.z) == quadrant)
            quadrant_reward = in_quadrant / len(flock_boids)
            q = quadrant - 1
            qx = q & 1
            qy = (q >> 1) & 1
            qz = (q >> 2) & 1
            target_x = width/4 if qx == 0 else 3*width/4
            target_y = height/4 if qy == 0 else 3*height/4
            target_z = depth/4 if qz == 0 else 3*depth/4
            move_toward = 0
            for b in flock_boids:
                to_target = np.array([target_x - b.x, target_y - b.y, target_z - b.z])
                velocity = np.array([b.dx, b.dy, b.dz])
                if np.linalg.norm(to_target) > 0 and np.linalg.norm(velocity) > 0:
                    move_toward += np.dot(to_target, velocity) / (np.linalg.norm(to_target) * np.linalg.norm(velocity))
            move_toward_reward = (move_toward / len(flock_boids) + 1) / 2
            cohesion = 0
            if len(flock_boids) > 1:
                dists = [distance(b1, b2) for i, b1 in enumerate(flock_boids) for b2 in flock_boids[i+1:]]
                avg_dist = np.mean(dists)
                max_possible = np.sqrt(width**2 + height**2 + depth**2)
                cohesion = 1.0 - min(avg_dist / max_possible, 1.0)
            total_reward = (2*quadrant_reward + 2*move_toward_reward + cohesion) / 5
            return total_reward, quadrant_reward, move_toward_reward, cohesion

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

def get_leader_state(leader, quadrant):
    # State: positions/velocities of up to 5 nearest neighbors + quadrant
    neighbors = [b for b in boids if b.flock == leader.flock and b is not leader]
    neighbors = sorted(neighbors, key=lambda b: distance(leader, b))[:5]
    state = []
    for b in neighbors:
        state.extend([b.x - leader.x, b.y - leader.y, b.z - leader.z, b.dx, b.dy, b.dz])
    while len(state) < 6*5:
        state.append(0)
    state.append(quadrant)
    return np.array(state, dtype=np.float32)

def update_boids():
    if enable_leader:
        for i, leader in enumerate(leader_boids):
            if leader_control_mode == "trajectory":
                pos = leader_trajectory.get_position(global_step[0], leader_idx=i, total_leaders=len(leader_boids))
                # Update velocity for traces
                if len(leader.history) > 0:
                    prev = np.array(leader.history[-1])
                    leader.dx, leader.dy, leader.dz = pos - prev
                else:
                    leader.dx, leader.dy, leader.dz = 0, 0, 0
                leader.x, leader.y, leader.z = pos
                leader.history.append((leader.x, leader.y, leader.z))
                if len(leader.history) > 100:
                    leader.history.pop(0)
                continue
            # DQN logic
            quadrant = quadrant_assignments[leader.flock - 1]
            state = get_leader_state(leader, quadrant)
            action = shared_leader_agent.select_action(state)
            move_map = [
                np.array([1,0,0]), np.array([-1,0,0]),
                np.array([0,1,0]), np.array([0,-1,0]),
                np.array([0,0,1]), np.array([0,0,-1]),
                np.array([0,0,0])
            ]
            move = move_map[action]
            leader.dx += move[0]
            leader.dy += move[1]
            leader.dz += move[2]
            fly_towards_center(leader)
            avoid_others(leader)
            match_velocity(leader)
            limit_speed(leader)
            keep_within_bounds(leader)
            leader.x += leader.dx
            leader.y += leader.dy
            leader.z += leader.dz
            leader.history.append((leader.x, leader.y, leader.z))
            if len(leader.history) > 0:
                leader.history.pop(0)
            total_reward, quadrant_reward, move_toward_reward, cohesion = leader_reward(leader, quadrant)
            # Handle None values for reward logging
            q_str = f"{quadrant_reward:.3f}" if quadrant_reward is not None else "-"
            m_str = f"{move_toward_reward:.3f}" if move_toward_reward is not None else "-"
            # print(f"Leader {i+1} | Flock: {leader.flock} | Quadrant: {quadrant} | TotalReward: {total_reward:.3f} | QReward: {q_str} | MoveReward: {m_str} | CohesionReward: {cohesion:.3f}")
            next_state = get_leader_state(leader, quadrant)
            done = False
            shared_leader_agent.store(state, action, total_reward, next_state, done)
        # Train and log loss (once per frame, using pooled experiences)
        if leader_control_mode == "dqn":
            loss = shared_leader_agent.train()
            if loss is not None:
                loss_history.append(loss)
            if global_step[0] % 100 == 0:
                print(f"Step {global_step[0]} | Leaders | Loss: {loss if loss is not None else 'N/A'}")

    # All boids use standard rules if leader logic is off, otherwise only regular boids
    for boid in boids:
        if enable_leader and boid.is_leader:
            continue
        fly_towards_center(boid)
        avoid_others(boid)
        match_velocity(boid)
        limit_speed(boid)
        keep_within_bounds(boid)
        boid.x += boid.dx
        boid.y += boid.dy
        boid.z += boid.dz
        boid.history.append((boid.x, boid.y, boid.z))
        if len(boid.history) > 0:
            boid.history.pop(0)

    # Update predators
    if enable_predators:
        for predator in predators:
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
            predator.history.append((predator.x, predator.y, predator.z))
            if len(predator.history) > 0:
                predator.history.pop(0)



# Step counter for saving models
global_step = [0]
loss_history = []

def animate(frame, ax, fig):
    update_boids()
    global_step[0] += 1
    # Save shared model every model_save_interval steps
    if global_step[0] % model_save_interval == 0:
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, f"shared_leader_agent_step{global_step[0]}.pth")
        shared_leader_agent.save(save_path)
        print(f"Saved shared_leader_agent at step {global_step[0]} to {save_path}")
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
    # Draw quadrant indicators for each flock if enabled
    if show_quadrant_visu:
        for i, flock_id in enumerate(range(1, num_flocks + 1)):
            color = flock_colors[(flock_id - 1) % len(flock_colors)]
            quadrant = quadrant_assignments[flock_id - 1]
            # Decode quadrant index (1-8) to qx, qy, qz
            q = quadrant - 1
            qx = q & 1
            qy = (q >> 1) & 1
            qz = (q >> 2) & 1
            x0 = 0 if qx == 0 else width/2
            x1 = width/2 if qx == 0 else width
            y0 = 0 if qy == 0 else height/2
            y1 = height/2 if qy == 0 else height
            z0 = 0 if qz == 0 else depth/2
            z1 = depth/2 if qz == 0 else depth
            # Draw wireframe box for quadrant
            for s, e in [((x0,y0,z0),(x1,y0,z0)), ((x0,y0,z0),(x0,y1,z0)), ((x0,y0,z0),(x0,y0,z1)),
                        ((x1,y1,z1),(x0,y1,z1)), ((x1,y1,z1),(x1,y0,z1)), ((x1,y1,z1),(x1,y1,z0)),
                        ((x1,y0,z0),(x1,y1,z0)), ((x1,y0,z0),(x1,y0,z1)), ((x0,y1,z0),(x1,y1,z0)),
                        ((x0,y1,z0),(x0,y1,z1)), ((x0,y0,z1),(x1,y0,z1)), ((x0,y0,z1),(x0,y1,z1))]:
                ax.plot([s[0],e[0]], [s[1],e[1]], [s[2],e[2]], color=color, alpha=0.3, linewidth=2)
    # Draw each flock
    for flock_id in range(1, num_flocks + 1):
        xs = [b.x for b in boids if b.flock == flock_id]
        ys = [b.y for b in boids if b.flock == flock_id]
        zs = [b.z for b in boids if b.flock == flock_id]
        color = flock_colors[(flock_id - 1) % len(flock_colors)]
        ax.scatter(xs, ys, zs, color=color, label=f"Flock {flock_id}")
    # Draw leader auras as simple transparent spheres
    for leader in leader_boids:
        color = flock_colors[(leader.flock - 1) % len(flock_colors)]
        aura_radius = 30
        ax.scatter(leader.x, leader.y, leader.z, s=aura_radius**2, color=color, alpha=0.18, edgecolors='none')
    # Draw traces for each boid
    for b in boids:
        if len(b.history) > 1:
            h = np.array(b.history)
            color = flock_colors[(b.flock - 1) % len(flock_colors)]
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

def run_visualization(save_gif=False):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    global ani
    ani = FuncAnimation(fig, lambda frame: animate(frame, ax, fig), interval=16, cache_frame_data=False)
    if save_gif:
        import matplotlib.animation as animation
        vid_dir = "vid"
        if not os.path.exists(vid_dir):
            os.makedirs(vid_dir)
        gif_path = os.path.join(vid_dir, "boid_simulation.gif")
        writer = animation.PillowWriter(fps=30)
        ani.save(gif_path, writer=writer)
        print(f"GIF saved to {gif_path}")
    else:
        plt.show()
    # Show loss curve in a second figure
    if loss_history:
        plt.figure()
        plt.plot(loss_history, label='DQN Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('DQN Training Loss Curve')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Set save_gif=True to save GIF, False to show interactively
    run_visualization(save_gif=False)
