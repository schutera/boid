import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

try:
    from PIL import Image
except ImportError:  # Pillow is optional unless render_frame is used
    Image = None

# Minimal parameters
width, height, depth = 80, 80, 80
num_boids = 10
num_flocks = 2
visual_range = 36  # Slightly increased to encourage more interaction
speed_limit = 4.0  # Allow faster movement
margin = 10
turn_factor = 1.0  # Stronger turning at boundaries
centering_factor = 0.02  # Stronger pull towards center
avoid_factor = 0.12  # Stronger avoidance
matching_factor = 0.12  # More velocity matching
min_distance = 10  # Slightly reduced to allow closer flying
flock_colors = ["#e76f51", "#2a9d8f", "#ffffff", "#264653", "#e9c46a", "#f4a261"]
jitter_strength = 0.5  # More random movement for liveliness



class Boid:
    def __init__(self, flock=1):
        self.x = np.random.rand() * width
        self.y = np.random.rand() * height
        self.z = np.random.rand() * depth
        self.dx = np.random.rand() * 8 - 4
        self.dy = np.random.rand() * 8 - 4
        self.dz = np.random.rand() * 8 - 4
        self.flock = flock

    def position(self):
        return np.array([self.x, self.y, self.z])

    def velocity(self):
        return np.array([self.dx, self.dy, self.dz])

boids = []
for flock_id in range(1, num_flocks + 1):
    boids.extend([Boid(flock=flock_id) for _ in range(num_boids)])

def distance(b1, b2):
    return np.sqrt((b1.x - b2.x)**2 + (b1.y - b2.y)**2 + (b1.z - b2.z)**2)

def fly_towards_center(boid):
    center_x, center_y, center_z, num_neighbors = 0, 0, 0, 0
    for other in boids:
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
    move_x, move_y, move_z = 0, 0, 0
    for other in boids:
        if other is not boid and distance(boid, other) < min_distance:
            move_x += boid.x - other.x
            move_y += boid.y - other.y
            move_z += boid.z - other.z
    boid.dx += move_x * avoid_factor
    boid.dy += move_y * avoid_factor
    boid.dz += move_z * avoid_factor

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
        boid.dx += (avg_dx - boid.dx) * matching_factor
        boid.dy += (avg_dy - boid.dy) * matching_factor
        boid.dz += (avg_dz - boid.dz) * matching_factor

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

def step_simulation():
    for boid in boids:
        fly_towards_center(boid)
        avoid_others(boid)
        match_velocity(boid)
        # Introduce a small amount of random drift to keep the flock lively without heavy compute.
        boid.dx += np.random.uniform(-jitter_strength, jitter_strength)
        boid.dy += np.random.uniform(-jitter_strength, jitter_strength)
        boid.dz += np.random.uniform(-jitter_strength, jitter_strength)
        limit_speed(boid)
        keep_within_bounds(boid)
        boid.x += boid.dx
        boid.y += boid.dy
        boid.z += boid.dz

def draw_scene(ax, fig=None, edge_buffer=0):
    ax.cla()
    ax.set_facecolor('black')
    if fig is not None:
        fig.patch.set_facecolor('black')
        fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0 - edge_buffer, width + edge_buffer)
    ax.set_ylim(0 - edge_buffer, height + edge_buffer)
    ax.set_zlim(0 - edge_buffer, depth + edge_buffer)
    ax.margins(0)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    # Draw each flock with depth-scaled size and color
    from matplotlib.colors import to_rgb
    for flock_id in range(1, num_flocks + 1):
        base_rgb = np.array(to_rgb(flock_colors[(flock_id - 1) % len(flock_colors)]))
        fb = [b for b in boids if b.flock == flock_id]
        xs = [b.x for b in fb]
        ys = [b.y for b in fb]
        zs = [b.z for b in fb]
        # near (y=0) → large+bright, far (y=depth) → small+gray
        t = [1 - b.y / depth for b in fb]  # 1=near, 0=far
        s = [3 + ti * 25 for ti in t]
        # Blend toward gray (0.3, 0.3, 0.3) for far boids, full color for near
        gray = np.array([0.3, 0.3, 0.3])
        colors = [gray + ti * (base_rgb - gray) for ti in t]
        ax.scatter(xs, ys, zs, color=colors, s=s, depthshade=False)


def animate(frame, ax, fig):
    step_simulation()
    draw_scene(ax, fig)

def create_figure(figsize=(7, 7), edge_buffer=0):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0, 0, 1, 1])
    try:
        ax.set_box_aspect((width, height, depth))
    except AttributeError:
        pass
    draw_scene(ax, fig, edge_buffer=edge_buffer)
    return fig, ax


def render_frame(fig, ax, edge_buffer=0, auto_step=True, resize=None):
    if Image is None:
        raise ImportError(
            "Pillow is required to render frames to images. Install it via "
            "pip install pillow before calling render_frame()."
        )
    if auto_step:
        step_simulation()
    draw_scene(ax, fig, edge_buffer=edge_buffer)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    try:
        rgb_bytes = fig.canvas.tostring_rgb()
        image = Image.frombytes('RGB', (width, height), rgb_bytes)
    except AttributeError:
        # Matplotlib >=3.9 removed tostring_rgb; buffer_rgba() is the supported path.
        rgba = fig.canvas.buffer_rgba()
        image = Image.frombuffer('RGBA', (width, height), rgba, 'raw', 'RGBA', 0, 1).convert('RGB')
    if resize is not None:
        image = image.resize(resize, Image.LANCZOS)
    return image

def run_interactive(interval=30, figsize=(7, 7)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(
        fig,
        animate,
        fargs=(ax, fig),
        interval=interval,
        cache_frame_data=False,
    )
    plt.show()
    return ani


if __name__ == "__main__":
    run_interactive()
