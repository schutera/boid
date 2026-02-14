"""
Standalone desktop visualization of the 3D boid simulation.

Usage:
    python visualize_boids.py                          # default: 10 boids, 1 flock, trajectory leaders
    python visualize_boids.py --num-boids 30 --num-flocks 3
    python visualize_boids.py --checkpoint checkpoints/shared_leader_agent_step200000.pth
    python visualize_boids.py --no-leaders
    python visualize_boids.py --save output.gif --frames 300
    python3 visualize_boids.py --tft --num-flocks 2 --no-traces   # run on ST7735 TFT
"""

import argparse
import os
import sys
import time
import numpy as np

# Defer matplotlib import until we know the backend
_matplotlib_imported = False

def _setup_matplotlib(use_agg=False):
    global plt, FuncAnimation, _matplotlib_imported
    if _matplotlib_imported:
        return
    import matplotlib
    if use_agg:
        matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.animation import FuncAnimation as _FA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    plt = _plt
    FuncAnimation = _FA
    _matplotlib_imported = True

from leader_trajectory import LeaderTrajectory


# ---------------------------------------------------------------------------
# Simulation defaults (overridden by CLI args)
# ---------------------------------------------------------------------------
WIDTH, HEIGHT, DEPTH = 500, 500, 500
VISUAL_RANGE = 75
SPEED_LIMIT = 5
MARGIN = 20
TURN_FACTOR = 2
CENTERING_FACTOR = 0.005
AVOID_FACTOR_INTRA = 0.05
AVOID_FACTOR_INTER = 0.15
MATCHING_FACTOR_INTRA = 0.05
MIN_DISTANCE = 20
FLOCK_COLORS = [
    "#00c3ff", "#ffffff", "#264653", "#2a9d8f",
    "#e9c46a", "#f4a261", "#e76f51",
]
TRACE_LENGTH = 50


class Boid:
    def __init__(self, flock=1, is_leader=False):
        self.x = np.random.rand() * WIDTH
        self.y = np.random.rand() * HEIGHT
        self.z = np.random.rand() * DEPTH
        self.dx = np.random.rand() * 10 - 5
        self.dy = np.random.rand() * 10 - 5
        self.dz = np.random.rand() * 10 - 5
        self.flock = flock
        self.is_leader = is_leader
        self.history = []


def distance(b1, b2):
    return np.sqrt((b1.x - b2.x)**2 + (b1.y - b2.y)**2 + (b1.z - b2.z)**2)


def fly_towards_center(boid, boids):
    cx, cy, cz, n = 0, 0, 0, 0
    for other in boids:
        if other.flock == boid.flock and distance(boid, other) < VISUAL_RANGE:
            cx += other.x; cy += other.y; cz += other.z; n += 1
    if n:
        boid.dx += (cx/n - boid.x) * CENTERING_FACTOR
        boid.dy += (cy/n - boid.y) * CENTERING_FACTOR
        boid.dz += (cz/n - boid.z) * CENTERING_FACTOR


def avoid_others(boid, boids):
    mx_in, my_in, mz_in = 0, 0, 0
    mx_out, my_out, mz_out = 0, 0, 0
    for other in boids:
        if other is not boid and distance(boid, other) < MIN_DISTANCE:
            if other.flock == boid.flock:
                mx_in += boid.x - other.x; my_in += boid.y - other.y; mz_in += boid.z - other.z
            else:
                mx_out += boid.x - other.x; my_out += boid.y - other.y; mz_out += boid.z - other.z
    boid.dx += mx_in * AVOID_FACTOR_INTRA + mx_out * AVOID_FACTOR_INTER
    boid.dy += my_in * AVOID_FACTOR_INTRA + my_out * AVOID_FACTOR_INTER
    boid.dz += mz_in * AVOID_FACTOR_INTRA + mz_out * AVOID_FACTOR_INTER


def match_velocity(boid, boids):
    ax, ay, az, n = 0, 0, 0, 0
    for other in boids:
        if other.flock == boid.flock and distance(boid, other) < VISUAL_RANGE:
            ax += other.dx; ay += other.dy; az += other.dz; n += 1
    if n:
        boid.dx += (ax/n - boid.dx) * MATCHING_FACTOR_INTRA
        boid.dy += (ay/n - boid.dy) * MATCHING_FACTOR_INTRA
        boid.dz += (az/n - boid.dz) * MATCHING_FACTOR_INTRA


def limit_speed(boid):
    speed = np.sqrt(boid.dx**2 + boid.dy**2 + boid.dz**2)
    if speed > SPEED_LIMIT:
        boid.dx = (boid.dx / speed) * SPEED_LIMIT
        boid.dy = (boid.dy / speed) * SPEED_LIMIT
        boid.dz = (boid.dz / speed) * SPEED_LIMIT


def keep_within_bounds(boid):
    if boid.x < MARGIN:   boid.dx += TURN_FACTOR * (MARGIN - boid.x) / MARGIN
    if boid.x > WIDTH - MARGIN:  boid.dx -= TURN_FACTOR * (boid.x - (WIDTH - MARGIN)) / MARGIN
    if boid.y < MARGIN:   boid.dy += TURN_FACTOR * (MARGIN - boid.y) / MARGIN
    if boid.y > HEIGHT - MARGIN: boid.dy -= TURN_FACTOR * (boid.y - (HEIGHT - MARGIN)) / MARGIN
    if boid.z < MARGIN:   boid.dz += TURN_FACTOR * (MARGIN - boid.z) / MARGIN
    if boid.z > DEPTH - MARGIN:  boid.dz -= TURN_FACTOR * (boid.z - (DEPTH - MARGIN)) / MARGIN


def get_leader_state(leader, boids):
    neighbors = sorted(
        [b for b in boids if b.flock == leader.flock and b is not leader],
        key=lambda b: distance(leader, b),
    )[:5]
    state = []
    for b in neighbors:
        state.extend([b.x - leader.x, b.y - leader.y, b.z - leader.z, b.dx, b.dy, b.dz])
    while len(state) < 30:
        state.append(0)
    state.append(1)  # dummy quadrant indicator
    return np.array(state, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Standalone 3D boid visualization")
    p.add_argument("--num-boids", type=int, default=10, help="Boids per flock (default: 10)")
    p.add_argument("--num-flocks", type=int, default=1, help="Number of flocks (default: 1)")
    p.add_argument("--num-leaders", type=int, default=3, help="Leaders per flock (default: 3)")
    p.add_argument("--no-leaders", action="store_true", help="Disable leader boids entirely")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to DQN checkpoint for leader control")
    p.add_argument("--save", type=str, default=None, metavar="PATH", help="Save animation as GIF to PATH")
    p.add_argument("--frames", type=int, default=300, help="Number of frames when saving GIF (default: 300)")
    p.add_argument("--no-traces", action="store_true", help="Disable boid trace trails")
    p.add_argument("--rotate", action="store_true", help="Auto-rotate camera")
    p.add_argument("--tft", action="store_true", help="Output to ST7735 TFT display (Raspberry Pi)")
    return p.parse_args()


def main():
    args = parse_args()

    enable_leaders = not args.no_leaders
    use_dqn = args.checkpoint is not None and enable_leaders
    num_leaders = args.num_leaders if enable_leaders else 0

    # Build boids
    boids = []
    leader_boids = []
    for flock_id in range(1, args.num_flocks + 1):
        for _ in range(num_leaders):
            leader = Boid(flock=flock_id, is_leader=True)
            boids.append(leader)
            leader_boids.append(leader)
        for _ in range(max(0, args.num_boids - num_leaders)):
            boids.append(Boid(flock=flock_id))

    # Trajectory controller (used when no DQN checkpoint)
    trajectory = LeaderTrajectory(
        trajectory_type="circle",
        center=[WIDTH/2, HEIGHT/2, DEPTH/2],
        radius=min(WIDTH, HEIGHT, DEPTH) * 0.3,
        speed=0.02,
        axis="z",
    )

    # DQN agent (optional)
    agent = None
    if use_dqn:
        from dqn_agent import DQNAgent
        state_dim = 6 * 5 + 1
        action_dim = 7
        agent = DQNAgent(state_dim, action_dim)
        if os.path.exists(args.checkpoint):
            agent.load(args.checkpoint)
            agent.epsilon = 0.0  # greedy for visualization
            print(f"Loaded DQN checkpoint from {args.checkpoint}")
        else:
            print(f"Checkpoint {args.checkpoint} not found — falling back to trajectory mode")
            agent = None

    step = [0]
    move_map = [
        np.array([1,0,0]), np.array([-1,0,0]),
        np.array([0,1,0]), np.array([0,-1,0]),
        np.array([0,0,1]), np.array([0,0,-1]),
        np.array([0,0,0]),
    ]

    def update():
        # Leader updates
        if enable_leaders:
            for i, leader in enumerate(leader_boids):
                if agent is not None:
                    # DQN control
                    state = get_leader_state(leader, boids)
                    action = agent.select_action(state)
                    move = move_map[action]
                    leader.dx += move[0]; leader.dy += move[1]; leader.dz += move[2]
                    fly_towards_center(leader, boids)
                    avoid_others(leader, boids)
                    match_velocity(leader, boids)
                    limit_speed(leader)
                    keep_within_bounds(leader)
                    leader.x += leader.dx; leader.y += leader.dy; leader.z += leader.dz
                else:
                    # Trajectory control
                    pos = trajectory.get_position(step[0], leader_idx=i, total_leaders=len(leader_boids))
                    if leader.history:
                        prev = np.array(leader.history[-1])
                        leader.dx, leader.dy, leader.dz = pos - prev
                    else:
                        leader.dx = leader.dy = leader.dz = 0
                    leader.x, leader.y, leader.z = pos
                leader.history.append((leader.x, leader.y, leader.z))
                if len(leader.history) > TRACE_LENGTH:
                    leader.history.pop(0)

        # Regular boid updates
        for boid in boids:
            if boid.is_leader:
                continue
            fly_towards_center(boid, boids)
            avoid_others(boid, boids)
            match_velocity(boid, boids)
            limit_speed(boid)
            keep_within_bounds(boid)
            boid.x += boid.dx; boid.y += boid.dy; boid.z += boid.dz
            boid.history.append((boid.x, boid.y, boid.z))
            if len(boid.history) > TRACE_LENGTH:
                boid.history.pop(0)

        step[0] += 1

    def animate(frame, ax, fig):
        update()
        ax.cla()
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")
        ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.set_zlim(0, DEPTH)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_axis_off()

        # Compute camera-relative depth for size scaling
        # Project all boid positions to get screen-space depth
        from mpl_toolkits.mplot3d import proj3d
        all_positions = np.array([[b.x, b.y, b.z] for b in boids])
        if len(all_positions) > 0:
            # Get the full 3D→2D projection matrix
            proj_mat = ax.get_proj()
            # Transform to homogeneous coords and project
            ones = np.ones((len(all_positions), 1))
            coords_h = np.hstack([all_positions, ones])  # Nx4
            projected = coords_h @ proj_mat.T  # Nx4
            # z_screen: lower = closer to camera, higher = farther
            z_screen = projected[:, 2]
            # Normalize to [0, 1] range (0 = nearest, 1 = farthest)
            z_min, z_max = z_screen.min(), z_screen.max()
            if z_max > z_min:
                z_norm = (z_screen - z_min) / (z_max - z_min)
            else:
                z_norm = np.full_like(z_screen, 0.5)
            # Map to sizes: nearest → large (60), farthest → small (5)
            sizes = 5 + (1 - z_norm) * 55
            # Also scale alpha: nearest → brighter, farthest → dimmer
            alphas = 0.3 + (1 - z_norm) * 0.7
        else:
            sizes = np.array([])
            alphas = np.array([])

        # Build a lookup from boid index to depth-scaled size/alpha
        boid_sizes = {id(b): (sizes[i], alphas[i]) for i, b in enumerate(boids)}

        # Draw flocks with depth-scaled sizes
        for flock_id in range(1, args.num_flocks + 1):
            color = FLOCK_COLORS[(flock_id - 1) % len(FLOCK_COLORS)]
            flock_boids = [b for b in boids if b.flock == flock_id]
            xs = [b.x for b in flock_boids]
            ys = [b.y for b in flock_boids]
            zs = [b.z for b in flock_boids]
            s = [boid_sizes[id(b)][0] for b in flock_boids]
            a = [boid_sizes[id(b)][1] for b in flock_boids]
            # scatter with per-point alpha via RGBA colors
            import matplotlib.colors as mcolors
            rgba_base = np.array(mcolors.to_rgba(color))
            colors_arr = np.tile(rgba_base, (len(flock_boids), 1))
            for j in range(len(flock_boids)):
                colors_arr[j, 3] = a[j]
            ax.scatter(xs, ys, zs, color=colors_arr, s=s, depthshade=False)

        # Leader auras (also depth-scaled)
        for leader in leader_boids:
            color = FLOCK_COLORS[(leader.flock - 1) % len(FLOCK_COLORS)]
            ls, la = boid_sizes.get(id(leader), (30, 0.5))
            aura_s = ls * 15  # scale aura proportionally
            ax.scatter(leader.x, leader.y, leader.z, s=aura_s, color=color, alpha=la * 0.2, edgecolors="none", depthshade=False)

        # Traces
        if not args.no_traces:
            for b in boids:
                if len(b.history) > 1:
                    h = np.array(b.history)
                    color = FLOCK_COLORS[(b.flock - 1) % len(FLOCK_COLORS)]
                    ax.plot(h[:, 0], h[:, 1], h[:, 2], color=color, alpha=0.4, linewidth=0.8)

        # Auto-rotate camera
        if args.rotate:
            ax.view_init(elev=25, azim=step[0] * 0.5)

    # Set up matplotlib backend
    _setup_matplotlib(use_agg=args.tft or args.save)

    # Set up figure
    if args.tft:
        fig = plt.figure(figsize=(2.0, 1.6))
    else:
        fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0, 0, 1, 1])

    if args.tft:
        # --- TFT output mode: render frames via Agg, push to ST7735 ---
        # Add rbpi/ to path so we can import run_on_tft
        rbpi_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rbpi")
        if rbpi_dir not in sys.path:
            sys.path.insert(0, rbpi_dir)
        from run_on_tft import initialize_display, show_text
        from PIL import Image

        device, gpio_backend = initialize_display()
        show_text(device, "Boids!", f"{args.num_flocks} flocks")
        time.sleep(1.0)

        print("[Press CTRL+C to stop]")
        try:
            while True:
                animate(step[0], ax, fig)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                try:
                    raw = fig.canvas.tostring_rgb()
                    frame = Image.frombytes("RGB", (w, h), raw)
                except AttributeError:
                    rgba = fig.canvas.buffer_rgba()
                    frame = Image.frombuffer("RGBA", (w, h), rgba, "raw", "RGBA", 0, 1).convert("RGB")
                frame = frame.resize((device.width, device.height), Image.LANCZOS)
                device.display(frame)
                time.sleep(0.035)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                device.clear()
                device.show()
            except Exception:
                pass
            if gpio_backend is not None:
                try:
                    gpio_backend.cleanup()
                except Exception:
                    pass
    elif args.save:
        from matplotlib.animation import PillowWriter
        _ani = FuncAnimation(fig, lambda f: animate(f, ax, fig), frames=args.frames, interval=33, cache_frame_data=False)
        save_dir = os.path.dirname(args.save)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        _ani.save(args.save, writer=PillowWriter(fps=30))
        print(f"Saved {args.frames}-frame GIF to {args.save}")
    else:
        _ani = FuncAnimation(fig, lambda f: animate(f, ax, fig), interval=33, cache_frame_data=False)
        plt.show()

    return locals().get("_ani")  # prevent garbage collection


if __name__ == "__main__":
    _keep = main()
