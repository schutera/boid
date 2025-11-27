import numpy as np

def fit_circle_3d(points):
    # Fit a plane to the points using SVD
    centroid = np.mean(points, axis=0)
    pts_centered = points - centroid
    _, _, vh = np.linalg.svd(pts_centered)
    normal = vh[2]
    # Project points onto the plane
    def project_to_plane(pt):
        return pt - np.dot(pt - centroid, normal) * normal
    projected = np.array([project_to_plane(pt) for pt in points])
    # Find 2 orthogonal axes in the plane
    u = vh[0]
    v = vh[1]
    # Express projected points in plane coordinates
    plane_coords = np.array([[np.dot(pt-centroid, u), np.dot(pt-centroid, v)] for pt in projected])
    # Fit 2D circle to plane_coords
    A = np.hstack((2*plane_coords, np.ones((len(plane_coords),1))))
    b = np.sum(plane_coords**2, axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    center_2d = sol[:2]
    radius = np.sqrt(sol[2] + center_2d[0]**2 + center_2d[1]**2)
    # Convert center_2d back to 3D
    center_3d = centroid + center_2d[0]*u + center_2d[1]*v
    return center_3d, radius, normal, u, v

def circularity_reward(flock_boids):
    positions = np.array([[b.x, b.y, b.z] for b in flock_boids])
    velocities = np.array([[b.dx, b.dy, b.dz] for b in flock_boids])
    if len(positions) < 3:
        return 0.0
    center, radius, normal, u, v = fit_circle_3d(positions)
    # Proximity to circle
    projected = np.array([pos - np.dot(pos - center, normal) * normal for pos in positions])
    plane_coords = np.array([[np.dot(pt-center, u), np.dot(pt-center, v)] for pt in projected])
    dists = np.abs(np.sqrt(np.sum(plane_coords**2, axis=1)) - radius)
    proximity_reward = 1.0 - min(np.mean(dists) / (0.2*radius+1e-6), 1.0)  # normalized to 20% of radius
    # Tangential velocity alignment
    tangential_rewards = []
    streak_rewards = []
    for i, pt in enumerate(plane_coords):
        if np.linalg.norm(pt) == 0:
            tangential_rewards.append(0)
            streak_rewards.append(0)
            if hasattr(flock_boids[i], 'circularity_streak'):
                flock_boids[i].circularity_streak = 0
            continue
        tangent_2d = np.array([-pt[1], pt[0]]) / np.linalg.norm(pt)
        tangent_3d = tangent_2d[0]*u + tangent_2d[1]*v
        vel = velocities[i]
        if np.linalg.norm(vel) == 0:
            tangential_rewards.append(0)
            streak_rewards.append(0)
            if hasattr(flock_boids[i], 'circularity_streak'):
                flock_boids[i].circularity_streak = 0
            continue
        tangential = np.dot(vel, tangent_3d) / np.linalg.norm(vel)
        tangential_score = (tangential + 1) / 2
        tangential_rewards.append(tangential_score)
        # Streak logic: consider boid "following circle" if close and tangential
        close_enough = dists[i] < 0.2*radius
        moving_tangent = tangential_score > 0.7
        if close_enough and moving_tangent:
            if not hasattr(flock_boids[i], 'circularity_streak'):
                flock_boids[i].circularity_streak = 1
            else:
                flock_boids[i].circularity_streak += 1
        else:
            flock_boids[i].circularity_streak = 0
        # Reward increases with streak length (e.g., up to 2x for 50+ steps)
        streak_multiplier = min(1.0 + 0.02 * flock_boids[i].circularity_streak, 2.0)
        streak_rewards.append(streak_multiplier)
    tangential_reward = np.mean(tangential_rewards)
    avg_streak_multiplier = np.mean(streak_rewards)
    # Merge both rewards (average), then scale by streak multiplier
    circular_reward = ((proximity_reward + tangential_reward) / 2) * avg_streak_multiplier
    return circular_reward
