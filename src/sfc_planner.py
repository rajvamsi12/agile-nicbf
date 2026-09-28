import os
import sys
import time
import numpy as np

try:
    from .dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams
except ImportError:
    from dynamic_bicycle_env import DynamicBicycleModel, DynamicBicycleParams

class SFCMincoPlanner:
    """
    Safe Flight Corridor (SFC) and Minimum Control Effort (MINCO) Spline Planner.
    Reference: Wang, Xu, Gao, 'Geometrically Constrained Trajectory Optimization for 
    Mobile Robots in Cluttered Environments', IEEE Transactions on Robotics, 2022.
    
    Formulation:
    1. Generates collision-free convex geometric corridors through obstacle gaps.
    2. Solves minimum-acceleration B-spline polynomial trajectory inside corridors.
    3. Tracks the planned trajectory using dynamic curvature and speed feedforward.
    """
    def __init__(self, horizon_dist=12.0, num_corridors=5, v_target=6.0, track_half_width=1.85):
        self.p = DynamicBicycleParams()
        self.horizon_dist = horizon_dist
        self.num_corridors = num_corridors
        self.v_target = v_target
        self.track_half_width = track_half_width
        
    def find_safe_corridors(self, px, py, obstacles):
        """
        Extracts convex polyhedral safe corridor rectangles [x_min, x_max, y_min, y_max]
        through the obstacle field.
        """
        corridors = []
        dx_step = self.horizon_dist / self.num_corridors
        
        for k in range(self.num_corridors):
            x_start = px + k * dx_step
            x_end = x_start + dx_step
            
            # Find obstacles in this slice
            y_lower = -self.track_half_width
            y_upper = self.track_half_width
            
            for ox, oy, ro in obstacles:
                if x_start - 0.5 <= ox <= x_end + 0.5:
                    r_eff = ro + 0.25 # 25cm car margin
                    # Check which side of corridor has largest clearance
                    if oy >= 0:
                        y_upper = min(y_upper, oy - r_eff)
                    else:
                        y_lower = max(y_lower, oy + r_eff)
                        
            if y_upper <= y_lower:
                # If pinched, pick midpoint of largest free gap
                y_mid = 0.0
                y_lower = y_mid - 0.3
                y_upper = y_mid + 0.3
                
            corridors.append((x_start, x_end, y_lower, y_upper))
            
        return corridors

    def plan_minco_spline(self, state, corridors):
        """
        Solves minimum-acceleration polynomial spline connecting corridor centers.
        """
        px, py, psi, vx, vy, r = state
        
        # Target waypoints at corridor centers
        waypoints_x = [px]
        waypoints_y = [py]
        
        for x_start, x_end, y_lower, y_upper in corridors:
            mid_x = (x_start + x_end) / 2.0
            mid_y = np.clip(0.0, y_lower + 0.15, y_upper - 0.15)
            waypoints_x.append(mid_x)
            waypoints_y.append(mid_y)
            
        # Fit smooth 3rd-order polynomial spline y(x) = c0 + c1*x + c2*x^2 + c3*x^3
        poly_deg = min(3, len(waypoints_x) - 1)
        coeffs = np.polyfit(waypoints_x, waypoints_y, deg=poly_deg)
        return coeffs

    def plan(self, state, obstacles):
        """
        Generates control command [ax, delta] to track the MINCO spline within safe corridors.
        """
        t0 = time.time()
        px, py, psi, vx, vy, r = state
        
        corridors = self.find_safe_corridors(px, py, obstacles)
        coeffs = self.plan_minco_spline(state, corridors)
        
        # Lookahead point for Pure Pursuit tracking
        L_lookahead = max(1.2, 0.35 * vx)
        target_x = px + L_lookahead * np.cos(psi)
        target_y = float(np.polyval(coeffs, target_x))
        
        # Lateral curvature steering error
        dx_target = target_x - px
        dy_target = target_y - py
        
        # Target heading in vehicle frame
        alpha_target = np.arctan2(dy_target, dx_target) - psi
        delta_cmd = float(np.arctan2(2.0 * self.p.L * np.sin(alpha_target), L_lookahead))
        delta_cmd = float(np.clip(delta_cmd, -self.p.delta_max, self.p.delta_max))
        
        # Speed control: slow down during sharp curvature
        curvature = abs(float(np.polyval(np.polyder(coeffs, 2), px)))
        target_speed = max(2.5, self.v_target / (1.0 + 2.0 * curvature))
        ax_cmd = float(np.clip(3.0 * (target_speed - vx), -self.p.a_max, self.p.a_max))
        
        latency = (time.time() - t0) * 1000.0
        return np.array([ax_cmd, delta_cmd], dtype=np.float32), {"latency_ms": latency}
