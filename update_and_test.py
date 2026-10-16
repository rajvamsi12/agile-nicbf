import re

# 1. Update safe_controller.py
with open('/home/raj/agile_nicbf/safe_controller.py', 'r') as f:
    sc = f.read()

# Update hazard_horizon to 6.0m
old_hazard = 'hazard_horizon = min(4.5, max(1.5, 0.6 * vx)) + lf'
new_hazard = 'hazard_horizon = min(6.0, max(2.5, 0.95 * vx)) + lf'
if old_hazard in sc:
    sc = sc.replace(old_hazard, new_hazard)
    print(Updated hazard_horizon to 6.0m)
else:
    print(Warning: old_hazard string not found directly)

# Update obstacle filter bounds
old_filter = 'if not (-r_eff - lr <= dx_body <= hazard_horizon and abs(dy_body) < (r_eff + 0.60)):'
new_filter = 'if not (-r_eff - lr - 0.10 <= dx_body <= hazard_horizon and abs(dy_body) < (r_eff + 0.80)):'
if old_filter in sc:
    sc = sc.replace(old_filter, new_filter)
    print(Updated obstacle filter bounds)
else:
    print(Warning: old_filter string not found directly)

with open('/home/raj/agile_nicbf/safe_controller.py', 'w') as f:
    f.write(sc)

# 2. Update dynamic_bicycle_env.py to restore exact GitHub simulator values
with open('/home/raj/agile_nicbf/dynamic_bicycle_env.py', 'r') as f:
    dbe = f.read()

# Make ClutteredRacingEnv default to clean github values (no noise, no motor lag)
dbe = re.sub(
    r'def __init__\(self, num_obstacles=10, track_length=50\.0, track_width=4\.0, seed=42,\s*tau_motor=[^,]+,\s*sv_max=[^,]+,\s*r_sense=[^,]+,\s*enable_noise=[^)]+\):',
    'def __init__(self, num_obstacles=10, track_length=50.0, track_width=4.0, seed=42,\n                 tau_motor=0.0, sv_max=3.2, r_sense=15.0, enable_noise=False):',
    dbe
)

# Constant friction nominal 1.0489 (exact GitHub default)
dbe = re.sub(
    r'self\.model\.p\.mu = float\(self\.rng\.uniform\(0\.95, 1\.05\) \* 1\.0489\)',
    'self.model.p.mu = 1.0489',
    dbe
)

with open('/home/raj/agile_nicbf/dynamic_bicycle_env.py', 'w') as f:
    f.write(dbe)

print(dynamic_bicycle_env.py updated to GitHub defaults)
