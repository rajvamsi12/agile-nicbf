with open('/home/raj/agile_nicbf/src/dynamic_bicycle_env.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if 'def step_rk4_torch(self, state, control, dt=0.02):' in line:
        new_lines.append(line)
        new_lines.append(lines[i+1])
        new_lines.append('        a_lim = min(self.p.a_max, self.p.mu * self.p.g)\n')
        new_lines.append('        ax_safe = torch.clamp(control[:, 0], -a_lim, a_lim)\n')
        # skip next 2 lines
        continue
    if i > 0 and 'def step_rk4_torch' in lines[i-1]:
        continue # skip docstring
    if i > 1 and 'def step_rk4_torch' in lines[i-2]:
        continue # skip ax_safe
        
    if 'def step_rk4_numpy(self, state, control, dt=0.02):' in line:
        new_lines.append(line)
        new_lines.append(lines[i+1])
        new_lines.append('        a_lim = min(self.p.a_max, self.p.mu * self.p.g)\n')
        new_lines.append('        u_safe = np.array([\n')
        new_lines.append('            np.clip(control[0], -a_lim, a_lim),\n')
        continue
    if i > 0 and 'def step_rk4_numpy' in lines[i-1]:
        continue # skip docstring
    if i > 1 and 'def step_rk4_numpy' in lines[i-2]:
        continue # skip u_safe = np.array([
    if i > 2 and 'def step_rk4_numpy' in lines[i-3]:
        continue # skip np.clip(control[0]

    if 'cmd_ax = float(np.clip(control[0], -self.model.p.a_max, self.model.p.a_max))' in line:
        new_lines.append('        a_lim = min(self.model.p.a_max, self.model.p.mu * self.model.p.g)\n')
        new_lines.append('        cmd_ax = float(np.clip(control[0], -a_lim, a_lim))\n')
        continue
    if 'self.actual_ax = float(np.clip(self.actual_ax, -self.model.p.a_max, self.model.p.a_max))' in line:
        new_lines.append('        self.actual_ax = float(np.clip(self.actual_ax, -a_lim, a_lim))\n')
        continue

    new_lines.append(line)

with open('/home/raj/agile_nicbf/src/dynamic_bicycle_env.py', 'w') as f:
    f.writelines(new_lines)

with open('/home/raj/agile_nicbf/dynamic_bicycle_env.py', 'w') as f:
    f.writelines(new_lines)

print('Patch applied successfully!')
