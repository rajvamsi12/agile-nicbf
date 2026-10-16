import sys, collections; sys.path[:0] = ['.', 'src']
from stress_grid import load
def row(name, R):
    c = collections.Counter(r['outcome'] for r in R); n = len(R); pc = lambda k: 100 * c.get(k, 0) / n
    print(f"{name:34s} n={n} succ {pc('success'):5.1f} coll {pc('collision'):5.1f} oob {pc('out_of_bounds'):5.1f} "
          f"UNSAFE {pc('collision')+pc('out_of_bounds'):5.1f} stall {pc('stalled'):5.1f} tmo {pc('timeout'):4.1f} "
          f"v={sum(r['mean_speed'] for r in R)/n:.2f} intv={sum(r['intervention_rate'] for r in R)/n:.2f}")
for spec in sys.argv[1:]:
    name, f = spec.rsplit('=', 1)
    R = load([f])
    if 'clamp_on' in f: R = [r for r in R if r['method'] == 'mppi_alone' and r['mu'] == 1.0 and r['v_target'] == 6.0]
    row(name, R)
