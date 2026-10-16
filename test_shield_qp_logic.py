"""
Logic tests for NeuralICBFController.filter_control (QP assembly, slack, wall fallback, metrics).

These tests do NOT need torch / qpsolvers: heavy modules are stubbed and the QP back-end is replaced by a
scipy solver. They check the *logic* of filter_control. They do NOT validate the autograd Jacobians
(compute_*_gradient) -- run the finite-difference test on a machine with torch for that.
"""
import sys, os, types
import numpy as np
from types import SimpleNamespace
from scipy.optimize import linprog, minimize

# ---- stub heavy imports -----------------------------------------------------------------------
for name in ["torch", "torch.nn", "qpsolvers", "dynamic_bicycle_env", "train_neural_icbf", "mppi_planner"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.nn"].Module = object
sys.modules["qpsolvers"].solve_qp = lambda *a, **k: None
for cls in ["DynamicBicycleModel", "DynamicBicycleParams", "ClutteredRacingEnv"]:
    setattr(sys.modules["dynamic_bicycle_env"], cls, object)
sys.modules["train_neural_icbf"].NeuralBarrierNet = object
sys.modules["mppi_planner"].MPPIPlanner = object

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import safe_controller as sc


def scipy_qp(P_diag, q, G, h, lb, ub):
    n = len(P_diag)
    bounds = list(zip(lb, ub))
    res = linprog(np.zeros(n), A_ub=G, b_ub=h, bounds=bounds, method="highs")
    if res.status != 0:
        return None, "infeasible"
    r = minimize(lambda z: 0.5 * z @ (P_diag * z) + q @ z, res.x, jac=lambda z: P_diag * z + q,
                 constraints=[{"type": "ineq", "fun": lambda z: h - G @ z, "jac": lambda z: -G}],
                 bounds=bounds, method="SLSQP", options={"ftol": 1e-12, "maxiter": 200})
    return r.x, "ok"


def make_shield(obs_fn, wall_fn):
    s = object.__new__(sc.NeuralICBFController)
    s.p = SimpleNamespace(a_max=9.51, delta_max=0.4189, sv_max=3.2)
    s.mode, s.r_margin, s.alpha_w, s.wall_limit = "neural_icbf", 0.26, 2.5, 1.75
    s.w_slack_lin = 100.0
    s.w_slack, s.grad_floor, s.brake_decel, s.tol_ax, s.tol_delta = 1000.0, 1e-3, 4.0, 0.05, 0.005
    s.solver = "stub"
    s.compute_neural_icbf_gradient = obs_fn
    s.compute_wall_hocbf_gradient = wall_fn
    s._solve_qp = lambda P, q, G, h, lb, ub: scipy_qp(P, q, G, h, lb, ub)
    return s

STATE = np.array([10.0, 0.0, 0.0, 6.0, 0.0, 0.0], dtype=np.float32)   # px,py,psi,vx,vy,r
OBS_AHEAD = np.array([[12.0, 0.0, 0.3]], dtype=np.float32)               # inside hazard window
WALL_OK = lambda st, u: (0.0, 0.0, np.array([0.0, 2.0]))                 # py_dot, py_ddot, p_wall (far from walls)


def test_all_satisfied_returns_candidate_exactly():
    s = make_shield(lambda st, u, o: (0.8, 1.0, np.array([0.5, 0.5])), WALL_OK)
    u, info = s.filter_control(STATE, np.array([0.0, 0.0]), np.array([3.0, 0.01]), OBS_AHEAD)
    assert info["qp_status"] == "skipped_all_satisfied" and not info["qp_active"]
    assert np.allclose(u, [3.0, 0.01])


def test_slew_clip_alone_is_not_an_intervention():
    s = make_shield(lambda st, u, o: (0.8, 1.0, np.array([0.5, 0.5])), WALL_OK)
    u, info = s.filter_control(STATE, np.array([0.0, 0.0]), np.array([3.0, 0.40]), OBS_AHEAD)
    assert not info["qp_active"]                       # steering was only slew-clipped
    assert abs(info["dev_from_nom"][1]) > 0.3          # but the deviation from u_nom is reported
    assert abs(u[1] - 3.2 * 0.02) < 1e-6


def test_violated_row_is_satisfied_with_zero_slack_when_reachable():
    # needs n^T du >= 0.5 ; n=(1,0) -> raise ax by 0.5 (reachable)
    s = make_shield(lambda st, u, o: (0.1, -0.5, np.array([1.0, 0.0])), WALL_OK)
    u, info = s.filter_control(STATE, np.array([0.0, 0.0]), np.array([3.0, 0.0]), OBS_AHEAD)
    assert info["qp_status"] == "ok" and info["slack_max"] < 1e-6
    assert u[0] >= 3.5 - 1e-4 and info["qp_active"]


def test_tiny_gradient_triggers_brake_not_noise_steering():
    s = make_shield(lambda st, u, o: (-0.2, -0.5, np.array([1e-6, 1e-6])), WALL_OK)
    u, info = s.filter_control(STATE, np.array([0.0, 0.0]), np.array([3.0, 0.0]), OBS_AHEAD)
    assert info["n_no_authority"] == 1 and info["brake_override"]
    assert u[0] <= -4.0 + 1e-4                         # braking override
    assert abs(u[1]) < 1e-6                            # no noise-driven steering


def test_unreachable_row_is_flagged_and_brakes():
    s = make_shield(lambda st, u, o: (-0.2, -50.0, np.array([0.0, 1.0])), WALL_OK)   # needs 50 rad of steering
    u, info = s.filter_control(STATE, np.array([0.0, 0.0]), np.array([3.0, 0.0]), OBS_AHEAD)
    assert info["n_unreachable"] == 1 and info["brake_override"] and info["slack_max"] > 1.0
    assert u[0] <= -4.0 + 1e-4


def test_satisfied_rows_are_kept_so_swerve_cannot_break_them():
    # A (violated): n=(0.6,0.8), needs 0.6*dax + 0.8*ddelta >= 0.1 (min-norm would use ddelta=0.08, but the
    # servo window is +-0.064). B (satisfied at u_cand): margin 0.3 - 10*ddelta >= 0  =>  ddelta <= 0.03.
    # If B were dropped (old behaviour) the QP would steer ~0.064 rad into B's violation.
    calls = iter([(0.1, -0.5, np.array([3.0, 4.0])), (0.3, 0.30, np.array([0.0, -10.0]))])
    s = make_shield(lambda st, u, o: next(calls), WALL_OK)
    obs = np.array([[12.0, 0.0, 0.3], [12.5, 0.2, 0.3]], dtype=np.float32)
    u, info = s.filter_control(STATE, np.array([0.0, 0.0]), np.array([3.0, 0.0]), obs)
    du = info["v_star"]
    assert info["n_obstacle_rows"] == 2 and info["qp_status"] == "ok"
    assert du[1] <= 0.03 + 1e-5                        # B respected
    assert 0.6 * du[0] + 0.8 * du[1] >= 0.1 - 1e-5     # A satisfied (throttle makes up the rest)
    assert info["slack_max"] < 1e-5


def test_wall_rows_are_hard_and_respected():
    # top-wall margin violated; needs p_wall . du <= margin with p_wall=(0, 2) -> d_delta <= -0.01 (reachable)
    s = make_shield(lambda st, u, o: (0.8, 1.0, np.array([0.5, 0.5])),
                    lambda st, u: (0.0, 0.0, np.array([0.0, 2.0])))
    st = STATE.copy(); st[1] = 1.76                     # slightly beyond wall_limit -> top margin < 0
    u, info = s.filter_control(st, np.array([0.0, 0.05]), np.array([3.0, 0.05]), OBS_AHEAD)
    assert info["qp_status"] == "ok" and not info["wall_fallback"]
    assert info["wall_margin_min"] < 0 and u[1] < 0.05


def test_wall_infeasible_uses_fallback_toward_center():
    # top margin hugely negative, tiny steering authority -> infeasible -> fallback
    s = make_shield(lambda st, u, o: (0.8, 1.0, np.array([0.5, 0.5])),
                    lambda st, u: (3.0, 5.0, np.array([0.0, 0.5])))
    st = STATE.copy(); st[1] = 1.8
    u, info = s.filter_control(st, np.array([0.0, 0.10]), np.array([3.0, 0.10]), np.empty((0, 3), np.float32))
    assert info["qp_status"] == "infeasible" and info["wall_fallback"] and info["qp_active"]
    assert u[0] <= -4.0 + 1e-4                          # braking
    assert u[1] < 0.10                                  # steered away from the +y wall (d py_ddot/d delta > 0)
    assert abs(u[1] - (0.10 - 3.2 * 0.02)) < 1e-6       # at the slew limit


def test_no_input_ellipse_throttle_cut_on_open_track():
    s = make_shield(lambda st, u, o: (0.8, 1.0, np.array([0.5, 0.5])), WALL_OK)
    u, info = s.filter_control(STATE, np.array([9.0, 0.40]), np.array([9.5, 0.4189]), np.empty((0, 3), np.float32))
    assert u[0] == np.float32(9.5)                      # full throttle + full lock untouched


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} LOGIC TESTS PASSED SUCCESSFULLY!")
