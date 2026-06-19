# The Optimal Racing Line — minimum-lap-time optimal control

I don't drive the car. I give a solver a track and a tyre-grip limit, and it
computes the fastest way round — the racing line is the *output* of an
optimal-control problem, not something drawn by hand.

![the optimal racing line](output/the_lap.png)

## What it is

Minimum-lap-time is a **free-final-time optimal-control problem**. The standard
trick removes the free final time: use **arc length `s` along the track
centreline as the independent variable**, so the finish is a fixed boundary
(`s = L`) and lap time becomes the integral to minimise. The car (a point mass on
a friction circle) has states `n` (lateral offset), `ξ` (heading vs the track
tangent) and `v` (speed); controls are longitudinal and lateral acceleration
`(a_t, a_n)` limited by the grip ellipse `a_t² + a_n² ≤ (μg)²`.

The continuous problem is transcribed to a **nonlinear program by direct
collocation** (trapezoidal defects between nodes), with **hand-derived analytic
Jacobians** handed to the solver (scipy SLSQP). The whole point of optimal
control on a circuit — that the line through one corner depends on the next — is
solved in one shot, closed-loop (periodic boundary conditions for the lap).

The racing line **emerges**: it brakes in a straight line, uses the full track
width to straighten corners (a larger effective radius → higher apex speed),
trail-brakes into the apex, and gets back to power on exit — none of it imposed.

## Headline result

On a stylised 2.4 km **Grand Circuit** (μ = 1.5):

- **Racing line: 38.76 s** — vs **42.19 s** constrained to the geometric centreline
  → the optimal line is **3.43 s (8.1 %) faster**, the entire value of a racing line.
- The car is at the **grip limit for ~100 % of the lap** (the g-g diagram shows
  every point on the friction circle — trail-braking, apex, power-down).
- `max defect ≈ 5·10⁻¹¹` — the trajectory satisfies the dynamics to machine
  precision.

## Real car physics (optional, on top of the same core)

Switch on **engine power limit** (`a_t ≤ P/(mv)`), **aerodynamic drag**
(`a_drag = ½ρC_dA v²/m`) and **downforce** — and the grip limit grows with speed,
`a_grip(v) = μ(g + ½ρC_lA v²/m)`. The g-g "circle" becomes a **speed-dependent
performance envelope**: high-speed corners pull more g than slow ones, the
defining feature of a real race car. Each term is one line in the dynamics plus
its analytic Jacobian row (`output/aero_envelope.png`).

## Verification

| Check | Result | Reference |
|---|---|---|
| Analytic Jacobians vs finite difference | < 1.2·10⁻⁶ | exact derivatives |
| Forward car: steady corner radius | exact | `v² = μgR` |
| Forward car: RK4 order | slope 4.04 | 4th-order |
| **Skidpad lap time (closed form)** | matches | `T = 2π√(R/μg)` |
| **Mesh convergence** of lap time | converges | trapezoidal `~ds²` |
| **Friction-circle saturation** | ~100 % at limit | optimal control rides the limit |

Tight defects alone only show the NLP satisfied its own constraints, so the lap
time is also checked against a closed-form optimum (the skidpad) and shown to
converge under mesh refinement.

## Lap-time sensitivity (the optimiser as a design tool)

Re-solving the lap while sweeping parameters turns the model into a trade study
(`output/sensitivity.png`): **lap time vs grip** (wet → slicks), **vs power**
(diminishing returns), **vs downforce** (what it buys). Warm-started across
parameters, so the whole study is cheap. This is what dynamic optimisation is
*for* — answering "how much does +X buy?", not drawing a line.

## Run it

```bash
python solver/verify_car.py     # M1: forward car + RK4 (energy, corner, order)
python solver/verify_jac.py     # analytic Jacobians vs finite difference
python solver/run_lap.py        # THE LAP: racing line + g-g + centreline gain + web
python solver/verify_lap.py     # skidpad closed-form + mesh convergence + saturation
python solver/sweeps.py         # aero envelope + lap-time sensitivity sweeps
```
Then **double-click `web/index.html`** — the car runs the optimal lap in real
lap-time, speed-coloured, with a live g-g dot (data injected, no server).

## Method notes and limitations

Point-mass on a friction circle is a deliberate baseline — every term is
hand-checkable. It has no load transfer or yaw dynamics; the single-track
(bicycle) model with a combined-slip tyre is the natural next step (and would
warm-start from this solution). SLSQP is a dense SQP; for much finer meshes an
interior-point solver (IPOPT) exploiting sparsity is the right tool — the
transcription and analytic Jacobians here are exactly what such a solver needs.
Variable scaling + a quasi-steady-state warm start + mesh continuation are what
make the closed-circuit NLP converge cleanly.

---

*Method: free-final-time OCP → arc-length reformulation → direct collocation →
NLP with analytic Jacobians. Verified against `v=√(μgR)`, the skidpad closed-form
lap time `2π√(R/μg)`, mesh convergence, and finite-difference Jacobian checks.*
