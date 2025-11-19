#import "@preview/diatypst:0.8.0": *
#show: slides.with(
  title: "Under-Actuated Manipulator Control (is hard)",
  subtitle: "& other revelations in manipulator kinematics",
  date: "20.11.2025",
  authors: ("Vignesh Vembar", "Krish Pandya"),
  // footer-title: "",
  // footer-subtitle: "",
  layout: "large",
  count: "number"
)

= Representing the Arm

== DH Parameters

Denavit Hartenberg (DH) parameters describe each joint using four values $a_i, alpha_i, d_i, theta_i$ and produce forward kinematics through a fixed transform pattern. They were popular because they give a compact, minimal representation.

#align(center)[
#image("images/dh_parameters.png", width: 40%)
] 

== Issues with DH Parameters

- Frame assignment rules are unintuitive and heavily constrained.
- Small mistakes in axis alignment break the model.
- Hard to insert tool frames, offsets, or calibration transforms.

#link("https://www.researchgate.net/publication/382909728_From_Unified_Robot_Description_Format_to_DH_Parameters_Examinations_of_Two_Different_Approaches_for_Manipulator")[Some people]
 have tried to bridge the gap between modern methods like URDF and DH, but for this project we used a different approach.

== ETS 

- Elementary Transform Sequence (ETS) represents each joint as a separate transform.
- More intuitive frame assignment.
- Simply compose transforms to get forward kinematics.

$
  ""^0 T_e = product_"i=1"^n E_i(eta_i)
$

#quote(attribution: [arxiv.org/abs/2207.01796])[
  This tutorial [...] deliberately avoids the commonly-used Denavit-Hartenberg parameters which we feel confound pedagogy.
]

= Metrics to Visualize Reachability

== Manipulability

$
  omega = sqrt(det(J * J^T))
$

But for our 5-DOF arm, `J.shape() == (6, 5)`, so $J * J^T$ is not full rank and $det(J * J^T) = 0$ always.

So instead, we use the singular values of `J` to define manipulability as:

$
  omega = sqrt(product_"i=1"^6 sigma_i)
$

== Condition

Another useful metric is the condition number of the Jacobian, defined as the ratio of the maximum and minimum singular values:

$
  kappa = sigma_"max" / sigma_"min"
$

$kappa >> 1$ implies the arm is close to a singularity, while $kappa approx 1$ implies good dexterity.

== Visualizing the Reachable Workspace

#align(center)[
  #image("images/reachable_workspace.png")
  #image("images/manipulability_scatter.png", width: 70%)
]

= The Problem and its Naive Solution

== 5-DOF, What are we missing?

#align(center)[
  #image("images/so101.png", width: 60%)
]

There is no end-effector yaw.

== Feasibility Projected IK

- 5-DOF arm can only move on a 5D “surface” inside the 6D space
- Project desired pose to closest feasible pose on that surface

$
  q^* &= arg min_q ||W_x (f(q)-x_d)||^2 \
  Delta x &approx J Delta q \
  Delta x_"achievable" &= J J^+ Delta x
$

From this we solve for:

$
  
$

== Where it Fails

- Because the yaw axis is missing, when the end effector moves off the 2D plane defined by the first joint axis, the IK solver continuously fails to find a solution.
- It barely leaves the 2D plane in practice.
- The "projected" solution is always just on (or very close to) the 2D plane.

= The Workaround that Works

== Constraining the IK
- Instead, we constrain the IK solver.
- Never solve for the first joint angle, so the solver thinks the motion is on a 2D plane.
- Use a geometric IK to get back full position control.
- End effector yaw is still not controllable :( but at least the solver doesnt continuously fail when we leave the 2D plane.

== Fool the IK solver

#align(center)[
  #image("images/so101_side.png", width: 70%)
]

The solver never gets end effector y-axis position or yaw updates, and never updates the first joint angle.

This way the solver doesnt continuously fail when we leave the 2D plane.

== Limitations

- Only works for position control, not full 6D pose control.
- End effector yaw is completely uncontrollable, and pitch and roll are constrained to the rotating 2D plane.


= Demo