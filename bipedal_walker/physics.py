"""
physics.py - Core 2D physics simulation using pymunk.

This file builds a 7-segment ragdoll (torso + 2 thighs + 2 shins + 2 feet)
in a pymunk physics space and lets an RL agent control 6 joint motors to
make it walk.  The coordinate system matches pygame's screen coords: +x is
right, +y is DOWN, so gravity = (0, +980) pulls things downward.

Classes:
    RagdollBody  -- builds and manages the ragdoll segments & joints
    PhysicsWorld -- owns the pymunk Space, the ground, and the ragdoll
"""

import math
import numpy as np
import pymunk

from config import CONFIG

# how fast the motors can spin (rad/s).  actions in [-1, 1] get scaled
# to [-MAX_MOTOR_RATE, +MAX_MOTOR_RATE].  10 rad/s gives smooth
# proportional control -- the agent can request gentle or aggressive
# joint movement and the motor's max_force caps the actual torque.
MAX_MOTOR_RATE = 10.0

# normalize observations so all components land roughly in [-1, 1].
# velocities are ~hundreds px/s, height is ~150px, angles are ~1 rad —
# without scaling, the network struggles with the mixed magnitudes.
_VEL_SCALE = 300.0       # linear velocity normalization
_HEIGHT_SCALE = 200.0     # height normalization
_ANG_VEL_SCALE = 10.0     # angular velocity normalization


# ── ragdoll ────────────────────────────────────────────────────────────

class RagdollBody:
    """
    A 7-segment bipedal ragdoll: torso, left thigh, left shin, left foot,
    right thigh, right shin, right foot.  Six motorized joints
    (2 hips + 2 knees + 2 ankles) connect the segments.

    The observation space is 19 floats (see get_state for the layout).
    The action space is 6 floats in [-1, 1], one per motor.
    """

    # names of the 6 motors, in the order the agent's action vector uses
    MOTOR_NAMES = [
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]

    def __init__(self, space, config):
        self.space = space
        self.config = config

        # we'll store every body/shape/constraint so we can nuke them on reset
        self.bodies = {}   # name -> pymunk.Body
        self.shapes = {}   # name -> pymunk.Shape
        self.joints = []   # flat list of all constraints (pivots, limits, motors)
        self.motors = {}   # name -> pymunk.SimpleMotor (also in self.joints)

        # shapes in the same non-zero group don't collide with each other,
        # but they still collide with group-0 shapes (like the ground)
        self._group_filter = pymunk.ShapeFilter(group=1)

        self._build_ragdoll()

    # ── construction helpers ───────────────────────────────────────────

    def _make_box_segment(self, name, mass, width, height, position, friction=1.0):
        """
        Create a rectangular rigid body and add it to the space.

        `position` is the center of the rectangle in world coords.
        Returns the pymunk.Body so we can hook up joints to it.
        """
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = position

        hw, hh = width / 2, height / 2
        shape = pymunk.Poly(body, [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
        shape.friction = friction
        shape.filter = self._group_filter

        self.space.add(body, shape)
        self.bodies[name] = body
        self.shapes[name] = shape
        return body

    def _make_joint(self, name, body_a, body_b, anchor_a, anchor_b,
                    min_angle, max_angle):
        """
        Wire up a PivotJoint + RotaryLimitJoint + SimpleMotor between
        two bodies.  Anchors are in each body's local coordinates.
        """
        # pivot keeps the two anchor points glued together
        pivot = pymunk.PivotJoint(body_a, body_b, anchor_a, anchor_b)
        pivot.collide_bodies = False

        # rotation limits so the joint can't bend past a realistic range
        limit = pymunk.RotaryLimitJoint(body_a, body_b, min_angle, max_angle)

        # motor lets the agent drive the joint (starts at rate=0, i.e. off)
        motor = pymunk.SimpleMotor(body_a, body_b, 0.0)
        if "hip" in name:
            motor.max_force = self.config["hip_motor_force"]
        elif "knee" in name:
            motor.max_force = self.config["knee_motor_force"]
        else:
            motor.max_force = self.config["ankle_motor_force"]

        self.space.add(pivot, limit, motor)
        self.joints.extend([pivot, limit, motor])
        self.motors[name] = motor

    # ── build the whole ragdoll ────────────────────────────────────────

    def _build_ragdoll(self):
        c = self.config

        # ---- torso ----
        torso_x = c["torso_start_x"]
        torso_y = c["torso_start_y"]
        self._make_box_segment(
            "torso", c["torso_mass"],
            c["torso_width"], c["torso_height"],
            (torso_x, torso_y),
            friction=0.8,
        )

        # the hips sit at the bottom of the torso, slightly inset from the
        # edges so the legs aren't glued right at the corners
        hip_offset_x = c["torso_width"] * 0.25
        torso_bottom_y = torso_y + c["torso_height"] / 2

        left_hip_world = (torso_x - hip_offset_x, torso_bottom_y)
        right_hip_world = (torso_x + hip_offset_x, torso_bottom_y)

        # ---- thighs (upper legs) ----
        # each thigh hangs straight down from its hip, center at mid-length
        ul = c["upper_leg_length"]
        uw = c["upper_leg_width"]

        self._make_box_segment(
            "left_thigh", c["upper_leg_mass"], uw, ul,
            (left_hip_world[0], left_hip_world[1] + ul / 2),
        )
        self._make_box_segment(
            "right_thigh", c["upper_leg_mass"], uw, ul,
            (right_hip_world[0], right_hip_world[1] + ul / 2),
        )

        # ---- shins (lower legs) ----
        ll = c["lower_leg_length"]
        lw = c["lower_leg_width"]

        left_knee_world = (left_hip_world[0], left_hip_world[1] + ul)
        right_knee_world = (right_hip_world[0], right_hip_world[1] + ul)

        self._make_box_segment(
            "left_shin", c["lower_leg_mass"], lw, ll,
            (left_knee_world[0], left_knee_world[1] + ll / 2),
        )
        self._make_box_segment(
            "right_shin", c["lower_leg_mass"], lw, ll,
            (right_knee_world[0], right_knee_world[1] + ll / 2),
        )

        # ---- feet ----
        fw = c["foot_width"]
        fh = c["foot_height"]

        # ankle world position = bottom of shin
        left_ankle_world = (left_knee_world[0], left_knee_world[1] + ll)
        right_ankle_world = (right_knee_world[0], right_knee_world[1] + ll)

        # foot center: ankle pivot sits on the foot's top edge, aligned so
        # the shin's left side is flush with the foot's left side.
        # shin left = ankle_x - lw/2, foot left = foot_cx - fw/2
        # flush => foot_cx = ankle_x + (fw - lw) / 2
        self._make_box_segment(
            "left_foot", c["foot_mass"], fw, fh,
            (left_ankle_world[0] + (fw - lw) / 2, left_ankle_world[1] + fh / 2),
        )
        self._make_box_segment(
            "right_foot", c["foot_mass"], fw, fh,
            (right_ankle_world[0] + (fw - lw) / 2, right_ankle_world[1] + fh / 2),
        )

        # ---- joints ----
        # anchor coords are in each body's LOCAL frame
        torso_body = self.bodies["torso"]

        # hip anchors on the torso (bottom-left and bottom-right)
        left_hip_anchor_torso = (-hip_offset_x, c["torso_height"] / 2)
        right_hip_anchor_torso = (hip_offset_x, c["torso_height"] / 2)

        # thigh anchors: top of thigh is at -half_length in local y
        thigh_top = (0, -ul / 2)
        # bottom of thigh is at +half_length
        thigh_bottom = (0, ul / 2)
        # top of shin
        shin_top = (0, -ll / 2)

        # left hip
        self._make_joint(
            "left_hip", torso_body, self.bodies["left_thigh"],
            left_hip_anchor_torso, thigh_top,
            c["hip_min_angle"], c["hip_max_angle"],
        )
        # right hip
        self._make_joint(
            "right_hip", torso_body, self.bodies["right_thigh"],
            right_hip_anchor_torso, thigh_top,
            c["hip_min_angle"], c["hip_max_angle"],
        )
        # left knee
        self._make_joint(
            "left_knee", self.bodies["left_thigh"], self.bodies["left_shin"],
            thigh_bottom, shin_top,
            c["knee_min_angle"], c["knee_max_angle"],
        )
        # right knee
        self._make_joint(
            "right_knee", self.bodies["right_thigh"], self.bodies["right_shin"],
            thigh_bottom, shin_top,
            c["knee_min_angle"], c["knee_max_angle"],
        )

        # ankle joints: shin center-bottom -> foot top edge (left-aligned)
        shin_bottom = (0, ll / 2)
        # foot anchor in local coords: top edge, offset so shin left = foot left
        foot_ankle = (lw / 2 - fw / 2, -fh / 2)

        # left ankle
        self._make_joint(
            "left_ankle", self.bodies["left_shin"], self.bodies["left_foot"],
            shin_bottom, foot_ankle,
            c["ankle_min_angle"], c["ankle_max_angle"],
        )
        # right ankle
        self._make_joint(
            "right_ankle", self.bodies["right_shin"], self.bodies["right_foot"],
            shin_bottom, foot_ankle,
            c["ankle_min_angle"], c["ankle_max_angle"],
        )

    # ── public API ─────────────────────────────────────────────────────

    def get_foot_contacts(self):
        """Return dict of foot name -> bool for ground contact."""
        ground_y = self.config["ground_y"]
        margin = 8  # pixels above ground_y to count as contact
        contacts = {}
        for name in ("left_foot", "right_foot"):
            body = self.bodies[name]
            in_contact = False
            for v in self.shapes[name].get_vertices():
                if body.local_to_world(v).y >= ground_y - margin:
                    in_contact = True
                    break
            contacts[name] = in_contact
        return contacts

    def get_state(self):
        """
        Build the observation vector for the RL agent.

        Returns a numpy array of 19 floats:
            [ 0] torso x-velocity           (normalized)
            [ 1] torso y-velocity           (normalized)
            [ 2] torso angle                (rad)
            [ 3] torso angular velocity     (normalized)
            [ 4] torso height above ground  (normalized)
            [ 5] left hip  relative angle   (rad)
            [ 6] left hip  angular velocity (normalized)
            [ 7] right hip relative angle   (rad)
            [ 8] right hip angular velocity (normalized)
            [ 9] left knee  relative angle  (rad)
            [10] left knee  angular velocity(normalized)
            [11] right knee relative angle  (rad)
            [12] right knee angular velocity(normalized)
            [13] left ankle  relative angle (rad)
            [14] left ankle  angular velocity(normalized)
            [15] right ankle relative angle (rad)
            [16] right ankle angular velocity(normalized)
            [17] left foot ground contact   (0.0 or 1.0)
            [18] right foot ground contact  (0.0 or 1.0)
        """
        torso = self.bodies["torso"]

        # height above ground: in screen coords (y-down), higher y = lower,
        # so height = ground_y - torso_y
        height = self.config["ground_y"] - torso.position.y

        state = [
            torso.velocity.x / _VEL_SCALE,
            torso.velocity.y / _VEL_SCALE,
            torso.angle,
            torso.angular_velocity / _ANG_VEL_SCALE,
            height / _HEIGHT_SCALE,
        ]

        # relative angle & angular velocity for each joint
        # (relative = child_angle - parent_angle)
        for name in self.MOTOR_NAMES:
            motor = self.motors[name]
            parent, child = motor.a, motor.b
            rel_angle = child.angle - parent.angle
            rel_ang_vel = (child.angular_velocity - parent.angular_velocity) / _ANG_VEL_SCALE
            state.append(rel_angle)
            state.append(rel_ang_vel)

        # foot ground contact signals
        contacts = self.get_foot_contacts()
        state.append(1.0 if contacts["left_foot"] else 0.0)
        state.append(1.0 if contacts["right_foot"] else 0.0)

        return np.array(state, dtype=np.float32)

    def apply_actions(self, actions):
        """
        Set the 6 motor rates from the agent's action vector.

        `actions` is a length-6 array with values in [-1, 1].
        Order: left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.
        """
        for i, name in enumerate(self.MOTOR_NAMES):
            self.motors[name].rate = float(actions[i]) * MAX_MOTOR_RATE

    def is_fallen(self):
        """
        The ragdoll has fallen if any non-foot body part touches the
        ground (any vertex near or below ground_y).

        Uses a 2px margin matching the ground segment's collision radius.
        The old 5px margin caused false positives: when feet support the
        body, shin-bottom vertices sit ~10px above ground, but joint flex
        under load could push them into the 5px zone.

        Shins are checked only over their upper 80% (knee side).  The
        bottom 20% near the ankle joint is excluded so that minor clipping
        at the ankle doesn't end the episode — the intent is to penalise
        knees and torso hitting the ground, not the ankle region.
        """
        fall_y = self.config["ground_y"] - 2

        # torso and thighs: all vertices count
        for name in ("torso", "left_thigh", "right_thigh"):
            body = self.bodies[name]
            for v in self.shapes[name].get_vertices():
                if body.local_to_world(v).y >= fall_y:
                    return True

        # shins: skip the bottom 20% (local y > 0.3 * lower_leg_length,
        # i.e. the ankle-end quarter of the shin box).
        ll = self.config["lower_leg_length"]
        shin_check_limit = ll * 0.3   # == ll/2 - 0.2*ll
        for name in ("left_shin", "right_shin"):
            body = self.bodies[name]
            for v in self.shapes[name].get_vertices():
                if v.y > shin_check_limit:   # ankle-end — skip
                    continue
                if body.local_to_world(v).y >= fall_y:
                    return True

        return False

    def get_torso_x(self):
        """X-position of the torso center (for measuring forward progress)."""
        return self.bodies["torso"].position.x

    def get_torso_angle(self):
        """Torso angle in radians (0 = perfectly upright)."""
        return self.bodies["torso"].angle


# ── world ──────────────────────────────────────────────────────────────

class PhysicsWorld:
    """
    Top-level wrapper: owns the pymunk Space, the ground, and the
    ragdoll.  This is what the environment class will talk to.
    """

    def __init__(self, config=CONFIG):
        self.config = config

        # set up the pymunk space with downward gravity
        self.space = pymunk.Space()
        self.space.gravity = (0, config["gravity"])

        # flat ground stretching way out in both directions so the walker
        # never runs off the edge.  radius=2 gives it a thin but nonzero
        # collision thickness.  bounds are ~1e6 px so long visualization
        # runs (no step cap) can't walk off the edge in any practical time.
        ground = pymunk.Segment(
            self.space.static_body,
            (-1_000_000, config["ground_y"]),
            (1_000_000, config["ground_y"]),
            radius=2.0,
        )
        ground.friction = 1.0
        self.space.add(ground)
        self._ground_shape = ground

        # build the ragdoll
        self.ragdoll = RagdollBody(self.space, config)

    def step(self, actions):
        """
        Apply motor actions, advance physics by one timestep, and
        return the new observation vector.
        """
        self.ragdoll.apply_actions(actions)
        self.space.step(self.config["physics_dt"])
        return self.ragdoll.get_state()

    def reset(self):
        """
        Tear down the current ragdoll and build a fresh one at the
        starting position.  Returns the initial observation.
        """
        # remove all constraints first (they reference the bodies)
        for constraint in self.ragdoll.joints:
            self.space.remove(constraint)

        # then remove shapes and bodies
        for name in list(self.ragdoll.bodies.keys()):
            self.space.remove(self.ragdoll.shapes[name])
            self.space.remove(self.ragdoll.bodies[name])

        # brand new ragdoll
        self.ragdoll = RagdollBody(self.space, self.config)
        return self.ragdoll.get_state()

    def get_body_positions(self):
        """
        Return a dict of body-part rendering data for the renderer.

        Keys: 'torso', 'left_thigh', 'left_shin', 'left_foot', 'right_thigh', 'right_shin', 'right_foot'
        Each value is a dict with:
            'position'  -- (x, y) center of the body in world coords
            'angle'     -- rotation in radians
            'vertices'  -- list of (x, y) tuples, polygon corners in world coords
        """
        result = {}
        for name, body in self.ragdoll.bodies.items():
            shape = self.ragdoll.shapes[name]
            # transform the polygon's local vertices to world coordinates
            world_verts = [
                (body.local_to_world(v).x, body.local_to_world(v).y)
                for v in shape.get_vertices()
            ]
            result[name] = {
                "position": (body.position.x, body.position.y),
                "angle": body.angle,
                "vertices": world_verts,
            }
        return result
