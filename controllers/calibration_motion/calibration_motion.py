# OUTPUT:
# Running 12 × -90° (encoder-driven, no yaw feedback)...
# Using b = 0.057125 m, r = 0.02007 m, speed = 5.0%
# Turn 01: LΔ=+2.2420 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4776 rad
# Turn 02: LΔ=+2.2420 rad, RΔ=-2.2420 rad, (R-L)Δ=-4.4839 rad
# Turn 03: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Turn 04: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Turn 05: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Turn 06: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Turn 07: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Turn 08: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Turn 09: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Turn 10: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Turn 11: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Turn 12: LΔ=+2.2357 rad, RΔ=-2.2357 rad, (R-L)Δ=-4.4714 rad
# Summary:
#   Target total:     -1080.00°
#   Final drift:      -0.12°  (~-0.010°/turn)
#   True total:       -1080.12°
#   Σ (R-L) encoders: -53.6752 rad
#   b_used (command): 0.057125 m
#   b_true (diag):    0.057144 m


from controller import Supervisor
import math
import numpy as np

class EncoderDrivenTurns:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        # Robot parameters (edit as needed)

        # 20 turns:

        # pre-calibrated (baseline - pulled from the PROTO file) Final drift:      +60.72°  (~+3.036°/turn)
        # self.r = 0.02               # wheel radius (m)
        # self.b_used = 0.055         # wheelbase *to use for commanding* (m) — your calibrated value
        # self.max_wheel_speed = 6.28    # rad/s (e-puck)
        # self.turn_speed_frac = 0.05    # constant wheel speed fraction (no PID)
        # self.turn_angle_deg = -90      # per turn (clockwise = negative)

        # post-calibration Final drift:      +0.07°  (~+0.003°/turn)
        self.r = 0.02007              # wheel radius (m)
        self.b_used = 0.057125         # wheelbase *to use for commanding* (m) — your calibrated value
        self.max_wheel_speed = 6.28    # rad/s (e-puck)
        self.turn_speed_frac = 0.05    # constant wheel speed fraction (no PID)
        self.turn_angle_deg = -90      # per turn (clockwise = negative)

        # Devices
        self.robot_node = self.supervisor.getFromDef("epuck")
        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.left_enc = self.supervisor.getDevice("left wheel sensor")
        self.right_enc = self.supervisor.getDevice("right wheel sensor")
        self.left_enc.enable(self.timestep)
        self.right_enc.enable(self.timestep)

        # prime encoders
        self.supervisor.step(self.timestep)

    # ---------- helpers ----------
    def get_yaw_deg(self):
        rot = self.robot_node.getField("rotation").getSFRotation()
        yaw = rot[3] if rot[2] >= 0 else -rot[3]
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        return math.degrees(yaw)

    @staticmethod
    def wrap_deg(d):
        while d > 180: d -= 360
        while d <= -180: d += 360
        return d

    # ---------- core ----------
    def _turn_once_by_encoder(self, angle_deg: float, speed_frac: float, stop_tol_rad: float = 0.0):
        """
        Execute one pure rotation using encoder deltas only.
        - angle_deg: desired body rotation (+CCW, -CW)
        - speed_frac: constant wheel speed as fraction of max
        - stop_tol_rad: encoder rad tolerance before stopping (0 => stop as soon as both reach/exceed target)
        """
        # target per-wheel rotation magnitude for pure spin:
        # |Δφ_wheel| = (b * |θ|) / (2 r)
        theta = abs(math.radians(angle_deg))
        dphi_target = (self.b_used * theta) / (2.0 * self.r)  # rad

        # wheel directions
        w = speed_frac * self.max_wheel_speed
        if angle_deg < 0:   # clockwise body rotation: left forward, right backward
            l_cmd, r_cmd = +w, -w
            l_sign, r_sign = +1.0, -1.0
        else:               # counter-clockwise body rotation
            l_cmd, r_cmd = -w, +w
            l_sign, r_sign = -1.0, +1.0

        # starting encoder values
        L0 = self.left_enc.getValue()
        R0 = self.right_enc.getValue()

        # command constant speed until both wheels have reached their target delta
        self.left_motor.setVelocity(l_cmd)
        self.right_motor.setVelocity(r_cmd)

        # loop until both wheels reach/exceed target minus tolerance
        while True:
            if self.supervisor.step(self.timestep) == -1:
                break
            L = self.left_enc.getValue() - L0
            R = self.right_enc.getValue() - R0
            # signed progress along commanded directions
            L_prog = l_sign * L
            R_prog = r_sign * R
            # stop if both wheels reached target (within optional tolerance)
            if (L_prog >= max(0.0, dphi_target - stop_tol_rad)) and (R_prog >= max(0.0, dphi_target - stop_tol_rad)):
                break

        # stop wheels
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        # settle a few ticks
        for _ in range(5):
            self.supervisor.step(self.timestep)

        # return actual signed deltas for reporting
        Ld = self.left_enc.getValue() - L0
        Rd = self.right_enc.getValue() - R0
        return Ld, Rd

    def run_sequence(self, num_turns: int = 48, stop_tol_rad: float = 0.0, npz_file="turn_errors.npz"):
        """
        Run N x turn_angle_deg using encoder-driven stopping.
        Records yaw after each turn and computes per-turn drift.
        Saves results to an NPZ for plotting and prints the summary.
        """
        print(f"Running {num_turns} × {self.turn_angle_deg}° (encoder-driven, no yaw feedback)...")
        print(f"Using b = {self.b_used:.6f} m, r = {self.r:.5f} m, speed = {self.turn_speed_frac*100:.1f}%")

        start_yaw = self.get_yaw_deg()
        D_total = 0.0
        yaw_per_turn = []
        error_per_turn = []

        for i in range(1, num_turns + 1):
            Ld, Rd = self._turn_once_by_encoder(self.turn_angle_deg, self.turn_speed_frac, stop_tol_rad=stop_tol_rad)
            D_total += (Rd - Ld)
            current_yaw = self.get_yaw_deg()
            yaw_per_turn.append(current_yaw)
            # error relative to expected cumulative angle
            expected_yaw = start_yaw + i * self.turn_angle_deg
            drift = self.wrap_deg(current_yaw - expected_yaw)
            error_per_turn.append(drift)
            print(f"Turn {i:02d}: LΔ={Ld:+.4f} rad, RΔ={Rd:+.4f} rad, (R-L)Δ={Rd-Ld:+.4f} rad, yaw={current_yaw:.2f}°, error={drift:.2f}°")

        # total drift / final values
        end_yaw = self.get_yaw_deg()
        e_deg = self.wrap_deg(end_yaw - start_yaw)
        theta_target_deg = num_turns * self.turn_angle_deg
        theta_true_deg = theta_target_deg + e_deg
        theta_true_rad = math.radians(theta_true_deg)
        b_true_diag = (self.r * D_total) / (theta_true_rad) if abs(theta_true_rad) > 1e-9 else float("nan")

        # save npz file
        np.savez(npz_file,
                yaw=np.array(yaw_per_turn),
                error=np.array(error_per_turn),
                turn_angle=self.turn_angle_deg)
        print(f"\n✓ Saved per-turn rotation errors to {npz_file}")
        
        print("\nSummary:")
        print(f"  Target total:     {theta_target_deg:+.2f}°")
        print(f"  Final drift:      {e_deg:+.2f}°  (~{e_deg/num_turns:+.3f}°/turn)")
        print(f"  True total:       {theta_true_deg:+.2f}°")
        print(f"  Σ (R-L) encoders: {D_total:+.4f} rad")
        print(f"  b_used (command): {self.b_used:.6f} m")
        print(f"  b_true (diag):    {b_true_diag:.6f} m")

        return yaw_per_turn, error_per_turn

def main():
    tester = EncoderDrivenTurns()
    # Choose how many turns you want (multiples of 4 recommended: 12, 24, 48, ...)
    # stop_tol_rad=0.0 means stop exactly when both encoders reach or exceed the target (no creep/guardrails)
    tester.run_sequence(num_turns=20, stop_tol_rad=0.0)

if __name__ == "__main__":
    main()