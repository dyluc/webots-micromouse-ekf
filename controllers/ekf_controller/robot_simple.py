from controller import Supervisor
import numpy as np
import math

class RobotWrapper():
    # PROTO: https://github.com/cyberbotics/webots/blob/master/projects/robots/gctronic/e-puck/protos/E-puck.proto
    def __init__(self):
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("epuck")
        self.timestep = 16
        self.max_velocity = 6.28

        # fetch and initialise the default epuck sensors, ignoring F, R, R and L sensors
        self.ps = [
            self.supervisor.getDevice("ps0"),
            self.supervisor.getDevice("ps1"),
            self.supervisor.getDevice("ps2"),
            self.supervisor.getDevice("ps3"),
            self.supervisor.getDevice("ps4"),
            self.supervisor.getDevice("ps5"),
            self.supervisor.getDevice("ps6"),
            self.supervisor.getDevice("ps7")
        ]
            
        for s in self.ps:
            s.enable(self.timestep)

        # fetch and initialise the wheel motors
        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # fetch and initialise the wheel position sensors
        self.left_wheel_sensor = self.supervisor.getDevice("left wheel sensor")
        self.right_wheel_sensor = self.supervisor.getDevice("right wheel sensor")
        self.left_wheel_sensor.enable(self.timestep)
        self.right_wheel_sensor.enable(self.timestep)

        # fetch and intialise the camera for landmark detection
        self.camera = self.supervisor.getDevice("camera1")
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)

        # === robot odometry using EKF ===
        self.xhat = np.array([-1.35, -1.35, 1.57]) # current robot pose [x, y, theta]
        self.P = np.diag([1e-6, 1e-6, 1e-6]) # initial uncertainty about pose

        # process noise covariance: uncertainty in odometry motion model x, y, theta
        self.Vhat = np.diag([1e-6, 1e-6, 1e-6]) # high confidence due to odometry accuracy

        # control input placeholder: incremental motion [d_centre, d_theta]
        self.u = np.zeros(2)

        # measurement noise covariance: 2x2 block per landmark, stacked diagonally
        sigma_r = 0.03 # standard deviation of distance measurement (m) - found from calibration
        sigma_phi = np.deg2rad(2.0) # standard deviation of bearing measurement (rad) - found from calibration
        self.What = np.array([[sigma_r**2, 0],
                      [0, sigma_phi**2]])

        self.prev_l, self.prev_r = None, None
        self.wheel_radius, self.axle_length = 0.02007, 0.057125 # found from calibration

        # test landmark global positions
        self.lm = {
            "landmark1": (-1.35, -0.09),
            "landmark2": (-0.09, -1.17),
            "landmark3": (0.09, 0.27),
            "landmark4": (-0.09, 0.45),
            "landmark5": (0.99, 0.45),
            "landmark6": (0.81, -0.45),
            "landmark7": (0.09, -1.35)
        }

        # observation coordinates for each landmark, useful information for plotting
        self.lm_log = {
            "landmark1": [],
            "landmark2": [],
            "landmark3": [],
            "landmark4": [],
            "landmark5": [],
            "landmark6": [],
            "landmark7": []
        }

    def update_odometry(self):
        # get current wheel motor values
        l, r = self.left_wheel_sensor.getValue(), self.right_wheel_sensor.getValue()

        # first step
        if self.prev_l is None and self.prev_r is None:
            self.prev_l, self.prev_r = l, r

        # compute incremental distances
        dl = (l - self.prev_l) * self.wheel_radius
        dr = (r - self.prev_r) * self.wheel_radius

        # update previous motor values
        self.prev_l, self.prev_r = l, r

        # compute incremental motion
        d_centre = (dl + dr) / 2.0 # forward motion
        d_theta = (dr - dl) / self.axle_length # change in heading

        # update control inputs
        self.u[0] = d_centre
        self.u[1] = d_theta

        self._ekf_predict()

        # dead reckoning approach
        # self.theta += d_theta
        # self.x += d_centre * math.cos(self.theta)
        # self.y += d_centre * math.sin(self.theta)

    def _ekf_predict(self):
        """
        compute the predicted state using the odometry increments (dead reckoning)
        compute the jacobian of the process model
        compute the predicted covariance using the jacobian and process noise, storing both predictions (xhat and P)
        """
        d_centre, d_theta = self.u
        theta = self.xhat[2]
        xhat_pred = np.array([0.0, 0.0, 0.0])

        xhat_pred[0] = self.xhat[0] + d_centre * math.cos(theta)
        xhat_pred[1] = self.xhat[1] + d_centre * math.sin(theta)
        xhat_pred[2] = self.xhat[2] + d_theta

        # construct the process model jacobian (derivate of the process/motion model)
        F_k = np.array([[1, 0, -d_centre * math.sin(theta)],
                [0, 1,  d_centre * math.cos(theta)],
                [0, 0, 1]])
        
        # predict the P covariance
        P_pred = F_k @ self.P @ F_k.T + self.Vhat

        self.xhat = xhat_pred
        self.P = P_pred

        # clamp heading range to [-pi, pi]
        self.xhat[2] = (self.xhat[2] + np.pi) % (2 * np.pi) - np.pi

    def ekf_update(self):
        # update if observations are available

        z, lm = self._get_z() # actual measurement and landmark coords
        if z is not None: # only run the update if there are landmark observations available to correct estimates
            
            # first compute the expected location
            lmx, lmy = lm
            x_pred, y_pred, theta_pred = self.xhat

            # compute z hat
            expected_distance = math.sqrt((lmx - x_pred)**2 + (lmy - y_pred)**2)
            expected_bearing = math.atan2(lmy - y_pred, lmx - x_pred) - theta_pred
            zhat = np.array([expected_distance, expected_bearing]) # predicted measurement

            # compute innovation (difference between actual measurement and predicted measurement)
            v = z - zhat
            v[1] = (v[1] + np.pi) % (2 * np.pi) - np.pi # clamp heading range to [-pi, pi]

            # compute measurement jacobian (similar to process model jacobian in predict step)
            dx = lmx - x_pred
            dy = lmy - y_pred
            q = dx**2 + dy**2
            H_k = np.array([
                [-dx / math.sqrt(q), -dy / math.sqrt(q), 0],
                [dy / q, -dx / q, -1]
            ])

            # compute the innovation covariance and Kalman gain
            S = H_k @ self.P @ H_k.T + self.What
            K = self.P @ H_k.T @ np.linalg.inv(S)

            # update the pose using the Kalman gain and computed innovation
            self.xhat = self.xhat + K @ v

            # update the P covariance too
            self.P = (np.eye(3) - K @ H_k) @ self.P
            # I = np.eye(3) # identity matrix
            # self.P = (I - K @ H_k) @ self.P @ (I - K @ H_k).T + K @ self.What @ K.T

            # clamp heading range to [-pi, pi]
            self.xhat[2] = (self.xhat[2] + np.pi) % (2 * np.pi) - np.pi

    def set_wheel_velocity(self, lmv, rmv):
        self.left_motor.setVelocity(lmv)
        self.right_motor.setVelocity(rmv)
    
    def get_prox_sensors(self):
        return [s.getValue() for s in self.ps]
    
    def _get_z(self): # measurement vector: [distance, bearing]
        """
        Findings from camera module recognition calibration:

        Avoid integrating measurements when object distance is < ~0.4m, otherwise we risk distorting the 
        odometry with noisy readings. Above ~0.4m with landmark sizes 0.5x0.5x0.5, field of view 90 degrees
        keeps the estimates stable and variance low.

        The distance bias is roughly ~0.027 m

        Distance offset: -0.027 m, variance: 0.000009 m^2
        Bearing offset:  -0.01 deg, variance: 0.04 deg^2
        """
        calibrated_distance_offset = 0.027
        discard_distance_threshold = 0.4

        recognition_objects = self.camera.getRecognitionObjects()
        if recognition_objects: 
            obj = recognition_objects[0] # take first recognised object in field of view
            pos, name = obj.getPosition(), obj.getModel()

            # ignore z axis, we can treat distance calculation as a planar measurement
            distance = ((pos[0]**2 + pos[1]**2)**0.5) + calibrated_distance_offset

            # discard this measurement if too close to the landmark (avoid bearing errors)
            if distance < discard_distance_threshold:
                return None, None

            # compute bearing to landmark
            bearing = math.atan2(pos[1], pos[0])

            # get the associated landmark coordinates
            lm = self.lm[name]

            # track coordinates where landmarks observations are in used to correct trajectory
            self.lm_log[name].append(self.xhat) # record pose

            print(f"z: approximate corrected distance, bearing: {distance}, {bearing} to {name}")

            return np.array([distance, bearing]), lm
        return None, None
    