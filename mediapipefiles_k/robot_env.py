import numpy as np
import gymnasium as gym
from gymnasium import spaces
import socket
import json

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Empty
    ROS2_AVAILABLE = True
except ImportError:
    print("Warning: rclpy not found. Setting up blazing-fast Windows-to-WSL TCP Bridge for Gazebo!")
    ROS2_AVAILABLE = False

class Op3GymEnv(gym.Env):
    """
    Custom Environment that follows the Gymnasium interface.
    This wraps the ROBOTIS OP3 in Gazebo via ROS2.
    It expects a 23-dimensional action vector [-1, 1] representing the joint angles.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(Op3GymEnv, self).__init__()
        
        # 23 joint angles mapping to the humanoid servos.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float32)
        self.joint_gains = np.array([
            1.8, 1.8, 1.7, 1.7,
            1.5, 1.5, 1.3, 1.3,
            1.4, 1.4, 1.4, 1.4,
            1.3, 1.3, 1.2, 1.2,
            1.2, 1.2, 1.0, 1.0,
            1.0, 1.0, 1.0
        ], dtype=np.float32)
        self.prev_action_rads = np.zeros(23, dtype=np.float32)
        
        # 132 landmark coordinates representing the imitation target state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(132,), dtype=np.float32)
        
        self.current_state = np.zeros(132, dtype=np.float32)
        
        # Map indices to literal OP3 (or Softbot) joint names in Gazebo URDF
        self.joint_names = [
            'r_sho_pitch', 'l_sho_pitch', 'r_sho_roll', 'l_sho_roll',
            'r_el', 'l_el', 'r_hip_yaw', 'l_hip_yaw',
            'r_hip_roll', 'l_hip_roll', 'r_hip_pitch', 'l_hip_pitch',
            'r_knee', 'l_knee', 'r_ank_pitch', 'l_ank_pitch',
            'r_ank_roll', 'l_ank_roll', 'head_pan', 'head_tilt',
            'j_20', 'j_21', 'j_22' 
        ]
        
        if ROS2_AVAILABLE:
            rclpy.init()
            self.node = rclpy.create_node('op3_gail_env_node')
            self.publisher_ = self.node.create_publisher(JointState, '/joint_states', 10)
            self.subscriber_ = self.node.create_subscription(JointState, '/joint_states', self._joint_callback, 10)
            self.reset_client = self.node.create_client(Empty, '/reset_world')
        else:
            # Set up the incredibly fast, low-latency Socket Bridge to talk directly to your WSL partition!
            # We use TCP because WSL2 explicitly allows TCP Localhost forwarding from Windows natively.
            self.tcp_ip = "127.0.0.1" 
            self.tcp_port = 5005
            self.connected = False

    def _joint_callback(self, msg):
        pass

    def step(self, action):
        # Calculate radians (Assume ~180 deg max rotation for scaling)
        raw_action_rads = np.clip(action * np.pi * self.joint_gains, -np.pi, np.pi).astype(np.float32)
        action_rads = np.clip((0.65 * self.prev_action_rads) + (0.35 * raw_action_rads), -np.pi, np.pi)
        self.prev_action_rads = action_rads
        action_rads = action_rads.tolist()
        
        # --- 1. Publish Action to Gazebo ---
        if ROS2_AVAILABLE:
            msg = JointState()
            msg.name = self.joint_names
            msg.position = action_rads
            self.publisher_.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.01)
        else:
            # Blast the actions directly payload to WSL2 in micro-seconds!
            try:
                if not self.connected:
                    if not getattr(self, 'connection_attempted', False):
                        print(f"🔄 Attempting TCP connection to WSL Bridge at {self.tcp_ip}:{self.tcp_port}...")
                        self.connection_attempted = True
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.settimeout(1.0) # wait up to 1 second for first connection
                    self.sock.connect((self.tcp_ip, self.tcp_port))
                    self.sock.settimeout(None) # then return to blocking for quick writes
                    self.connected = True
                    print(f"✅ Robot Env connected to WSL Bridge at {self.tcp_ip}:{self.tcp_port} (TCP)")
                
                payload = json.dumps({
                    "names": self.joint_names,
                    "positions": action_rads
                }) + "\n"
                
                #Debug: Print first 2 joints to verify they aren't zero
                print(f"Sent {len(action_rads)} joints: max_abs={np.max(np.abs(action_rads)):.2f}, Head_Pan={action_rads[18]:.2f}, R_Sho_Pitch={action_rads[0]:.2f}")
                
                self.sock.sendall(payload.encode('utf-8'))
            except Exception as e:
                if self.connected:
                    print(f"⚠️ Robot Env TCP connection dropped: {e}")
                elif not getattr(self, 'connection_failed_printed', False):
                    print(f"❌ Failed to connect to WSL TCP Bridge at {self.tcp_ip}:{self.tcp_port} -> {e}")
                    self.connection_failed_printed = True
                self.connected = False
                if getattr(self, 'sock', None):
                    self.sock.close()
                    self.sock = None
            
        # --- 2. Get Next State ---
        # Mocking Forward Kinematics resolution of OP3 joints to MediaPipe 132-dim Landmarks
        next_state = self.current_state + np.random.normal(0, 0.01, size=(132,)).astype(np.float32)
        self.current_state = next_state
        
        # --- 3. Compute Reward ---
        # In GAIL, the reward is internally computed by the Discriminator network.
        # We return 0 here, because Stable Baselines3 will override it using the imitation reward.
        reward = 0.0
        
        # --- 4. Done conditions ---
        done = False
        truncated = False
        info = {}
        
        return self.current_state, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if ROS2_AVAILABLE:
            if self.reset_client.wait_for_service(timeout_sec=1.0):
                req = Empty.Request()
                self.reset_client.call_async(req)
                rclpy.spin_once(self.node, timeout_sec=0.1)
            else:
                self.node.get_logger().warn('Reset service /reset_world not available.')

        # Reset state back to T-pose/Base layout
        self.current_state = np.zeros(132, dtype=np.float32)
        return self.current_state, {}

    def close(self):
        if ROS2_AVAILABLE:
            self.node.destroy_node()
            rclpy.shutdown()

# Quick test if run directly
if __name__ == '__main__':
    env = Op3GymEnv()
    print(f"✅ Created Environment.")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    obs, info = env.reset()
    action = env.action_space.sample()  # Random 23-dim vector
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"Successfully ran step. Output obs shape: {next_obs.shape}")