import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import socket
import json

class WslBridgeNode(Node):
    def __init__(self):
        super().__init__('windows_wsl_bridge')
        
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        
        # We listen on 0.0.0.0 so we can capture TCP packets forwarded by WSL2 local bridge
        self.tcp_ip = "0.0.0.0"
        self.tcp_port = 5005
        
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.tcp_ip, self.tcp_port))
        self.server_sock.listen(1)
        self.server_sock.setblocking(False)
        
        self.client_sock = None
        self.buffer = ""
        
        # High frequency timer exactly parallel to the 20hz cap we made in live_demo.py
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.get_logger().info("✅ Bridge Node established! Listening for Windows ML joint commands on TCP 5005...")

    def timer_callback(self):
        if self.client_sock is None:
            try:
                self.client_sock, addr = self.server_sock.accept()
                self.client_sock.setblocking(False)
                self.get_logger().info(f"✅ Connected to ML Pipeline at {addr}!")
            except BlockingIOError:
                pass
            return
            
        try:
            data = self.client_sock.recv(4096)
            if not data:
                self.get_logger().warn("⚠️ Client disconnected.")
                self.client_sock.close()
                self.client_sock = None
                return
                
            self.buffer += data.decode('utf-8')
            
            # Flush out the stream based on JSON newline endings
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                if line.strip():
                    msg_dict = json.loads(line)
                    
                    joint_msg = JointState()
                    joint_msg.header.stamp = self.get_clock().now().to_msg()
                    joint_msg.name = msg_dict['names']
                    joint_msg.position = msg_dict['positions']
                    # ROS2 requires velocity and effort arrays to be same length as position
                    joint_msg.velocity = [0.0] * len(joint_msg.position)
                    joint_msg.effort = [0.0] * len(joint_msg.position)
                    
                    self.publisher_.publish(joint_msg)
                    self.get_logger().info(f"✅ Published {len(joint_msg.name)} joints | Head_Pan={joint_msg.position[18]:.2f}, R_Sho={joint_msg.position[0]:.2f}")
                    
        except BlockingIOError:
            pass 
        except Exception as e:
            self.get_logger().error(f"Error parsing packet: {e}")
            if self.client_sock:
                self.client_sock.close()
                self.client_sock = None

def main(args=None):
    rclpy.init(args=args)
    node = WslBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
