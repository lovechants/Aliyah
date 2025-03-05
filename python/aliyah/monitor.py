import os 
import threading
import time
import socket
import json 
from typing import Dict, Any, Optional, List
from datetime import datetime
import zmq

def log_debug(msg):
    with open('/tmp/aliyah_python.log', 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        f.write(f'[{timestamp}] {msg}\n')

class TrainingMonitor:
    def __init__(self):
        self.paused = False
        self.should_stop = False
        self._lock = threading.Lock()
        self._connected = False
        self._setup_zmq()
        self._last_batch_update = time.time()
        self._last_viz_update = time.time()
        
    def _setup_zmq(self):
        log_debug("Setting up ZMQ sockets")
        try:
            self.context = zmq.Context()
            
            # Setup command socket (REP) - we BIND here
            self.command_socket = self.context.socket(zmq.REP)
            self.command_socket.bind("tcp://127.0.0.1:5555")
            
            # Setup metrics socket (PUB) - we BIND here
            self.metrics_socket = self.context.socket(zmq.PUB)
            self.metrics_socket.bind("tcp://127.0.0.1:5556")
            
            # Wait for sockets to be ready
            time.sleep(0.1)
            
            self._connected = True
            log_debug("ZMQ sockets bound successfully")
            
            # Start command listener
            self._start_control_listener()
            
        except Exception as e:
            log_debug(f"Failed to setup ZMQ: {e}")
            self._connected = False
            self.context = None

    def _start_control_listener(self):
        def listener():
            log_debug("Starting command listener")
            while not self.should_stop:
                try:
                    # Non-blocking receive
                    try:
                        message = self.command_socket.recv_string(zmq.NOBLOCK)
                        log_debug(f"Received command: {message}")
                        
                        # Send acknowledgment
                        self.command_socket.send_string("ACK")
                        
                        # Handle command
                        self._handle_command(message)
                    except zmq.Again:
                        time.sleep(0.01)  # Small sleep when no message
                        continue
                        
                except Exception as e:
                    log_debug(f"Command listener error: {e}")
                    time.sleep(0.1)
            log_debug("Command listener stopping")

        self.control_thread = threading.Thread(target=listener, daemon=True)
        self.control_thread.start()
        log_debug("Command listener thread started")

    def check_control(self) -> bool:
        if self.paused:
            self.send_update("status", {"state": "paused"})
            while self.paused and not self.should_stop:
                time.sleep(0.1)  # Small sleep to prevent busy waiting
            # Only send resumed status if we weren't stopped
            if not self.should_stop:
                self.send_update("status", {"state": "resumed"})
        return not self.should_stop

    def _handle_command(self, cmd: str):
        try:
            cmd_data = json.loads(cmd.strip())
            command = cmd_data.get("command")
            log_debug(f"Processing command: {command}")
            
            if command == "pause":
                self.paused = True
                self.send_update("status", {"state": "paused"})
            elif command == "resume":
                self.paused = False
                self.send_update("status", {"state": "resumed"})
            elif command == "stop":
                self.should_stop = True
                self.send_update("status", {"state": "stopped"})
            else:
                log_debug(f"Unknown command: {command}")
        except Exception as e:
            log_debug(f"Error handling command: {e}")

    def log_batch(self, batch_idx: int, loss: float, accuracy: float, **extra_metrics):
        """Send batch metrics"""

        current_time = time.time()
        if current_time - self._last_batch_update < 0.1:
            return 
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            **extra_metrics  # Allow for additional metrics
        }
        
        self.send_update("batch", {
            "batch": batch_idx,
            "metrics": metrics
        })
        self._last_batch_update = current_time

    def log_epoch(self, epoch: int, loss: float, accuracy: float, **extra_metrics):
        """Send epoch metrics"""
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            **extra_metrics  # Allow for additional metrics | Ok need to fix this output but it essentially works but not fully TODO fix this and output parsing but whatever 
        }
        
        self.send_update("epoch", {
            "epoch": epoch,
            "metrics": metrics
        })
    def log_layer_activation(self, layer_idx: int, node_idx: int, activation: float):
        """Send individual node activation update"""
        self.send_update("activation", {
            "layer": layer_idx,
            "node": node_idx,
            "value": activation
        })

    def log_connection_flow(self, from_layer: int, from_node: int, 
                           to_layer: int, to_node: int, active: bool):
        """Send connection state update"""
        self.send_update("connection", {
            "from": {"layer": from_layer, "node": from_node},
            "to": {"layer": to_layer, "node": to_node},
            "active": active
        })

    def log_layer_state(self, layer_idx: int, activations: List[float]):
        """Send entire layer's activation state"""

        current_time = time.time()
        if current_time - self._last_viz_update < 0.25:
            return
        self.send_update("layer_state", {
            "layer": layer_idx,
            "activations": activations
        })
        self._last_viz_update = current_time

    def send_update(self, update_type: str, data: Dict[str, Any]):
        """Send any type of update to the UI"""
        if not self._connected:
            log_debug("Cannot send update - not connected")
            return

        try:
            message = {
                "type": update_type,
                "timestamp": time.time(),
                "data": data
            }
            log_debug(f"Preparing to send message: {json.dumps(message)}")
            
            with self._lock:
                self.metrics_socket.send_string(json.dumps(message))
                # Add small sleep to ensure message is sent
                time.sleep(0.001)
                
            log_debug(f"Successfully sent {update_type} update")
        except Exception as e:
            log_debug(f"Failed to send update: {e}")
            # Try to reconnect if there was an error
            self._setup_zmq()

monitor = TrainingMonitor()

