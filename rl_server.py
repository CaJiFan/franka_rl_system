#!/usr/bin/env python3
"""
rl_server.py
------------
A remote ZMQ Server that runs the Stable Baselines 3 SAC policy.
It waits for an observation vector from the robot, runs a forward pass,
and replies with the 6D action.

Run this on your CPU server!
"""

import zmq
import numpy as np
import time

try:
    from stable_baselines3 import SAC
except ImportError:
    print("Warning: stable_baselines3 not found! Make sure to install them on the server.")

def main():
    # 1. Load the SAC Model
    # CPU is perfectly fine (and often faster for single-batch real-time inference)
    # The device='cpu' flag ensures no GPU is required.
    model_path = "sac_wipe_policy.zip" # Update with your actual trained model path!
    
    print(f"Loading SAC policy from {model_path} onto CPU...")
    try:
        model = SAC.load(model_path, device='cpu')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model '{model_path}': {e}")
        print("Using a dummy random policy for network testing.")
        model = None

    # 2. Setup ZMQ Server (REP socket replies to REQ requests)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:5555")
    
    print("\n[RL Server] Listening on port 5555 for observations...")

    # Wait for requests from the robot
    while True:
        try:
            # 1. Receive the observation bytes
            message = socket.recv()
            
            # 2. Convert bytes back to a numpy array (Float32)
            obs = np.frombuffer(message, dtype=np.float32)
            
            t0 = time.time()
            # 3. Predict action
            if model is not None:
                # deterministic=True disables exploration noise during deployment
                action, _states = model.predict(obs, deterministic=True)
            else:
                # Dummy random action if no model is loaded (6D cartesian velocity)
                action = np.random.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)
            
            # Ensure action is float32 to match the C-struct byte format
            action = action.astype(np.float32)
            t1 = time.time()
            
            # 4. Send the 6D action back to the robot
            socket.send(action.tobytes())
            
            # Print stats occasionally to prove it's working without spamming the console
            if np.random.rand() < 0.05:
                print(f"Obs Shape: {obs.shape} | Output Action: {action.round(3)} | Inference time: {(t1-t0)*1000:.2f} ms")
                
        except KeyboardInterrupt:
            print("\nShutting down RL Server...")
            break
        except Exception as e:
            print(f"Error during inference loop: {e}")
            # Try to reply with zeros so the client doesn't freeze waiting for a response
            err_action = np.zeros(6, dtype=np.float32)
            socket.send(err_action.tobytes())

if __name__ == "__main__":
    main()
