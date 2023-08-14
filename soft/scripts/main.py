#!/usr/bin/env python3
import time
from pymavlink import mavutil

def connect_to_ardupilot():
    # Create a connection to the Ardupilot
    # Change the connection string as per your setup.
    # For serial connection: 'serial:/dev/ttyUSB0'
    # For UDP connection (common for Raspberry Pi): 'udp:127.0.0.1:14550'
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    print (1)

    # Wait for the autopilot to get connected
    master.wait_heartbeat()
    print (2)
    print("Heartbeat from Ardupilot (system %u component %u)" % (master.target_system, master.target_component))
    
    return master

def main():
    master = connect_to_ardupilot()

    while True:
        # Fetch messages
        try:
            message = master.recv_match(type=['POSITION', 'ATTITUDE'], blocking=True)
            if message:
                print(message)
        except Exception as e:
            print("Error: ", e)
            break

        time.sleep(0.2)

if __name__ == '__main__':
    main()

