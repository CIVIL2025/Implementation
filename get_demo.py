
import os
import cv2
import time
import hydra
import pickle
from natsort import os_sorted
from utils import FR3, Joystick
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='conf', config_name='get_demo')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.demo_name is None:
        print('Missing save name, here is a list of names')
        print('already saved in the destination folder')
        os.makedirs(cfg.save_dir, exist_ok=True)
        print(os_sorted(os.listdir(cfg.save_dir)))

        demo_name = input("input an unused demo name: ")
    else:
        demo_name = cfg.demo_name


    static_cam = hydra.utils.instantiate(cfg.cameras.static)
    gripper_cam = hydra.utils.instantiate(cfg.cameras.gripper)
    time.sleep(1)

    robot = FR3()
    control_mode = "v"

    print('[*] Connecting to robot...')
    conn = robot.connect(8080)
    print('[*] Connection complete')

    print('[*] Connecting to gripper')
    conn_gripper = robot.connect(8081)
    print('[*] Connection complete')

    interface = Joystick()

    # Home position
    # START = [-1.37002, 0.900083, 1.41625, -1.88775, -0.17674, 2.05773, -0.722304] # stirring task
    # START = [-1.45233, 0.91503, 1.39972, -1.92329, -0.310326, 2.04672, 2.38064] # coffee making
    # START = [-0.0413756, -0.08899825, 0.0123554, -1.80487, -0.000971178, 2.08393, 0.71632] # push button
    # START = [-0.11452, 0.0462251, 0.205407, -1.31481, -0.0074332, 1.37608, 0.882886] # red_plate red_cup
    # START = [-0.0832917, 0.0534155, 0.19788, -1.47245, -0.061943, 1.88297, 0.866635] # bowl pulling
    # START = [-0.123849, 0.173138, 0.407359, -2.13113, -0.110262, 2.30485, 1.13426] #placing
    START = [-0.177101, -0.0234967, 0.208954, -1.5112, -0.10339, 1.91946, 0.846834] # Doh

    robot.go2position(conn, START)
    robot.send2gripper(conn_gripper, "o")
    
    # Beacon Placement
    print("Press 'A' on the keyboard after you finished placing the beacons")

    while True:
        # if keyboard.is_pressed('a'):
        if static_cam.key_pressed == 'a':
            rvecs = static_cam.rvecs
            tvecs = static_cam.tvecs 
            corners = static_cam.corners
            ids = static_cam.ids

            if rvecs is not None:
                break

    record = False
    step_time = 0.1
    start_state = robot.readState(conn)
    print("[*] Press A to start recording demos")

    gripper_state = 1
    gripper_action = 1

    # Data to be saved
    trajectory = {'joint_state': [],
                  'ee_state': [],
                  'gripper_state': [],
                  'joint_vel': [],
                  'ee_vel': [],
                  'gripper_action': [],
                  'img': [],
                  'img_gripper': [],
                  'beacon_info': []
                  }

    run = True
    while run:

        z, (a_button, b_button, x_button, y_button, back_button), start_button = interface.getInput()
        z = interface.getAction(z)

        if a_button and not record:
            record = True
            print("[*] Recording Started")
            time.sleep(0.1)
            start_time = time.time()
        if b_button and record:
            record = False
            print(f"This demo has: {len(trajectory['joint_state'])} data points")
            print("[*] Press A to continue recording or START to save the recorded demo")
            time.sleep(1.0)
        if x_button:
            print("[*] Closing gripper...")
            robot.send2gripper(conn_gripper, "c")
            gripper_action = -1
        if y_button:
            print("[*] Opening gripper...")
            robot.send2gripper(conn_gripper, "o")
            gripper_action = 1
        if start_button:
            run = False

        # Convert joystick input to joint velocity
        state = robot.readState(conn)
        xdot = z 
        qdot = robot.xdot2qdot(xdot, state)

        robot.send2robot(conn, qdot, control_mode)

        # Update demo if recording 
        curr_time = time.time()
        if record and curr_time - start_time >= step_time:
            trajectory['joint_state'].append(state['q'])
            trajectory['ee_state'].append(state['x'])
            trajectory['gripper_state'].append(gripper_state)
            trajectory['img'].append(static_cam.frame)
            trajectory['img_gripper'].append(gripper_cam.frame)
            trajectory['joint_vel'].append(qdot)
            trajectory['ee_vel'].append(xdot)
            trajectory['gripper_action'].append(gripper_action)
            trajectory['beacon_info'].append([static_cam.rvecs, static_cam.tvecs, static_cam.corners, static_cam.ids])
            start_time = curr_time

        # Update binary gripper state
        if gripper_action == -1:
            gripper_state = -1
        elif gripper_action == 1:
            gripper_state = 1

        cv2.imshow('gripper', gripper_cam.frame)
        cv2.waitKey(1)


    print(f"This demo has: {len(trajectory['joint_state'])} data points")

    # Kill camera threads
    static_cam.done = True
    gripper_cam.done = True

    # Return robot home
    robot.go2position(conn, START)
    robot.send2gripper(conn_gripper, "o")

    # # Fix the first action
    # trajectory['joint_vel'][0] = np.array(trajectory['joint_state'][1]) - np.array(trajectory['joint_state'][0])
    # trajectory['ee_vel'][0] = np.array(trajectory['ee_state'][1]) - np.array(trajectory['ee_state'][0])

    os.makedirs(cfg.save_dir, exist_ok=True)
    # Save data before closing
    with open(f"{cfg.save_dir}/{demo_name}.pkl", "wb") as handle:
        pickle.dump(trajectory, handle)



if __name__== '__main__':
    main()