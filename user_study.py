
import os
import time
import hydra
import socket
import pickle
from natsort import os_sorted
from termcolor import colored
from omegaconf import DictConfig, OmegaConf

from utils import VoiceRecorder
from utils import FR3, Joystick


def mirror_send(msg, sock):
    try:
        
        sock.sendall((msg + '\n').encode())
    except: 
        pass

def demonstration(cfg, demo_name, gripper_cam, static_cam, conn, conn_gripper, interface, robot, sock, control_mode):
    user_id = cfg.user_id
    mode = cfg.mode
    os.makedirs(f"{cfg.save_dir}/user_{user_id}/{mode}", exist_ok=True)
    existing_demo = os_sorted(os.listdir(f"{cfg.save_dir}/user_{user_id}/{mode}"))
    for file in existing_demo:
        demo = file.split(".")[0]
        if demo == demo_name:
            print(colored("[ERROR]", "red") + f"Demo name already exists, please choose another name")
            return 0

    # Home position
    robot.send2gripper(conn_gripper, "o")
    START = [-1.45233, 0.91503, 1.39972, -1.92329, -0.310326, 2.04672, 2.38064] # coffee making
    robot.go2position(conn, START)

    print(colored(f"[*] Starting demonstration: {demo_name}", "green"))
    mirror_send(f"[*] Starting demonstration: {demo_name}", sock)
    mirror_send("\n", sock)

    # Beacon Placement
    marker_placing_time = 0.0
    if mode != "play":
        print("[*] Press 'A' and start placing the markers (if they are not placed) OR move around the objects:")
        mirror_send("[*] Press 'A' and start placing the markers (if they are not placed) OR move around the objects:", sock)

        while True: 
            z, (a_button, b_button, x_button, y_button, back_button), start_button = interface.getInput()
            if a_button:
                time.sleep(0.5)
                start_time = time.time()
                break

        print(colored("[*] Placing markers...", "green"))
        mirror_send("[*] Placing markers...", sock)
        mirror_send("\n", sock)
        print("[*] Press 'A' again when you finished placing the markers")
        mirror_send("[*] Press 'A' again when you finished placing the markers", sock)

        while True:
            z, (a_button, b_button, x_button, y_button, back_button), start_button = interface.getInput()
            if a_button:
                marker_placing_time = time.time() - start_time
                print(colored("[*] Finished placing markers", "green"))
                print(f'marker placing time: {marker_placing_time}')
                mirror_send("[*] Finished placing markers.", sock)
                mirror_send("\n", sock)
                time.sleep(0.5)
                break

    record = False
    step_time = 0.1
    start_state = robot.readState(conn)
    print(colored(f"[*] MODE: {mode}", "green"))
    print("[*] Press 'A' to start demonstrating the task:")
    if mode == 'traditional':
        mirror_send("[*] Press 'A' to start demonstrating the task:", sock)
    elif mode == 'play':
        mirror_send("[*] Press 'A' to start moving task relevant objets around:", sock)
    else:
        mirror_send("[*] Press 'A' to start demonstrating the task and giving verbal instructions:", sock)

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
                #   'user_cam': [],
                  'beacon_info': [],
                  'timestamp': []
                  }

    run = True
    recording_start_time = None
    while run:

        z, (a_button, b_button, x_button, y_button, back_button), start_button = interface.getInput()
        z = interface.getAction(z)

        if a_button and not record:
            record = True
            time.sleep(0.1)
            print(colored("[*] Recording Started...", "green"))
            mirror_send("[*] Recording Started...", sock)
            recording_start_time = time.time()
            start_time = recording_start_time
            voice_recorder = VoiceRecorder(f"{cfg.save_dir}/user_{user_id}/{mode}", demo_name)
            voice_recorder.keep_recording = True
            keep_recording = True
            voice_recorder.start()
            mirror_send("\n", sock)
            print("[*] Press 'start' to stop recording")
            mirror_send("[*] Press 'start' to stop recording", sock)
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
            if recording_start_time is not None:
                demonstration_time = time.time() - recording_start_time
            else: demonstration_time = 0
            if keep_recording:
                voice_recorder.keep_recording = False 
                keep_recording = False
                time.sleep(1)
            run = False
            
        if mode == 'play' and recording_start_time is not None and (time.time() - recording_start_time) > 120:
            demonstration_time = time.time() - recording_start_time
            if keep_recording:
                voice_recorder.keep_recording = False 
                keep_recording = False
                time.sleep(1)
            
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
            # trajectory['user_cam'].append(user_cam.frame)
            trajectory['joint_vel'].append(qdot)
            trajectory['ee_vel'].append(xdot)
            trajectory['gripper_action'].append(gripper_action)
            trajectory['beacon_info'].append([static_cam.rvecs, static_cam.tvecs, static_cam.corners, static_cam.ids])
            trajectory['timestamp'].append(curr_time - recording_start_time)
            start_time = curr_time

        # Update binary gripper state
        if gripper_action == -1:
            gripper_state = -1
        elif gripper_action == 1:
            gripper_state = 1


    print(colored("[*] Finished demonstrating the task", "green"))
    mirror_send("[*] Finished demonstrating the task.", sock)
    print(f"This demo has: {len(trajectory['joint_state'])} data points")
    print(f"Marker placement time: {marker_placing_time:.2f} seconds")
    print(f"Demo time: {demonstration_time:.2f} seconds")
    trajectory['marker_placing_time'] = marker_placing_time
    trajectory['demo_time'] = demonstration_time

    # Return robot home
    START = [-1.45233, 0.91503, 1.39972, -1.92329, -0.310326, 2.04672, 2.38064] # coffee making
    robot.go2position(conn, START)
    robot.send2gripper(conn_gripper, "o")

    # voice_recorder.save_recording(trajectory)
    voice_path = voice_recorder.save_path
    voice_recorder.join()

    # Let the user choose to save or not
    mirror_send("[*] Press 'start' to save the demo or 'back' to discard it", sock)
    save = False
    while True: 
        z, (a_button, b_button, x_button, y_button, back_button), start_button = interface.getInput()
        if back_button:
            time.sleep(0.5)
            save = False
            break
        if start_button:
            time.sleep(0.5)
            save = True
            break
    if not save:
        os.remove(voice_path) # delete the voice recording
        print("[*] Demo discarded.")
        mirror_send("[*] Demo discarded.", sock)
        return 0

    with open(f"{cfg.save_dir}/user_{user_id}/{mode}/{demo_name}.pkl", "wb") as handle:
        pickle.dump(trajectory, handle)
    print("[*] Demo saved.")
    mirror_send("[*] Demo saved.", sock)

    if mode != "traditional":
        return demonstration_time + marker_placing_time
    else:
        return demonstration_time


@hydra.main(version_base=None, config_path='conf', config_name='user_study')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    demo_num = 0
    total_time = 0

    static_cam = hydra.utils.instantiate(cfg.cameras.static)
    gripper_cam = hydra.utils.instantiate(cfg.cameras.gripper)
    # user_cam = hydra.utils.instantiate(cfg.cameras.user)
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

    # Setup socket
    MIRROR_HOST = 'localhost'
    MIRROR_PORT = 65432
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((MIRROR_HOST, MIRROR_PORT))
        mirror_enabled = True
    except ConnectionRefusedError:
        print("[!] Mirror listener not available")
        mirror_enabled = False
        return
    
    
    if cfg.mode == 'play':
        total_demo_time = 120
    else:
        total_demo_time = 300
        
    while total_time < total_demo_time:  # in 5 minutes

        demo_name = f"demo_{demo_num}"
        demo_time = demonstration(cfg, demo_name, gripper_cam, static_cam, conn, conn_gripper, interface, robot, sock, control_mode)
        demo_num += 1
        total_time += demo_time
        print(colored(f"Total time: {total_time:.2f} seconds", "green"))

    if mirror_enabled:
        sock.close()
    # Kill camera threads
    static_cam.done = True
    gripper_cam.done = True


if __name__== '__main__':
    main()