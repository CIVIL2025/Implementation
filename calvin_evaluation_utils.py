import contextlib
import logging
import numpy as np
from numpy import pi
import pyhash
from termcolor import colored


hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def generate_random_init_position(env):
    constraints = [
        (-0.28, 0.29),
        (-0.26, -0.14),
        (0.54, 0.61),
        [(-3.13, -2.96), (2.81, 3.13)],  
        (-0.34, 0.1),
        (1.15, 1.64)
    ]

    init_robot_obs = np.array([
        round(np.random.uniform(low, high), 4) if isinstance(low, tuple) is False 
        else round(np.random.choice([np.random.uniform(*low), np.random.uniform(*high)]), 4)
        for low, high in constraints
    ])
    init_position = ((init_robot_obs[:3]), (init_robot_obs[3:6]), ([1.0]))
    init_position = tuple(map(tuple, init_position)) 

    # print(colored('[target] ', 'green')+ f'{init_position}')

    obs, _, _, _ = env.step(init_position)
    error = sum(init_robot_obs - obs['robot_obs'][:6])
    step = 0
    while abs(error) > 0.001 and step < 50:
        obs, _, _, _ = env.step(init_position)
        error = sum(init_robot_obs - obs['robot_obs'][:6])
        step += 1
    # print(colored("[actual] ", 'yellow') + f"{obs['robot_obs'][:6]}")
    return obs['robot_obs']


def get_env_state_for_initial_condition(initial_condition, cfg_dict, env):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])

    # Bounds for three positions on the table
    block_table_min = [
        np.array([0.00000896e-02, -0.45000177e-01, 4.59990009e-01]), # middle
        np.array([2.29995412e-01, -0.1995140e-01, 4.59990010e-01]), # right
        np.array([-3.80000896e-01, -0.5000177e-01, 4.59990009e-01]), # left
    ]

    block_table_max = [
        np.array([1.00000896e-01, -1.20000177e-01, 4.59990009e-01]),
        np.array([4.00005412e-01, -1.2995140e-01, 4.59990010e-01]),
        np.array([-2.00000896e-01, -1.40000177e-01, 4.59990009e-01]), 
    ]

    block_table = []

    for table_min, table_max in zip(block_table_min, block_table_max):
        block_table.append(np.random.uniform(table_min, table_max))

    # we want to have a "deterministic" random seed for each initial condition
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        if "grasped" in initial_condition.keys() and initial_condition["grasped"]:
            robot_obs[-1] = -1.0
            robot_obs[-9] = 0.001
        if "red_on_middle" in cfg_dict["additional_constrains"]:
            block_table = np.array(block_table)
            np.random.shuffle(block_table[1:])
        elif "red_on_left" in cfg_dict["additional_constrains"]:
            block_table[0], block_table[2] = block_table[2], block_table[0]
            block_table = np.array(block_table)
            np.random.shuffle(block_table[1:])
        elif "red_on_right" in cfg_dict["additional_constrains"]:
            block_table[0], block_table[1] = block_table[1], block_table[0]
            block_table = np.array(block_table)
            np.random.shuffle(block_table[1:])
        elif "red_on_edge" in cfg_dict["additional_constrains"]:
            block_table[0], block_table[2] = block_table[2], block_table[0]
            block_table = np.array(block_table)
            np.random.shuffle(block_table[:2])
            np.random.shuffle(block_table[1:])
        else:
            np.random.shuffle(block_table)

        if "block_stacking" in cfg_dict["additional_constrains"] and initial_condition['lightbulb']:
            initial_condition['blue_block'] = 'table'
        elif "block_stacking" in cfg_dict["additional_constrains"] and not initial_condition['lightbulb']:
            initial_condition['pink_block'] = 'table'

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        elif initial_condition["slider"] == "right":
            scene_obs[0] = 0.0

        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        elif initial_condition["drawer"] == "closed":
            scene_obs[1] = 0.0

        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        elif initial_condition["red_block"] == "grasped":
            scene_obs[6:9] = robot_obs[:3]
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        else:
            scene_obs[12:15] = block_table[1]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[2]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    
    # Apply random action to set random robot initialization position
    if cfg_dict["random_robot_initial_position"]:
        env.reset(robot_obs, scene_obs)
        robot_obs = generate_random_init_position(env)

    return robot_obs, scene_obs
