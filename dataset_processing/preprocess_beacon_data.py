import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from natsort import os_sorted

data_fd = 'expert_demos/push_button'

demo_list = os_sorted(os.listdir(data_fd))
print(demo_list)


for path in demo_list:

    demo_path = f"{data_fd}/{path}"
    print(demo_path)
    with open(f"{data_fd}/{path}", "rb") as file:
        demo = pickle.load(file)

    steps = len(demo['img'])
    n_beacons = 1
    beacon_dim = 6
    known_ids = [2] # 1: plate, 2: cup. for the place cup on the plate task
    # BEACON_MAPPING = {1: slice(0, 6),
    #                 2: slice(6, 12)}
    BEACON_MAPPING = {2: slice(0, 6)}

    beacon_measurements = np.zeros((steps, n_beacons*beacon_dim))

    for i, info in enumerate(demo['beacon_info']):
        rvecs, tvecs, corners, ids = info
        beacon_reading = {id: [] for id in known_ids}

        if ids is not None:
            for j, id in enumerate(ids):
                if id in known_ids:
                    beacon_reading[id[0]].append(np.concatenate((tvecs[j], rvecs[j])).reshape(1, beacon_dim))

            avg_reading = np.empty((1, n_beacons*beacon_dim))
            avg_reading[:] = np.nan
            for id in known_ids:
                avg_reading[:, BEACON_MAPPING[id]] = np.mean(beacon_reading[id], axis=0)

            beacon_measurements[i] = avg_reading

    demo['beacons'] = beacon_measurements

    nan_index = np.isnan(beacon_measurements).any(axis=-1)
    
    print(f'This demo has {steps} samples, and {len(beacon_measurements[nan_index])} missing beacon readings')
    # print(beacon_measurements.shape)
    # print(beacon_measurements[:, 0].shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # ax.plot(beacon_measurements[:, 0], beacon_measurements[:, 1], beacon_measurements[:, 2], 'ro')
    # ax.plot(beacon_measurements[:, 6], beacon_measurements[:, 7], beacon_measurements[:, 8], 'bo')
    # plt.title(path.replace('.pkl', ''))
    # plt.show()

    with open(demo_path, "wb") as file:
        pickle.dump(demo, file)