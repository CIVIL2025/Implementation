import os
import glob
import tqdm
import natsort
import pickle
import numpy as np
import stable_whisper
import scipy.stats as st
import matplotlib.pyplot as plt

from math import sqrt
from random import sample
from collections import Counter
from difflib import SequenceMatcher


user_study_dir = "/projects/recon/CIVIL/data"
users = natsort.os_sorted((os.listdir(user_study_dir)))
# del users[0:3]
for user in users:
    if user.endswith(".csv"):
        users.remove(user)


civil = {}
traditional = {}
num_demos = {}

############################################## Generate Combined Data ########################################

# # list all civil and traditional files
# civil_position_1 = []
# civil_position_2 = []
# traditional_position_1 = []
# traditional_position_2 = []
# for user in users:
#     if "downsampled" in user:
#         civil_dir = f"{user_study_dir}/{user}/CIVIL"
#         civil_files = natsort.os_sorted(glob.glob(f"{civil_dir}/*.pkl"))
#         civil_position_1.extend(civil_files[:len(civil_files)//2])
#         civil_position_2.extend(civil_files[len(civil_files)//2:])
#         traditional_dir = f"{user_study_dir}/{user}/traditional"
#         traditional_files = natsort.os_sorted(glob.glob(f"{traditional_dir}/*.pkl"))
#         traditional_position_1.extend(traditional_files[:len(traditional_files)//2])
#         traditional_position_2.extend(traditional_files[len(traditional_files)//2:])


# for i in range(3):
#     subsampled_civil = []
#     subsampled_traditional = []
#     subsampled_civil.extend(sample(civil_position_1, 12))
#     subsampled_civil.extend(sample(civil_position_2, 12))
#     subsampled_traditional.extend(sample(traditional_position_1, 14))
#     subsampled_traditional.extend(sample(traditional_position_2, 15))

#     # copy the data
#     civil_save_dir = f"{user_study_dir}/civil_demo_set_{i+6}"
#     # traditional_save_dir = f"{user_study_dir}/traditional_demo_set_{i+6}"
#     os.makedirs(civil_save_dir, exist_ok=True)
#     # os.makedirs(traditional_save_dir, exist_ok=True)

#     for civil_file in subsampled_civil:
#         with open(civil_file, 'rb') as f:
#             data = pickle.load(f)
#         num_files = len(os.listdir(civil_save_dir))
#         civil_file_name = f"demo_{num_files}"
#         with open(f"{civil_save_dir}/{civil_file_name}.pkl", 'wb') as f:
#             pickle.dump(data, f)

#     # for traditional_file in subsampled_traditional:
#     #     with open(traditional_file, 'rb') as f:
#     #         data = pickle.load(f)
#     #     num_files = len(os.listdir(traditional_save_dir))
#     #     traditional_file_name = f"demo_{num_files}"
#     #     with open(f"{traditional_save_dir}/{traditional_file_name}.pkl", 'wb') as f:
#     #         pickle.dump(data, f)
#     print(f"Subsampled {len(subsampled_civil)} civil files and {len(subsampled_traditional)} traditional files")

############################################## Subjective Results #############################################
# # Load the data
# result_file = "user_study/subjective_survey.csv"
# with open(result_file, 'r') as f:
#     lines = f.readlines()
#     lines = [line.strip().split(",") for line in lines]
#     header = lines[0]
#     data = {key: [] for key in header}
#     for line in lines[1:]:
#         for i, key in enumerate(header):
#             data[key].append(line[i])


# intuitive_positive = list(data.keys())[4:-4:2]
# intuitive_negative = list(data.keys())[5:-4:2]
# seamless_positive = list(data.keys())[-4::2]
# seamless_negative = list(data.keys())[-3::2]

# print(intuitive_positive)
# print(intuitive_negative)
# print(seamless_positive)
# print(seamless_negative)

# intuitive_score = []
# seamless_score = []

# # for each row of data
# user_score = {}
# for i in range(len(data[intuitive_positive[0]])):
#     for key in intuitive_positive:
#         if i not in user_score.keys():
#             user_score[i] = {}
#         if "intuitive" not in user_score[i].keys():
#             user_score[i]["intuitive"] = []
#         num_str = data[key][i].strip('"')
#         num = int(num_str)
       
#         user_score[i]["intuitive"].append(num)
#     for key in intuitive_negative:
#         num_str = data[key][i].strip('"')
#         num = int(num_str)
#         user_score[i]["intuitive"].append(8 - num)

#     for key in seamless_positive:
#         if i not in user_score.keys():
#             user_score[i] = {}
#         if "seamless" not in user_score[i].keys():
#             user_score[i]["seamless"] = []
#         num_str = data[key][i].strip('"')
#         num = int(num_str)
#         user_score[i]["seamless"].append(num)
#     for key in seamless_negative:
#         num_str = data[key][i].strip('"')
#         num = int(num_str)
#         user_score[i]["seamless"].append(8 -num)

# avg_score_per_user = {}
# for user, score in user_score.items():
#     avg_score_per_user[user] = {}
#     avg_score_per_user[user]["intuitive"] = sum(score["intuitive"]) / len(score["intuitive"])
#     avg_score_per_user[user]["seamless"] = sum(score["seamless"]) / len(score["seamless"])

# print(avg_score_per_user)

# # get the average scoer across users
# intuitive_score = []
# seamless_score = []

# for user, score in avg_score_per_user.items():
#     intuitive_score.append(score["intuitive"])
#     seamless_score.append(score["seamless"])

# print("Intuitive score: ", intuitive_score)
# print("Seamless score: ", seamless_score)
# total = [intuitive_score, seamless_score]

# mean = [np.mean(total[i]) for i in range(len(total))]
# std = [st.sem(total[i]) for i in range(len(total))]
# x = np.arange(len(total))
# bar_width = 1

# def rgb_to_color(rgb):
#     return [c / 255 for c in rgb]

# colors = [rgb_to_color([255,153,0]), rgb_to_color([160, 212,164])]
# success_bars = plt.bar(x, mean, yerr=std, color=colors, width=bar_width, capsize=5)


# plt.ylabel('Score')
# plt.title(f'Subjective Score')
# plt.xticks(x, ["Intuitive", "Seamless"], fontsize=12)
# plt.yticks(np.arange(0, 8))
# plt.tight_layout()
# # Save the plot
# plt.savefig('subjective_score.svg', dpi=300)

################################################# Language Prompt ################################################
user_study_dir = "/projects/recon/human-placed-markers/user_study"
users = natsort.os_sorted((os.listdir(user_study_dir)))

model = stable_whisper.load_model("turbo")

def consine_similarity(a, b):
    vec1 = Counter(a)
    vec2 = Counter(b)

    dot_product = sum((vec1[ch] * vec2[ch] for ch in vec1))
    magnitude1 = sqrt(sum(count ** 2 for count in vec1.values()))
    magnitude2 = sqrt(sum(count ** 2 for count in vec2.values()))
    if not magnitude1 or not magnitude2:
        return 0.0
    res = dot_product / (magnitude1 * magnitude2)
    return res
    
similarity = {"red coffee cup": [],
              "white robot arm": [],
              "purple coffee pod": [],
              "black coffee maker on the right": [],
              "sugar box": [],
              }

for user in tqdm.tqdm(users):
    if user != "user_8_Jonah":
        continue
    text_list = []
    civil_dir = f"{user_study_dir}/{user}/CIVIL"
    audio_list =  natsort.os_sorted(glob.glob(f"{civil_dir}/*.wav"))
    for audio in audio_list:
        result = model.transcribe(audio)
        result = result.to_txt().split("\n")
        text_list.extend(result)
    print(text_list)
    
    for key in similarity.keys(): 
        for text in text_list:
            # compute similarity
            res = SequenceMatcher(None, key, text).ratio()
            # res = consine_similarity(key, text) # Consine is dependent on the length of the text
            similarity[key].append(res)  

# get the average similarity
for key in similarity.keys():  
    similarity[key] = sum(similarity[key]) / len(similarity[key])  

################################################# Demo time ################################################

# user_study_dir = "/projects/recon/human-placed-markers/user_study"
# users = natsort.os_sorted((os.listdir(user_study_dir)))

# for user in tqdm.tqdm(users):
#     # get all pickle files
#     civil_dir = f"{user_study_dir}/{user}/CIVIL"
#     civil_files = natsort.os_sorted(glob.glob(f"{civil_dir}/*.pkl"))
#     traditional_dir = f"{user_study_dir}/{user}/traditional"
#     traditional_files = natsort.os_sorted(glob.glob(f"{traditional_dir}/*.pkl"))

#     num_demos[user] = {"civil": len(civil_files), "traditional": len(traditional_files)}
#     civil[user] = {"marker_placing_times": [], "demo_times": []}
#     traditional[user] = {"marker_placing_times": [], "demo_times": []}
#     for civil_file in civil_files:
#         with open(civil_file, 'rb') as f:
#             data = pickle.load(f)
#         civil[user]["marker_placing_times"].append(data["marker_placing_time"])
#         civil[user]["demo_times"].append(data["demo_time"])
#     civil[user]["avg_marker_placing_time"] = sum(civil[user]["marker_placing_times"]) / len(civil[user]["marker_placing_times"])
#     civil[user]["avg_demo_time"] = sum(civil[user]["demo_times"]) / len(civil[user]["demo_times"])
    
#     for traditional_file in traditional_files:
#         print(traditional_file)
#         with open(traditional_file, 'rb') as f:
#             data = pickle.load(f)

#         traditional[user]["demo_times"].append(data["demo_time"])

#     traditional[user]["avg_demo_time"] = sum(traditional[user]["demo_times"]) / len(traditional[user]["demo_times"])


# for user, data in civil.items():
#     print(f"{user} CIVIL: {data['marker_placing_times'][0]:.2f}")
#     print(f"{user} CIVIL: {data['avg_demo_time']:.2f}")
# for user, data in traditional.items():
#     print(f"{user} TRADITIONAL: {data['avg_demo_time']:.2f}")

# # average out the CIVIL marker placing time
# avg_civil_marker_placing_time = []
# for user, data in civil.items():
#     avg_civil_marker_placing_time.append(data["marker_placing_times"][0])
# avg_civil_marker_placing_time = sum(avg_civil_marker_placing_time) / len(avg_civil_marker_placing_time)
# print(f"Avg CIVIL marker placing time: {avg_civil_marker_placing_time:.2f}")

# # average out the CIVIL demo time
# avg_civil_demo_time = []
# for user, data in civil.items():
#     avg_civil_demo_time.append(data["avg_demo_time"])
# avg_civil_demo_time = sum(avg_civil_demo_time) / len(avg_civil_demo_time)

# # average out the TRADITIONAL demo time
# avg_traditional_demo_time = []
# for user, data in traditional.items():
#     avg_traditional_demo_time.append(data["avg_demo_time"])
# avg_traditional_demo_time = sum(avg_traditional_demo_time) / len(avg_traditional_demo_time)

# # demo per minute 
# avg_civil_demo_time_per_minute = 60 / avg_civil_demo_time
# avg_traditional_demo_time_per_minute = 60 / avg_traditional_demo_time

# # demos in 7 minutes
# avg_civil_demos_in_7_minutes = avg_civil_demo_time_per_minute * 13
# avg_traditional_demos_in_7_minutes = avg_traditional_demo_time_per_minute * 13
# print(f"Avg CIVIL demo time: {avg_civil_demos_in_7_minutes:.2f}")
# print(f"Avg TRADITIONAL demo time: {avg_traditional_demos_in_7_minutes:.2f}")

# CIVIL demo in 5 minutes: 9.28
# traditional demo in 5 minutes: 11.00

# CIVIL demo in 7 minutes: 13.44
# traditional demo in 7 minutes: 15.02

# CIVIL demo in 8 minutes: 14.86
# traditional demo in 8 minutes: 17.60

# CIVIL demo in 10 minutes: 18.57
# traditional demo in 10 minutes: 21.99

# CIVIL demo in 13 minutes: 24.14
# traditional demo in 13 minutes: 28.59

###############################################rollout results#############################################

# reaching = [[[1,1,1], [0.6666666667, 0.6666666667,0], [1,1,0.6666666667]],[[0, 0,0.3333333333],[1, 0.3333333333, 0.3333333333],[0.6666666667, 0.6666666667, 0.3333333333]]]
# releasing = [[[1, 0.6666666667, 0.6666666667], [0.3333333333, 0, 0], [1, 0.3333333333, 0]], [[0, 0, 0], [0.3333333333, 0, 0], [0.6666666667, 0, 0]]]

# # calculate average of each model

# avg_reaching = []
# avg_releasing = []
# for i in range(len(reaching)):
#     avg_reaching.append([sum(reaching[i][j])/len(reaching[i][j]) for j in range(len(reaching[i]))])
#     avg_releasing.append([sum(releasing[i][j])/len(releasing[i][j]) for j in range(len(releasing[i]))])
# print("Avg reaching: ", avg_reaching)
# print("Avg releasing: ", avg_releasing)

# # mean and standard error for each method
# reach_mean = [np.mean(avg_reaching[i]) for i in range(len(avg_reaching))]
# reach_std = [st.sem(avg_reaching[i]) for i in range(len(avg_reaching))]
# mean = [np.mean(avg_releasing[i]) for i in range(len(avg_releasing))]
# std = [st.sem(avg_releasing[i]) for i in range(len(avg_releasing))]

# print("Mean reaching: ", reach_mean)
# print("Mean releasing: ", mean)
# print("Std reaching: ", reach_std)
# print("Std releasing: ", std)

# # plot the results
# x = np.arange(len(reach_mean))
# bar_width = 0.35
# def rgb_to_color(rgb):
#     return [c / 255 for c in rgb]
# colors = [rgb_to_color([255,153,0]), rgb_to_color([160, 212,164]), rgb_to_color([0, 0, 255]), rgb_to_color([255, 0, 0])]


# reach_bars = plt.bar(x, reach_mean, yerr=reach_std, color=colors[:2], width=bar_width, capsize=5)
# success_bars = plt.bar(x, mean, yerr=std, color=colors[2:], width=bar_width, capsize=5)


# plt.ylabel('Score')
# plt.title(f'Rollout Results')
# plt.xticks(x, ["CIVIL", "TRADITIONAL"], fontsize=12)
# plt.yticks(np.arange(0, 1.1, 0.1))
# plt.tight_layout()
# # Save the plot
# plt.savefig('rollout_results.svg', dpi=300)