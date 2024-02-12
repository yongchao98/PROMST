# Box moving to target with collisions
from prompt_env2 import *
from LLM import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time

def corner_position(pg_row_i, pg_column_j):
  corner_position_list = [(float(pg_row_i), float(pg_column_j)), (float(pg_row_i), float(pg_column_j + 1)), (float(pg_row_i + 1), float(pg_column_j)),
   (float(pg_row_i + 1), float(pg_column_j + 1))]
  return corner_position_list

def judge_move_box2pos_box2target_func(key, value, pg_dict_original):
  if not (str(key[0] - 0.5) + '_' + str(key[1] - 0.5) in pg_dict_original.keys() \
          and str(key[0] - 0.5) + '_' + str(key[1] + 0.5) in pg_dict_original.keys() \
          and str(key[0] + 0.5) + '_' + str(key[1] - 0.5) in pg_dict_original.keys() \
          and str(key[0] + 0.5) + '_' + str(key[1] + 0.5) in pg_dict_original.keys() \
          and np.mod(key[0], 1) == 0.5 and np.mod(key[1], 1) == 0.5):
    return None, False, False, f'Agent[{float(key[0])}, {float(key[1])}] is not in the agent list. '

  if value[0] in pg_dict_original[str(key[0] - 0.5) + '_' + str(key[1] - 0.5)]:
    box_location = (key[0] - 0.5, key[1] - 0.5)
  elif value[0] in pg_dict_original[str(key[0] - 0.5) + '_' + str(key[1] + 0.5)]:
    box_location = (key[0] - 0.5, key[1] + 0.5)
  elif value[0] in pg_dict_original[str(key[0] + 0.5) + '_' + str(key[1] - 0.5)]:
    box_location = (key[0] + 0.5, key[1] - 0.5)
  elif value[0] in pg_dict_original[str(key[0] + 0.5) + '_' + str(key[1] + 0.5)]:
    box_location = (key[0] + 0.5, key[1] + 0.5)
  else:
    return None, False, False, ''

  if type(value[1]) == tuple and (np.abs(key[0]-value[1][0])==0.5 and np.abs(key[1]-value[1][1])==0.5):
    return box_location, True, False, ''
  elif type(value[1]) == str and value[1] in pg_dict_original[str(key[0])+'_'+str(key[1])] and value[0][:4] == 'box_' and value[1][:7] == 'target_' and value[0][4:] == value[1][7:]:
    return box_location, False, True, ''
  else:
    return None, False, False, f'Your assigned task for {key[0]}_{key[1]} is not in the doable action list; '


def state_update_func(pg_row_num, pg_column_num, pg_dict):
  pg_dict_copy = copy.deepcopy(pg_dict)
  state_update_prompt = ''
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      square_item_list = pg_dict_copy[str(i + 0.5) + '_' + str(j + 0.5)]
      state_update_prompt += f'Agent[{i+0.5}, {j+0.5}]: I am in square[{i+0.5}, {j+0.5}], I can observe {square_item_list}, I can do '
      action_list = []
      for corner_x, corner_y in corner_position(i, j):
        if len(pg_dict_copy[str(corner_x)+'_'+str(corner_y)]) == 1:
          box = pg_dict_copy[str(corner_x)+'_'+str(corner_y)][0]
          for surround_index in corner_position(i, j):
            if surround_index != (corner_x, corner_y):
              action_list.append(f'move({box}, position{surround_index})')
          if 'target'+box[3:] in pg_dict_copy[str(i+0.5)+'_'+str(j+0.5)]:
            action_list.append(f'move({box}, target{box[3:]})')
      state_update_prompt += f'{action_list}\n'
  return state_update_prompt


def action_from_response(pg_dict_input, original_response_dict):
  collision_check = False
  env_act_feedback = ''
  pg_dict_original = copy.deepcopy(pg_dict_input)
  transformed_dict = {}
  for key, value in original_response_dict.items():
    coordinates = tuple(map(float, re.findall(r"\d+\.?\d*", key)))

    # match the item and location in the value
    match = re.match(r"move\((.*?),\s(.*?)\)", value)
    if match:
      item, location = match.groups()
      if "position" in location:
          location = tuple(map(float, re.findall(r"\d+\.?\d*", location)))
      transformed_dict[coordinates] = [item, location]

  for key, value in transformed_dict.items():
    print(f"Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
    box_location, judge_move_box2pos, judge_move_box2target, feedback = judge_move_box2pos_box2target_func(key, value, pg_dict_original)
    if judge_move_box2pos == True:
      pg_dict_original[str(box_location[0])+'_'+str(box_location[1])].remove(value[0])
      pg_dict_original[str(value[1][0])+'_'+str(value[1][1])].append(value[0])
    elif judge_move_box2target == True:
      pg_dict_original[str(box_location[0])+'_'+str(box_location[1])].remove(value[0])
      pg_dict_original[str(key[0])+'_'+str(key[1])].remove(value[1])
    else:
      #print(f"Error, Iteration Num: {iteration_num}, Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
      env_act_feedback += f'Your assigned task for {key[0]}_{key[1]} is not in the doable action list; '
  for key, value in transformed_dict.items():
    box_location, judge_move_box2pos, judge_move_box2target, feedback = judge_move_box2pos_box2target_func(key, value,
                                                                                                 pg_dict_original)
    if judge_move_box2pos == True and len(pg_dict_original[str(value[1][0]) + '_' + str(value[1][1])]) > 1:
      collision_check = True
      break

  return pg_dict_original, collision_check, env_act_feedback

def env_create(pg_row_num = 5, pg_column_num = 5, box_num_low_bound = 2, box_num_upper_bound = 2, color_list = ['blue', 'red', 'green', 'purple', 'orange']):
  # pg_dict records the items in each square over steps, here in the initial setting, we randomly assign items into each square
  pg_dict = {}
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      pg_dict[str(i+0.5)+'_'+str(j+0.5)] = []
  for i in range(pg_row_num+1):
    for j in range(pg_column_num+1):
      pg_dict[str(float(i))+'_'+str(float(j))] = []

  for color in color_list:
    box_num = random.randint(box_num_low_bound, box_num_upper_bound)
    for _ in range(box_num):
      N_box = random.randint(0, pg_row_num*pg_column_num - 1)
      a_box = N_box // pg_column_num
      b_box = N_box % pg_column_num
      N_target = random.randint(0, pg_row_num*pg_column_num - 1)
      a_target = N_target // pg_column_num
      b_target = N_target % pg_column_num
      corner_list = [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
      random.shuffle(corner_list)
      for random_x, random_y in corner_list:
        if len(pg_dict[str(float(a_box) + random_x)+'_'+str(float(b_box) + random_y)]) == 0:
          pg_dict[str(float(a_box) + random_x) + '_' + str(float(b_box) + random_y)].append('box_' + color)
          pg_dict[str(a_target+0.5)+'_'+str(b_target+0.5)].append('target_' + color)
          break
  print(pg_dict)
  print('\n')
  return pg_dict

def create_env2(Saving_path, repeat_num = 10):
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  for i ,j in [(2,2), (2,4), (4,4), (4,8)]:
    if not os.path.exists(Saving_path+f'/env_pg_state_{i}_{j}'):
      os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/env_pg_state_{i}_{j}')
      os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}', exist_ok=True)

    for iteration_num in range(repeat_num):
      # Define the total row and column numbers of the whole playground, and the item number of each colored target and box
      pg_row_num = i; pg_column_num = j; box_num_low_bound = 1; box_num_upper_bound = 1
      # Define the used colors
      color_list = ['blue', 'red', 'green', 'purple', 'orange']
      pg_dict = env_create(pg_row_num, pg_column_num, box_num_low_bound, box_num_upper_bound, color_list)
      os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}/pg_state{iteration_num}', exist_ok=True)
      with open(Saving_path+f'/env_pg_state_{i}_{j}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'w') as f:
        json.dump(pg_dict, f)