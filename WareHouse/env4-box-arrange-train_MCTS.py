from LLM import *
from prompt_env4 import *
from env4_create import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
from env4_func import *
import tiktoken
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-experiment_trial_num', '--experiment_trial_num', type=int, default=1) # experiemnt trial number
parser.add_argument('-input_error_prompt_token_limit', '--input_error_prompt_token_limit', type=int, default=15000) # The token limit for the input error prompt
parser.add_argument('-model_name_promptLLM', '--model_name_promptLLM', default='gpt-4-1106-preview') # The model name for the promptLLM
parser.add_argument('-model_name_testLLM', '--model_name_testLLM', default='gpt-3.5-turbo-16k-0613') # The model name for the testLLM
parser.add_argument('-min_level', '--min_level', type=int, default=2) # The minimum levels of evolution for the prompt optimization

parser.add_argument('-n_children', '--n_children', type=int, default=8) # The number of son prompts to be expanded in each level
parser.add_argument('-n_selected', '--n_selected', type=int, default=2) # The number of best prompts for further optimization in each level

parser.add_argument('-prompt_method', '--prompt_method', default='PROMST') # The prompt optimization method
parser.add_argument('-with_score_model', '--with_score_model', default='False') # Whether to use the score model
parser.add_argument('-Training_path', '--Training_path', default='../WareHouse/train_set/') # The path to the training set
parser.add_argument('-Testing_path', '--Testing_path', default='../WareHouse/test_set/') # The path to the testing set
parser.add_argument('-base_path', '--base_path', default='../WareHouse/') # The base path for the training set

args = parser.parse_args()
experiment_trial_num = args.experiment_trial_num
input_error_prompt_token_limit = args.input_error_prompt_token_limit
model_name_promptLLM = args.model_name_promptLLM
model_name_testLLM = args.model_name_testLLM
min_level = args.min_level
prompt_method = args.prompt_method
with_score_model = args.with_score_model
Testing_path = args.Testing_path
Training_path = args.Training_path
base_path = args.base_path
n_children = args.n_children
n_selected = args.n_selected

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")

# Create the training and testing environments
if not os.path.exists(Testing_path) or not os.path.exists(Training_path):
    create_env4(Testing_path, repeat_num=10)
    create_env4(Training_path, repeat_num=10)

extra_prompt = 'Each agent can do the actions: ' \
               '1) When the robot is on the track, it can pick up one box whose location is 0.5 away from the robot (either location difference in x or y.). For example, "pick box_1.5_1.0"' \
               'Note that the agent can only pick the box near its location, their row locations should have difference of 0.5, and column difference should be 0.0, e.g., agent0 is in track_1 and column_3 and can do "pick box_1.5_3.0" or "pick box_0.5_3.0".\n' \
               '2) When the robot is on the track, it can move its position with distance 1 either to the left or to the right. For example, "move left", "move right"' \
               '3) When the robot is on the target, it can move its position to the track to get onto the track and carry the boxes. For example, "move to track_1"' \
               '4) When the robot is on the track, it can move its position to the target to pour the box into the target. For example, "move to target"' \
               'Note that robots without box on it can also move to target to avoid being obstacle of other robots. All robots moving to the target will pour their boxes. Hence, the final goal is to pour all the boxes into the target. Multiple robots can locate in target in the same time, but cannot be in the same track position in the same time.\n' \
               'The warehouse playground has left side column 0 and right side, if the agent column is at these two sides, they can only move right or move left but not both directions.\n' \
               'If the agent in the target, it can move to the left side of all the tracks\n' \
               'If the agent is in the left side of the track, it can move to the target and drop the box.'

collision_avoidance_prompt = '[Do remember that each position(track and column locations) can only accommodate one agent each step! Hence, you need to avoid the collision with other agents. Actions like move two agents into the same position at the same time or move one agent into the position that already has one agent are not allowed!]'

Starting_prompt_task_explain = f'''
    You are a central planner directing mobile transporting agents in a warehouse to pick boxes and place them into the target place.
    
    Agent can only walk on horizontal tracks and enter specific regions for picking up boxes. Each agent can only hold one box each time.
    
    {extra_prompt}
    
    Your task is to assign each agent the task in the next step. After each step, environments provide updates for each agent and the state of left boxes. Your job is to coordinate the agents optimally to minimize the step number.
    {collision_avoidance_prompt}
    
    Specify your action plan in this format: {{"agent0":"move left", "agent1":"move to track_1", "agent2":"pick box_1.5_1.0", "agent3":"move to target", "agent4":"move right", "agent5":"pick box_1.5_3.0"}}. Include an agent only if it has actions in the next step.

  '''

if not os.path.exists(base_path):
    raise ValueError(f'base_path {base_path} does not exist')

dir_path = os.path.join(base_path, f'prompt_optimization_train_result')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
save_path = os.path.join(dir_path, f'train_result_dir_trial_{experiment_trial_num}_PromptLLM_{model_name_promptLLM}_TestLLM_{model_name_testLLM}_PromptMethod_{prompt_method}')

tree = Tree(base_path=save_path, Testing_path = Testing_path, n_children=n_children, n_selected=n_selected,
            input_error_prompt_token_limit = input_error_prompt_token_limit, model_name_promptLLM = model_name_promptLLM, model_name_testLLM = model_name_testLLM, prompt_method = prompt_method, with_score_model = with_score_model)

# Check if the directory exists
if os.path.exists(save_path):
    # Remove the directory and its contents
    shutil.rmtree(save_path)

# Create the directory
os.makedirs(save_path)

path_each_prompt = os.path.join(save_path, f'prompt_{tree.node_index}')
if not os.path.exists(path_each_prompt):
    os.makedirs(path_each_prompt)
score, feedback_to_promptLLM_list, success_failure_list, error_string_list = tree.score_prompt(Starting_prompt_task_explain, path_each_prompt)
tree.root = Node(Starting_prompt_task_explain, score, 1, tree.node_index, parent=None,
                 feedback_to_promptLLM_list=feedback_to_promptLLM_list, success_failure_list=success_failure_list, error_string_list=error_string_list)

tree.evolve_tree(tree.root, min_level=min_level)
tree.display_tree(tree.root)  # Display the tree for visualization

# Save the tree to a file
tree.save_tree(tree.root, save_path)
tree.save_display_tree(tree.root)
