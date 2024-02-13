from LLM import *
from prompt_env1 import *
from env1_create import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
from env1_func import *
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
parser.add_argument('-Training_path', '--Training_path', default='../BoxNet1/train_set/') # The path to the training set
parser.add_argument('-Testing_path', '--Testing_path', default='../BoxNet1/test_set/') # The path to the testing set
parser.add_argument('-base_path', '--base_path', default='../BoxNet1/') # The base path for the training set

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
    create_env1(Testing_path, repeat_num = 10)
    create_env1(Training_path, repeat_num = 10)

Starting_prompt_task_explain = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects in its area. Agents can move a box to a neighboring square or a same-color target. Each square can contain many targets and boxes.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, square[0.5, 0.5]).

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. 

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
