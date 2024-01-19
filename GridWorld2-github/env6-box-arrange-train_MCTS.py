from LLM import *
from prompt_env6 import *
from env6_create import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
#from env6_func import *  #with score model
from env6_func_wo_score_model import * #without score model
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")

Starting_prompt_task_explain = f'''
    You (the robot) are in a grid-like field to pick up all the goals in order and avoid all the obstacles. Each goal and obstacle is assigned to a 1x1 square.

    The robot can move in four directions: up, down, left, and right. The robot can move to a square only if it is not occupied by an obstacle.
    
    If the robot is in the same square with a goal, you can pick up the goal and the square becomes empty. However, you should pick the goals in order, from 0 to larger.
    
    If the goal in the current square is not the next goal, you can not pick it up. You should move to other squares to find the next goal.
    
    [(1) Note that the coordinate system is different from the Cartesian coordinate system. The origin is at the top left corner. The coordinate representation is [row_number, column_number].
    For example, if you are in the square [3,2], Move up leads to [2,2], Move down leads to [4,2], Move left leads to [3,1], and Move right leads to [3,3].
    (2) The robot should pick up all the goals in order, index from 0 to larger. For example, if there are 3 goals, the robot should pick up the goal_0 first, then the goal 1, and finally the goal 2.
    (3) In your response, you can only use {{}} to specify your action. For example, {{Move up}}. Do not add any other words or symbols in your response. Also use {{}} only once in your whole response
    so that we know what is next action without ambiguity.]

    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
    
    Do remember do not move to the square occupied by an obstacle! Do remember do not move out of the field! Plan your action ineach step based on your relative distance to goals.
    
    All the possible actions are: Move up, Move down, Move left, Move right, Pick goal
    
    Specify your action in this format at the end of your answer: {{Move up}}, {{Move down}}, {{Move left}}, {{Move right}}, {{Pick goal}}.

  '''

# 'gpt-3.5-turbo-16k-0613' 'gpt-4-1106-preview' 'gpt-3point5-turbo-0301'
experiment_trial_num = 1; input_error_prompt_token_limit = 15000; model_name_promptLLM = 'gpt-4-1106-preview'; min_level = 2
Testing_path = '/home/ycchen/autoprompt/GridWorld2/test_set/'
base_path = f'/home/ycchen/autoprompt/GridWorld2/train_set/'
dir_path = os.path.join(base_path, f'round1_wo_score_model_gpt-4-1106-preview/')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
save_path = os.path.join(dir_path, f'initial_prompt')

tree = Tree(base_path=save_path, Testing_path = Testing_path, n_children=4, n_selected=2,
            input_error_prompt_token_limit = input_error_prompt_token_limit, model_name_promptLLM = model_name_promptLLM)

# Check if the directory exists
if os.path.exists(save_path):
    # Remove the directory and its contents
    shutil.rmtree(save_path)

# Create the directory
os.makedirs(save_path)

path_each_prompt = os.path.join(save_path, f'prompt_{tree.node_index}')
if not os.path.exists(path_each_prompt):
    os.makedirs(path_each_prompt)
score, feedback_to_promptLLM_list, success_failure_list = tree.score_prompt(Starting_prompt_task_explain, path_each_prompt)

tree.root = Node(Starting_prompt_task_explain, score, 1, tree.node_index, parent=None,
                 feedback_to_promptLLM_list=feedback_to_promptLLM_list, success_failure_list=success_failure_list)

tree.evolve_tree(tree.root, min_level=min_level)
tree.display_tree(tree.root)  # Display the tree for visualization

# Save the tree to a file
tree.save_tree(tree.root, save_path)
tree.save_display_tree(tree.root)
