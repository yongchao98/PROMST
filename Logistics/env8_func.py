from LLM import *
from prompt_env8 import *
from env8_create import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
import os
import pddlgym
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
from logistics_env.utils import *

class Node:
  def __init__(self, prompt, score, level, index, parent=None, feedback_to_promptLLM_list=None, success_failure_list=None, error_string_list=None):
    self.prompt = prompt
    self.score = score
    self.children = []
    self.level = level
    self.index = index
    self.parent = parent
    self.feedback_to_promptLLM_list = feedback_to_promptLLM_list
    self.success_failure_list = success_failure_list
    self.error_string_list = error_string_list

  def __str__(self):
    return f"Index: {self.index}, Level: {self.level}, Score: {self.score}"

  def add_child(self, child):
    self.children.append(child)

  def to_dict(self):
    return {
      'index': self.index,
      'prompt': self.prompt,
      'score': self.score,
      'level': self.level,
      'parent_index': self.parent.index if self.parent else None,
      'child_indices': [child.index for child in self.children]
    }


class Tree:
  def __init__(self, base_path, Testing_path, n_children=4, n_selected=2, input_error_prompt_token_limit = 15000, model_name_promptLLM = 'gpt-4-1106-preview', model_name_testLLM = 'gpt-3.5-turbo-16k-0613', prompt_method = 'PROMST', with_score_model = 'False'):
    self.node_index = 1
    self.n_children = n_children
    self.n_selected = n_selected
    self.base_path = base_path
    self.Testing_path = Testing_path
    self.input_error_prompt_token_limit = input_error_prompt_token_limit
    self.model_name_promptLLM = model_name_promptLLM
    self.model_name_testLLM = model_name_testLLM
    self.best_score_list_each_level = []
    self.prompt_method = prompt_method
    self.with_score_model = with_score_model

  # more work here
  def score_prompt(self, prompt, path_each_prompt):
    score, feedback_to_promptLLM_list, success_failure_list, error_string_list = total_score_in_training_set(
      prompt, self.Testing_path, path_each_prompt, self.node_index, self.model_name_testLLM)
    print(f'Prompt_{self.node_index} total score in training set: {score}\n\n')
    return score, feedback_to_promptLLM_list, success_failure_list, error_string_list

  # more work here
  def evolve_prompt(self, parent):
    prompt_trajectory_list = self.prompt_trajectory_list_func(parent)
    children_prompt_list = []
    for i in range(self.n_children):
      if self.prompt_method == 'PROMST':
        prompt_task_explain = new_prompt_construct_func_PROMST(prompt_trajectory_list, parent.feedback_to_promptLLM_list,
                                                    parent.success_failure_list, self.input_error_prompt_token_limit,
                                                    self.model_name_promptLLM, self.with_score_model)
      elif self.prompt_method == 'APE':
        prompt_task_explain = new_prompt_construct_func_APE(prompt_trajectory_list, parent.success_failure_list, self.model_name_promptLLM)
      elif self.prompt_method == 'APO':
        prompt_task_explain = new_prompt_construct_func_APO(prompt_trajectory_list, parent.error_string_list, self.input_error_prompt_token_limit, self.model_name_promptLLM)

      if prompt_task_explain == None:
        print('No more prompt can be constructed!')
        return []
      else:
        print(f'New Prompt_{i+1}: ', prompt_task_explain)
        children_prompt_list.append(prompt_task_explain)
    return children_prompt_list

  def build_level(self, parent):
    children_prompt_list = self.evolve_prompt(parent)
    if len(children_prompt_list) == 0:
      return 'finished'
    for child_prompt in children_prompt_list:
      self.node_index += 1
      path_each_prompt = os.path.join(self.base_path, f'prompt_{self.node_index}')
      if not os.path.exists(path_each_prompt):
        os.makedirs(path_each_prompt)

      child_score, feedback_to_promptLLM_list, success_failure_list, error_string_list = self.score_prompt(child_prompt, path_each_prompt)
      child_node = Node(child_prompt, child_score, parent.level + 1, self.node_index, parent, feedback_to_promptLLM_list, success_failure_list, error_string_list)
      parent.add_child(child_node)
    return 'unfinished'

  def evolve_tree(self, current_node, min_level):
    if current_node.level >= min_level and len(self.best_score_list_each_level) >= 3 and self.best_score_list_each_level[-1] < self.best_score_list_each_level[-2] < self.best_score_list_each_level[-3]:
      return

    finish_unfinish_judge = self.build_level(current_node)
    if finish_unfinish_judge == 'finished':
      return
    selected_children = sorted(current_node.children, key=lambda x: x.score, reverse=True)[:self.n_selected]
    self.best_score_list_each_level.append(selected_children[0].score)

    for child in selected_children:
      self.evolve_tree(child, min_level)

  def find_node_by_index(self, node, index):
    if node.index == index:
      return node
    for child in node.children:
      result = self.find_node_by_index(child, index)
      if result:
        return result
    return None

  def create_index_score_dict(self, node, index_score_dict):
    index_score_dict[node.index] = node.score
    for child in node.children:
      self.create_index_score_dict(child, index_score_dict)

  def prompt_trajectory_list_func(self, node):
    prompt_trajectory_list = [node.prompt]
    if node.parent is not None:
      prompt_trajectory_list = self.prompt_trajectory_list_func(node.parent) + prompt_trajectory_list
    return prompt_trajectory_list

  def save_tree(self, node, base_path):
    if not os.path.exists(base_path):
      os.makedirs(base_path)
    path_prompt_together = os.path.join(base_path, 'prompt_json_together')
    if not os.path.exists(path_prompt_together):
        os.makedirs(path_prompt_together)

    path_each_prompt = os.path.join(base_path, f'prompt_{node.index}')
    if not os.path.exists(path_each_prompt):
      os.makedirs(path_each_prompt)

    # Save the node data in its directory
    node_data = node.to_dict()
    with open(os.path.join(path_each_prompt, f'node_{node.index}.json'), 'w') as file:
      json.dump(node_data, file, indent=4)
    file.close()
    with open(os.path.join(path_prompt_together, f'node_{node.index}.json'), 'w') as file:
      json.dump(node_data, file, indent=4)
    file.close()
    # Recursively save each child in the same base directory
    for child in node.children:
      self.save_tree(child, base_path)

  def display_tree(self, node, indent=0):
    print('  ' * indent + str(node))
    for child in node.children:
      self.display_tree(child, indent + 1)

  def save_display_tree(self, node, indent=0):
    tree_file_path = os.path.join(self.base_path, 'tree.txt')
    # Check if the file exists and delete it if it does
    if os.path.exists(tree_file_path):
      os.remove(tree_file_path)
    # Recursive function to write the tree structure to the file
    self._write_tree_to_file(node, indent, tree_file_path)

    score_level_list_path = os.path.join(self.base_path, 'score_level_list.txt')
    with open(score_level_list_path, 'w') as f:
      f.write(str(self.best_score_list_each_level))
    f.close()

  def _write_tree_to_file(self, node, indent, file_path):
    with open(file_path, 'a') as f:
      f.write('  ' * indent + str(node) + '\n')
      for child in node.children:
        self._write_tree_to_file(child, indent + 1, file_path)
    f.close()

  def load_tree_from_files(self, result_path):
    # Load all node files
    node_files = [f for f in os.listdir(result_path) if f.endswith('.json')]
    print(f'Found {len(node_files)} node files')

    # Create a dictionary to hold the nodes
    nodes = {}

    # First pass: create all nodes without setting up parent-child relationships
    for file_name in node_files:
      with open(os.path.join(result_path, file_name), 'r') as file:
        node_data = json.load(file)
        node = Node(node_data['prompt'], node_data['score'], node_data['level'], node_data['index'])
        nodes[node.index] = node
      file.close()

    # Second pass: set up parent-child relationships
    for file_name in node_files:
      with open(os.path.join(result_path, file_name), 'r') as file:
        node_data = json.load(file)
        node = nodes[node_data['index']]
        if node_data['parent_index'] is not None:
          parent = nodes[node_data['parent_index']]
          node.parent = parent
          parent.add_child(node)
      file.close()

    # Sort the children of each node by their index
    for node in nodes.values():
      if hasattr(node, 'children'):
        node.children.sort(key=lambda child: child.index)

    # Find the root (node without a parent)
    root = next(node for node in nodes.values() if node.parent is None)
    return root

def check_num_goal_finished(goal, obs):
  goal = str(goal)
  num = 0
  if goal.startswith('AND'):
    goal = goal[4:-1]
    goal = goal.split(', ')
  for literal in list(obs.literals):
    if str(literal) in goal:
      num += 1
  return num, len(goal)

def score_in_training_set(Testing_path, prompt_task_explain, query_time_limit, index, model_name_testLLM):
  ### Your code here
  ### Read the dictionary file/environment, can be json or not, depending on the instance
  print('Looking at index', index)
  env = pddlgym.make("PDDLEnvManylogistics-v0")
  env.fix_problem_index(index)
  obs, debug_info = env.reset()
  init_obs = obs

  #print(f'prompt_task_explain: ', prompt_task_explain)
  rear_prompt_list = []  # The record list of all the rear prompts
  response_total_list = []  # The record list of all the responses
  env_act_feedback_list = []  # The record list of env act feedbacks
  dict_not_update_rounds = 0
  all_response_total_list = [] # The record list of every part of responses
  obs_list = [] # The record list of pg states in varied steps
  success_failure = ''

  ### Start the Game! Query LLM for response
  print(f'query_time_limit: {query_time_limit}')
  config_file = '../Logistics/logistics.yaml'
  data = read_config(config_file)
  problem_file, domain_file = debug_info['problem_file'], debug_info['domain_file']
  problem = get_problem(problem_file, domain_file)
  state_update_prompt, GOAL, PLAN, data = instance_to_text(problem, False, data)
  obs_list.append(state_update_prompt)

  for index_query_times in range(query_time_limit): # The upper limit of calling LLMs
    print(f'index_query_times_{index_query_times}:\n{state_update_prompt}')
    user_prompt_1 = input_prompt_1_func_total(prompt_task_explain, state_update_prompt, GOAL, response_total_list,
                              obs_list, env_act_feedback_list)
    rear_prompt_list.append(user_prompt_1)

    prompt_total = prompt_task_explain + user_prompt_1
    print('prompt: ', prompt_total)
    messages = message_construct_func([prompt_total], []) # message construction

    initial_response = GPT_response(messages, model_name_testLLM)
    print('initial response: ', initial_response)
    all_response_total_list.append(initial_response)

    success_failure = ''
    task_finish = 0
    initial_response.replace('{', '')
    initial_response.replace('}', '')
    start = initial_response.lower().split()[0]
    # start2 = initial_response.split('.')[-2]
    if start != 'load' and start != 'unload' and start != 'drive' and start != 'fly':
      success_failure = 'response in the wrong format'
    # elif start == 'load' or start != 'unload' or start != 'drive' or start != 'fly':
    else:
      if initial_response[-1]=='.':
        response = initial_response[:-1].lower()
      else:
        response = initial_response.lower()

    response_total_list.append(response)
    if success_failure == 'response in the wrong format':
      env_act_feedback_list.append('')
      print('system_error_feedback: ', success_failure)
      # if match:
      #   feedback_to_promptLLM = f'Here is the prompt of task description:\n{prompt_task_explain}. The response {response} is in the wrong format.'
      # else:
      feedback_to_promptLLM = f'Here is the prompt of task description:\n{prompt_task_explain}. The response {all_response_total_list[-1]} is in the wrong format.'
      break

    elif success_failure == '':
      ####Your code here, modify the action_from_response function
      obs, task_finish, problem_file, invalid_action, wrong_object, env_act_feedback = action_from_response(Testing_path, env, response, problem_file, domain_file)
      if invalid_action:
        print('invalid action')
        success_failure = 'Invalid action'
        feedback_to_promptLLM = f'Your response is: {response}\n\nThe action in the response is invalid action.'
        break
      if wrong_object:
        print('wrong object action')
        success_failure = 'wrong object action'
        feedback_to_promptLLM = f'Your response is: {response}\n\nThe action in the response is trying to refer to nonspecific or nonexist vehicle (truck, airplane).'
        break
      num_goal_finished_previous_step, _ = check_num_goal_finished(debug_info['goal'], init_obs)

      num_goal_finished_current_step, _  = check_num_goal_finished(debug_info['goal'], obs)

      if num_goal_finished_previous_step == num_goal_finished_current_step:
        dict_not_update_rounds += 1
      else:
        dict_not_update_rounds = 0
      print('#'*20)
      print('dict_not_update_rounds: ', dict_not_update_rounds)
      print('num_goal_finished_previous_step: ', num_goal_finished_previous_step)
      print('num_goal_finished_current_step: ', num_goal_finished_current_step)
      if dict_not_update_rounds > 15:
        success_failure = 'Stuck in the local loop.'
        system_error_feedback_2 = 'It seems the LLM is stuck in the current situation, always repeating the same answer. The task is stuck too, no box is placed successfully in recent rounds.'
        feedback_to_promptLLM = feedback_to_promptLLM_func(rear_prompt_list[-2],
                                                                                   response_total_list[-2],
                                                                                   env_act_feedback_list[-2],
                                                                                   rear_prompt_list[-1],
                                                                                   response_total_list[-1],
                                                                                   env_act_feedback_list[-1],
                                                                                   error_feedback=system_error_feedback_2)
        break
      state_update_prompt = state_update_func(problem_file)
      obs_list.append(state_update_prompt)
      env_act_feedback_list.append(env_act_feedback)

      if env_act_feedback != '':
        print('env_act_feedback: ', env_act_feedback)

      ####Your code here
      # Check whether the task has been completed
      if task_finish == 1:
        break

  if task_finish == 1:
    success_failure = 'success'
    feedback_to_promptLLM = 'The task is completed successfully.'
  elif success_failure == '':
    success_failure = 'failure over query time limit'
    system_error_feedback_3 = 'The task is not completed over the query time limit.'
    feedback_to_promptLLM = feedback_to_promptLLM_func(rear_prompt_list[-2],
                                                       response_total_list[-2],
                                                       env_act_feedback_list[-2],
                                                       rear_prompt_list[-1],
                                                       response_total_list[-1],
                                                       env_act_feedback_list[-1],
                                                       error_feedback=system_error_feedback_3)

  error_string = ''
  if success_failure != 'success':
    if len(rear_prompt_list) == 1:
      error_string = error_string_func_APO(rear_prompt_list[-1], response_total_list[-1])
    else:
      try:
        error_string = error_string_func_APO(rear_prompt_list[-2], response_total_list[-2], env_act_feedback_list[-2],
                                          rear_prompt_list[-1], response_total_list[-1])
      except:
        print('Length of rear_prompt_list: ', len(rear_prompt_list))
        print('Length of response_total_list: ', len(response_total_list))
        print('Length of env_act_feedback_list: ', len(env_act_feedback_list))
        raise error

  return init_obs, obs, success_failure, index_query_times, feedback_to_promptLLM, debug_info['goal'], error_string


def total_score_in_training_set(prompt_task_explain, Testing_path, Saving_path_result, prompt_num, model_name_testLLM):
  success_failure_list = []
  picked_goal_ratio_list = []
  feedback_to_promptLLM_list = []
  index_query_times_list = []
  error_string_list = []

  print(f'-------------------Model name: {model_name_testLLM}-------------------')
  query_time_limit = 20
  for index in range(12):
    print('-------###-------###-------###-------')
    init_obs, obs, success_failure, index_query_times, feedback_to_promptLLM, goal, error_string\
      = score_in_training_set(Saving_path_result, prompt_task_explain, query_time_limit, index, model_name_testLLM)
    
    num_goal_finished, num_goal = check_num_goal_finished(goal, obs)

    picked_goal_ratio = (num_goal_finished) / num_goal
    print('Picked_goal_ratio in this trial: ', picked_goal_ratio)
    picked_goal_ratio_list.append(picked_goal_ratio)
    success_failure_list.append(success_failure)
    feedback_to_promptLLM_list.append(feedback_to_promptLLM)
    index_query_times_list.append(index_query_times)
    if error_string != '':
      error_string_list.append(error_string)

    # pdb.set_trace()
    with open(Saving_path_result + f'/pg_dict_initial_{index}.json', 'w') as f:
      json.dump(str(init_obs), f)
    f.close()

    with open(Saving_path_result + f'/remaining_box_dict_{index}.json', 'w') as f:
      json.dump(str(obs), f)
    f.close()

    with open(Saving_path_result + f'/success_failure_{index}.txt', 'w') as f:
      f.write(success_failure)
    f.close()

    with open(Saving_path_result + f'/feedback_to_promptLLM_{index}.txt', 'w') as f:
      f.write(feedback_to_promptLLM)
    f.close()

    with open(Saving_path_result + f'/env_action_times_{index}.txt', 'w') as f:
      f.write(f'{index_query_times+1}')
    f.close()
    print(success_failure)
    print(f'Iteration number: {index_query_times+1}')

  print(f'\n\nPrompt_{prompt_num} Picked goal ratio ratio: {np.mean(picked_goal_ratio_list)}\n\n')
  with open(Saving_path_result + f'/total_score.txt', 'w') as f:
    f.write(f'{np.mean(picked_goal_ratio_list)}')
  f.close()
  with open(Saving_path_result + f'/prompt.txt', 'w') as f:
    f.write(f'{prompt_task_explain}')
  f.close()
  return np.mean(picked_goal_ratio_list), feedback_to_promptLLM_list, success_failure_list, error_string_list

def new_prompt_construct_func_PROMST(prompt_trajectory_list, feedback_to_promptLLM_list, success_failure_list, input_error_prompt_token_limit, model_name_promptLLM, with_score_model):
  prompt_task_explain = prompt_trajectory_list[-1]
  trajectory_prompts = ''
  for prompt_level_index, prompt_item in enumerate(prompt_trajectory_list):
    trajectory_prompts += f'{prompt_level_index+1}: {prompt_item}\n'

  error_string_query_limit = ''
  error_string_format = ''
  error_string_loop = ''
  error_string_invalid = ''
  error_string_noeffect = ''

  my_list = [index for index in range(len(success_failure_list))]
  random_index_list = random.sample(my_list, k=len(success_failure_list))

  for index in random_index_list:
    success_failure = success_failure_list[index]
    print(f'\n\nindex: {index}, success_failure: {success_failure}')
    if success_failure == 'failure over query time limit':
      if len(enc.encode(error_string_query_limit)) +  len(
              enc.encode(feedback_to_promptLLM_list[index]))< input_error_prompt_token_limit:
        error_string_query_limit += f'{feedback_to_promptLLM_list[index]}\n'
    elif success_failure == 'response in the wrong format':
      if len(enc.encode(error_string_format)) + len(
              enc.encode(feedback_to_promptLLM_list[index])) < input_error_prompt_token_limit:
        error_string_format += f'{feedback_to_promptLLM_list[index]}\n'
    elif success_failure == 'Invalid action':
      if len(enc.encode(error_string_invalid)) + len(
              enc.encode(feedback_to_promptLLM_list[index])) < input_error_prompt_token_limit:
        error_string_invalid += f'{feedback_to_promptLLM_list[index]}\n'
    elif success_failure == 'No effect action':
      if len(enc.encode(error_string_noeffect)) + len(
              enc.encode(feedback_to_promptLLM_list[index])) < input_error_prompt_token_limit:
        error_string_noeffect += f'{feedback_to_promptLLM_list[index]}\n'
    elif success_failure == 'Stuck in the local loop.':
      print('Token number: ', len(enc.encode(error_string_loop)) + len(enc.encode(feedback_to_promptLLM_list[index])))
      if len(enc.encode(error_string_loop)) + len(
              enc.encode(feedback_to_promptLLM_list[index])) < input_error_prompt_token_limit:
        error_string_loop += f'{feedback_to_promptLLM_list[index]}\n'

  error_feedback = ''; count_error_type = 0
  if error_string_query_limit != '':
    prompt_to_error_summary = prompt_to_error_summarizer_PROMST(prompt_task_explain, error_string_query_limit)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_query_limit_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_query_limit_summarized: {error_string_query_limit_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_query_limit_summarized

  if error_string_format != '':
    prompt_to_error_summary = prompt_to_error_summarizer_PROMST(prompt_task_explain, error_string_format)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_format_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_format_summarized: {error_string_format_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_format_summarized
  
  if error_string_invalid != '':
    prompt_to_error_summary = prompt_to_error_summarizer_PROMST(prompt_task_explain, error_string_invalid)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_collision_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_collision_summarized: {error_string_collision_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_collision_summarized

  if error_string_loop != '':
    prompt_to_error_summary = prompt_to_error_summarizer_PROMST(prompt_task_explain, error_string_loop)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_loop_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_loop_summarized: {error_string_loop_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_loop_summarized

  if error_string_noeffect != '':
    prompt_to_error_summary = prompt_to_error_summarizer_PROMST(prompt_task_explain, error_string_noeffect)
    messages = message_construct_func([prompt_to_error_summary], [])
    error_string_noeffect_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_noeffect_summarized: {error_string_noeffect_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_noeffect_summarized

  if error_feedback != '':
    prompt_to_promptLLM = prompt_to_promptLLM_func_total_with_env_act_feedback_PROMST_APO(prompt_task_explain, error_feedback, trajectory_prompts)
    messages = message_construct_func([prompt_to_promptLLM], [])  # message construction
    prompt_task_explain = GPT_response(messages, model_name_promptLLM)
    if with_score_model == 'False':
      pass
    elif with_score_model == 'True':
      ### Code needed to be added here###


      pass
    else:
      raise ValueError('with_score_model should be either True or False.')
    print(f'New prompt_task_explain: {prompt_task_explain}')
    return prompt_task_explain
  else:
    print(f'No error feedback, no need to update prompt_task_explain.')
    return None

def new_prompt_construct_func_APO(prompt_trajectory_list, error_string_list, input_error_prompt_token_limit, model_name_promptLLM):
  trajectory_prompts = ''
  for prompt_level_index, prompt_item in enumerate(prompt_trajectory_list):
    trajectory_prompts += f'{prompt_level_index+1}: {prompt_item}\n'

  my_list = [index for index in range(len(error_string_list))]
  random_index_list = random.sample(my_list, k=len(error_string_list))

  if len(error_string_list) != 0:
    error_string = ''
    index_count = 0
    for index in random_index_list:
      if len(enc.encode(error_string)) + len(
            enc.encode(error_string_list[index])) < input_error_prompt_token_limit:
        error_string += f'Example {index_count}: {error_string_list[index]}\n\n'
        index_count += 1

    prompt_task_explain = copy.deepcopy(prompt_trajectory_list[-1])
    error_string_total = prompt_to_self_error_summarizer_APO(prompt_task_explain, error_string)
    messages = message_construct_func([error_string_total], [])  # message construction
    error_feedback = GPT_response(messages, model_name_promptLLM)
    prompt_to_promptLLM = prompt_to_promptLLM_func_total_with_env_act_feedback_PROMST_APO(prompt_task_explain,
                                                                                          error_feedback,
                                                                                          trajectory_prompts)

    messages = message_construct_func([prompt_to_promptLLM], [])  # message construction
    prompt_task_explain = GPT_response(messages, model_name_promptLLM)
    print(f'New prompt_task_explain: {prompt_task_explain}')
    return prompt_task_explain
  else:
    print(f'No error feedback, no need to update prompt_task_explain.')
    return None

def new_prompt_construct_func_APE(prompt_trajectory_list, success_failure_list, model_name_promptLLM):
  prompt_task_explain = copy.deepcopy(prompt_trajectory_list[-1])
  if 'failure over query time limit' in success_failure_list or 'response in the wrong format' in success_failure_list or 'Stuck in the local loop.' in success_failure_list:
    prompt_to_promptLLM = prompt_to_promptLLM_func_total_APE(prompt_task_explain)
    messages = message_construct_func([prompt_to_promptLLM], [])  # message construction
    prompt_task_explain = GPT_response(messages, model_name_promptLLM)
    print(f'New prompt_task_explain: {prompt_task_explain}')
    return prompt_task_explain
  else:
    print(f'No error feedback, no need to update prompt_task_explain.')
    return None