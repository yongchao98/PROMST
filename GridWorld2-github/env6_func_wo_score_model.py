from LLM import *
from prompt_env6 import *
from env6_create import *
#from score_model_infer import evaluate_single_prompt
from sre_constants import error
from collections import Counter
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
import os
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")

class Node:
  def __init__(self, prompt, score, level, index, parent=None, feedback_to_promptLLM_list=None, success_failure_list=None):
    self.prompt = prompt
    self.score = score
    self.children = []
    self.level = level
    self.index = index
    self.parent = parent
    self.feedback_to_promptLLM_list = feedback_to_promptLLM_list
    self.success_failure_list = success_failure_list

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
  def __init__(self, base_path, Testing_path, n_children=4, n_selected=2, input_error_prompt_token_limit = 15000, model_name_promptLLM = 'gpt-4-1106-preview'):
    self.node_index = 1
    self.n_children = n_children
    self.n_selected = n_selected
    self.base_path = base_path
    self.Testing_path = Testing_path
    self.input_error_prompt_token_limit = input_error_prompt_token_limit
    self.model_name_promptLLM = model_name_promptLLM
    self.best_score_list_each_level = []

  # more work here
  def score_prompt(self, prompt, path_each_prompt):
    score, feedback_to_promptLLM_list, success_failure_list = total_score_in_training_set(
      prompt, self.Testing_path, path_each_prompt, self.node_index)
    print(f'Prompt_{self.node_index} total score in training set: {score}\n\n')
    return score, feedback_to_promptLLM_list, success_failure_list

  # more work here
  def evolve_prompt(self, parent):
    prompt_trajectory_list = self.prompt_trajectory_list_func(parent)
    children_prompt_list = []
    for i in range(self.n_children):
      prompt_task_explain = new_prompt_construct_func(prompt_trajectory_list, parent.feedback_to_promptLLM_list,
                                                    parent.success_failure_list, self.input_error_prompt_token_limit,
                                                    self.model_name_promptLLM)
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

      child_score, feedback_to_promptLLM_list, success_failure_list = self.score_prompt(child_prompt, path_each_prompt)
      child_node = Node(child_prompt, child_score, parent.level + 1, self.node_index, parent, feedback_to_promptLLM_list, success_failure_list)
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


def score_in_training_set(prompt_task_explain, Saving_path, query_time_limit, pg_row_num, pg_column_num, goal_num, obs_num, iteration_num, model_name = 'gpt-3', order_required = False):

  with open(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}_{goal_num}_{obs_num}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'r') as file:
    pg_dict = json.load(file)
  file.close()

  pg_dict_initial = copy.deepcopy(pg_dict)
  env = GridWorld(pg_row_num, pg_column_num, input_board=pg_dict)

  #print(f'prompt_task_explain: ', prompt_task_explain)
  rear_prompt_list = []  # The record list of all the rear prompts
  response_total_list = []  # The record list of all the responses
  env_act_feedback_list = []  # The record list of env act feedbacks
  dict_not_update_rounds = 0
  all_response_total_list = [] # The record list of every part of responses
  pg_state_list = [] # The record list of pg states in varied steps
  pg_state_list.append(pg_dict)
  success_failure = ''

  ### Start the Game! Query LLM for response
  print(f'query_time_limit: {query_time_limit}')
  state_update_prompt = state_update_func(pg_row_num, pg_column_num, pg_dict)

  for index_query_times in range(query_time_limit): # The upper limit of calling LLMs
    print(f'index_query_times_{index_query_times}:\n{state_update_prompt}')
    user_prompt_1 = input_prompt_1_func_total(prompt_task_explain, state_update_prompt, response_total_list,
                              pg_state_list, env_act_feedback_list)
    rear_prompt_list.append(user_prompt_1)

    prompt_total = prompt_task_explain + user_prompt_1
    messages = message_construct_func([prompt_total], []) # message construction

    initial_response = GPT_response(messages, model_name)
    all_response_total_list.append(initial_response)

    match = re.search(r'{.*}', initial_response, re.DOTALL)
    success_failure = ''
    if match:
      response = match.group()
      if response[0] == '{' and response[-1] == '}':
        if '{' in response[1:-1] and '}' in response[1:-1]:
          match = re.search(r'{.*}', response[:-1], re.DOTALL)
          if match:
            response = match.group()
        print(f'response: {response}\n')
      else:
        success_failure = 'response in the wrong format'
      if response[1:-1] not in ['Move up', 'Move down', 'Move left', 'Move right', 'Pick goal']:
        success_failure = 'response in the wrong format'
    else:
      success_failure = 'response in the wrong format'

    if success_failure == 'response in the wrong format':
      print('system_error_feedback: ', success_failure)
      if match:
        feedback_to_promptLLM = f'Here is the prompt of task description:\n{prompt_task_explain}. The response {response} is in the wrong format.'
      else:
        feedback_to_promptLLM = f'Here is the prompt of task description:\n{prompt_task_explain}. The response {all_response_total_list[-1]} is in the wrong format.'
      break

    elif success_failure == '':
      response_total_list.append(response)
      #state_update_prompt = state_update_func(pg_row_num, pg_column_num, pg_dict)
      #print('Current state: ', pg_dict)

      pg_dict_returned, collision_check, moveout_check, wrong_order_check, env_act_feedback = action_from_response(env, response[1:-1], order_required)
      if collision_check:
        print('Collision!')
        success_failure = 'Collision'
        feedback_to_promptLLM = f'Your response is: {response}\n\nThe action in the response leads to collision.'
        break
      elif moveout_check:
        print('Move out of the grid!')
        success_failure = 'Move out of the grid'
        feedback_to_promptLLM = f'Your response is: {response}\n\nThe action in the response leads to move out of the grid.'
        break
      elif wrong_order_check:
        print('Wrong picking up order!')
        success_failure = 'Wrong picking up order'
        feedback_to_promptLLM = f'Your response is: {response}\n\nThe action in the response leads to wrong order of picking up goals. Do remember pick up goal_0 first, then goal_1, goal_2, and so on.'
        break

      box_count_previous_step = 0
      for value in pg_dict.values():
        if any("box" in item for item in value if isinstance(item, list)):
          box_count_previous_step += 1

      box_count_current_step = 0
      for value in pg_dict_returned.values():
        if any("box" in item for item in value if isinstance(item, list)):
          box_count_current_step += 1

      if box_count_previous_step == box_count_current_step:
        dict_not_update_rounds += 1
      else:
        dict_not_update_rounds = 0
      print('#'*20)
      print('dict_not_update_rounds: ', dict_not_update_rounds)
      print('box_count_previous_step: ', box_count_previous_step)
      print('box_count_current_step: ', box_count_current_step)
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
      pg_dict = copy.deepcopy(pg_dict_returned)
      state_update_prompt = state_update_func(pg_row_num, pg_column_num, pg_dict)
      pg_state_list.append(pg_dict)
      env_act_feedback_list.append(env_act_feedback)

      if env_act_feedback != '':
        print('env_act_feedback: ', env_act_feedback)

      # Check whether the task has been completed
      if box_count_current_step == 0:
        break

  box_count_current_step = 0
  for value in pg_dict.values():
    if any("box" in item for item in value if isinstance(item, list)):
      box_count_current_step += 1
  if box_count_current_step == 0:
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

  remaining_box_dict = copy.deepcopy(pg_dict)

  return pg_dict_initial, remaining_box_dict, success_failure, index_query_times, feedback_to_promptLLM


def total_score_in_training_set(prompt_task_explain, Testing_path, Saving_path_result, prompt_num):
  model_name = 'gpt-4-1106-preview'  # gpt-4-1106-preview, gpt-3.5-turbo-16k-0613
  #model_name = 'gpt-3.5-turbo-16k-0613'
  #model_name = 'gpt-3.5-turbo-0301'
  success_failure_list = []
  picked_goal_ratio_list = []
  feedback_to_promptLLM_list = []
  index_query_times_list = []

  print(f'-------------------Model name: {model_name}-------------------')
  for pg_row_num, pg_column_num, goal_num, obs_num in [(4,4,3,4), (5,5,4,6), (6,6,5,10), (7,7,6,14)]:
    if pg_row_num == 7 or pg_row_num == 8:
      query_time_limit = 50
    else:
      query_time_limit = 30
    #for iteration_num in range(10):
    for iteration_num in range(4):
      print('-------###-------###-------###-------')
      print(f'Row num is: {pg_row_num}, Column num is: {pg_column_num}, Iteration num is: {iteration_num}\n\n')

      pg_dict_initial, remaining_box_dict, success_failure, index_query_times, feedback_to_promptLLM\
        = score_in_training_set(prompt_task_explain, Testing_path, query_time_limit, pg_row_num, pg_column_num, goal_num, obs_num, iteration_num, model_name = model_name, order_required = True)

      box_count_all = 0
      for value in pg_dict_initial.values():
        if any("box" in item for item in value if isinstance(item, list)):
          box_count_all += 1

      box_count_remaining = 0
      for value in remaining_box_dict.values():
        if any("box" in item for item in value if isinstance(item, list)):
          box_count_remaining += 1

      picked_goal_ratio = (box_count_all - box_count_remaining) / box_count_all
      print('Picked_goal_ratio in this trial: ', picked_goal_ratio)
      picked_goal_ratio_list.append(picked_goal_ratio)
      success_failure_list.append(success_failure)
      feedback_to_promptLLM_list.append(feedback_to_promptLLM)
      index_query_times_list.append(index_query_times)

      with open(Saving_path_result + f'/pg_dict_initial_{pg_row_num}_{pg_column_num}_{iteration_num}.json', 'w') as f:
        json.dump(pg_dict_initial, f)
      f.close()

      with open(Saving_path_result + f'/remaining_box_dict_{pg_row_num}_{pg_column_num}_{iteration_num}.json', 'w') as f:
        json.dump(remaining_box_dict, f)
      f.close()

      with open(Saving_path_result + f'/success_failure_{pg_row_num}_{pg_column_num}_{iteration_num}.txt', 'w') as f:
        f.write(success_failure)
      f.close()

      with open(Saving_path_result + f'/feedback_to_promptLLM_{pg_row_num}_{pg_column_num}_{iteration_num}.txt', 'w') as f:
        f.write(feedback_to_promptLLM)
      f.close()

      with open(Saving_path_result + f'/env_action_times_{pg_row_num}_{pg_column_num}_{iteration_num}.txt', 'w') as f:
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
  return np.mean(picked_goal_ratio_list), feedback_to_promptLLM_list, success_failure_list

def new_prompt_construct_func(prompt_trajectory_list, feedback_to_promptLLM_list, success_failure_list, input_error_prompt_token_limit, model_name_promptLLM):
  prompt_task_explain = prompt_trajectory_list[-1]
  trajectory_prompts = ''
  for prompt_level_index, prompt_item in enumerate(prompt_trajectory_list):
    trajectory_prompts += f'{prompt_level_index+1}: {prompt_item}\n'

  error_string_query_limit = ''
  error_string_format = ''
  error_string_loop = ''
  error_string_collision = ''
  error_string_moveout = ''
  error_string_wrong_order = ''

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
    elif success_failure == 'collision':
      if len(enc.encode(error_string_collision)) + len(
              enc.encode(feedback_to_promptLLM_list[index])) < input_error_prompt_token_limit:
        error_string_collision += f'{feedback_to_promptLLM_list[index]}\n'
    elif success_failure == 'Move out of the grid':
      if len(enc.encode(error_string_moveout)) + len(
              enc.encode(feedback_to_promptLLM_list[index])) < input_error_prompt_token_limit:
          error_string_moveout += f'{feedback_to_promptLLM_list[index]}\n'
    elif success_failure == 'Wrong picking up order':
      if len(enc.encode(error_string_wrong_order)) + len(
              enc.encode(feedback_to_promptLLM_list[index])) < input_error_prompt_token_limit:
          error_string_wrong_order += f'{feedback_to_promptLLM_list[index]}\n'
    elif success_failure == 'Stuck in the local loop.':
      print('Token number: ', len(enc.encode(error_string_loop)) + len(enc.encode(feedback_to_promptLLM_list[index])))
      if len(enc.encode(error_string_loop)) + len(
              enc.encode(feedback_to_promptLLM_list[index])) < input_error_prompt_token_limit:
        error_string_loop += f'{feedback_to_promptLLM_list[index]}\n'

  error_feedback = ''; count_error_type = 0
  if error_string_query_limit != '':
    prompt_to_error_summary = prompt_to_error_summarizer(prompt_task_explain, error_string_query_limit)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_query_limit_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_query_limit_summarized: {error_string_query_limit_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_query_limit_summarized

  if error_string_format != '':
    prompt_to_error_summary = prompt_to_error_summarizer(prompt_task_explain, error_string_format)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_format_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_format_summarized: {error_string_format_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_format_summarized

  if error_string_collision != '':
    prompt_to_error_summary = prompt_to_error_summarizer(prompt_task_explain, error_string_collision)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_collision_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_collision_summarized: {error_string_collision_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_collision_summarized

  if error_string_moveout != '':
    prompt_to_error_summary = prompt_to_error_summarizer(prompt_task_explain, error_string_moveout)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_moveout_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_moveout_summarized: {error_string_moveout_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_moveout_summarized

  if error_string_wrong_order != '':
    prompt_to_error_summary = prompt_to_error_summarizer(prompt_task_explain, error_string_wrong_order)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_wrong_order_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_wrong_order_summarized: {error_string_wrong_order_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_wrong_order_summarized

  if error_string_loop != '':
    prompt_to_error_summary = prompt_to_error_summarizer(prompt_task_explain, error_string_loop)
    messages = message_construct_func([prompt_to_error_summary], [])  # message construction
    error_string_loop_summarized = GPT_response(messages, model_name_promptLLM)
    #print(f'error_string_loop_summarized: {error_string_loop_summarized}')
    count_error_type += 1
    error_feedback += f'Error type {count_error_type}: \n' + error_string_loop_summarized

  if error_feedback != '':
    count_reprompt_time = 0
    prompt_to_promptLLM = prompt_to_promptLLM_func_total_with_env_act_feedback_MCTS(prompt_task_explain, error_feedback, trajectory_prompts)
    messages = message_construct_func([prompt_to_promptLLM], [])  # message construction
    prompt_task_explain = GPT_response(messages, model_name_promptLLM)
    print(f'New prompt_task_explain: {prompt_task_explain}')
    return prompt_task_explain
  else:
    #print(f'No error feedback, no need to update prompt_task_explain.')
    return None
