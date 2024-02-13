# Grid World with multiple goals and obstacles
from prompt_env7 import *
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
import pygame
import sys
import heapq
import math
import pdb
from blocksworld_env.utils import *
from pddlgym.structs import (Type, Predicate, Literal, LiteralConjunction,
                             LiteralDisjunction, Not, Anti, ForAll, Exists,
                             ProbabilisticEffect, TypedEntity, ground_literal,
                             DerivedPredicate, NoChange)
import pddlgym


# This code is to update the current surrounding environment information to LLM, i.e., notice the goals and obstacles on the left, right, up, down for one step
def state_update_func(problem_file):
  config_file = '../BlocksWorld/blocksworld.yaml'
  if not os.path.exists(config_file):
    raise FileNotFoundError(f'Config file {config_file} does not exist.')
  data = read_config(config_file)
  domain_file = '../BlocksWorld/env_data_BlocksWorld/pddlgym/pddl/manyblockssmallpiles.pddl'
  if not os.path.exists(domain_file):
    raise FileNotFoundError(f'Domain file {domain_file} does not exist.')
  problem = get_problem(problem_file, domain_file)
  INIT, GOAL, PLAN, data = instance_to_text(problem, False, data)
  
  state_update_prompt_start_overall_summary = INIT

  return state_update_prompt_start_overall_summary

def action_from_response(path, env, LLM_response, problem_file, domain_file):
  invalid_action = False
  task_finish = False
  env_act_feedback = ''
  ### Your Code Here, update pg_dict_input: location of robot, goals picked up should be deleted
  # If collision, collision_check = True
  # If current location is out of the whole field, moveout_check = True
  # If the robot pick up the goal with wrong order, wrong_order_check = True
  domain = pddlgym.parser.PDDLDomainParser(domain_file, 
            expect_action_preds=False,
            operators_as_actions=True)
  problem = pddlgym.parser.PDDLProblemParser(problem_file, domain.domain_name, 
                domain.types, domain.predicates, domain.actions, domain.constants)
  config_file = '../BlocksWorld/blocksworld.yaml'
  if not os.path.exists(config_file):
    raise FileNotFoundError(f'Config file {config_file} does not exist.')
  data = read_config(config_file)
  try:
    plan, readable_plan = text_to_plan_blocksworld(LLM_response, domain.actions, None, data)
  except:
    invalid_action = True
    return env.get_state(), task_finish, problem_file, invalid_action, env_act_feedback
  previous_state = env.get_state()
  if plan != '':
    if 'stack' in LLM_response or 'unstack' in LLM_response:
      arity = 2
    else:
      arity = 1
    predicate = Predicate(plan[1:-2].split()[0], arity)
    variables = [TypedEntity( v, Type('default')) for v in plan[1:-2].split()[1:]]
    action = Literal(predicate,variables)
    print('Action', action)
    #previous_state = env.get_state()
    try: 
      obs, reward, done, debug_info = env.step(action)
      # pdb.set_trace()
      problem_file = os.path.join(path, f'temp.pddl')
      problem.write(problem_file, initial_state = obs.literals, fast_downward_order=True)
      task_finish = done
    except pddlgym.core.InvalidAction:
      invalid_action = True
      return env.get_state(), task_finish, problem_file, invalid_action, env_act_feedback
  else:
    obs = previous_state
  if previous_state == obs:
    if 'pick' in LLM_response:
      reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. I can only pick up a block if my hand is empty, the block is on the table, and the block is clear.'
    elif 'put' in LLM_response:
      reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. I can only put down a block that I am holding.'
    elif 'stack' in LLM_response:
      reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. I can only stack a block on top of another block if I am holding the block being stacked and the block onto which I am stacking the block is clear.'
    elif 'unstack' in LLM_response:
      reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. I can only unstack a block from on top of another block if my hand is empty, the block I am unstacking was really on top of the other block, and the block I am unstacking is clear.'
    else:
      reason = f'Your assigned action {LLM_response} is not doable'
    
    print('-------noeffect action-------')
    env_act_feedback += f'{reason}'
  # pdb.set_trace()

  print('###########action from response############')

  # if LLM_response not in ['Move up', 'Move down', 'Move left', 'Move right', 'Pick goal']:
  #   env_act_feedback += f'Your assigned action {LLM_response} for the current robot is not in the doable action list.'

  return obs, task_finish, problem_file, invalid_action, env_act_feedback
