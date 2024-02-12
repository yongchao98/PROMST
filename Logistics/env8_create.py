# Grid World with multiple goals and obstacles
from prompt_env8 import *
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
from logistics_env.utils import *
from pddlgym.structs import (Type, Predicate, Literal, LiteralConjunction,
                             LiteralDisjunction, Not, Anti, ForAll, Exists,
                             ProbabilisticEffect, TypedEntity, ground_literal,
                             DerivedPredicate, NoChange)
import pddlgym


# This code is to update the current surrounding environment information to LLM, i.e., notice the goals and obstacles on the left, right, up, down for one step
def state_update_func(problem_file):
  config_file = '../Logistics/logistics.yaml'
  data = read_config(config_file)
  domain_file = '../Logistics/env_data_Logistics/pddlgym/pddl/manylogistics.pddl'
  problem = get_problem(problem_file, domain_file)
  INIT, GOAL, PLAN, data = instance_to_text(problem, False, data)
  
  state_update_prompt_start_overall_summary = INIT

  return state_update_prompt_start_overall_summary

def action_from_response(path, env, LLM_response, problem_file, domain_file):
  invalid_action = False
  noeffect_action = False
  task_finish = False
  wrong_object = False
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
  try:
    plan, readable_plan = text_to_plan_logistics(LLM_response, domain.actions, None)
  except ValueError:
    wrong_object = True
    return env.get_state(), task_finish, problem_file, invalid_action, wrong_object, env_act_feedback
  # pdb.set_trace()
  if plan != '':
    if 'drive' in LLM_response:
      arity = 4
    else:
      arity = 3
    predicate = Predicate(plan[1:-2].split()[0], arity)
    variables = [TypedEntity( v, Type('default')) for v in plan[1:-2].split()[1:]]
    action = Literal(predicate,variables)
  print('Action', action)
  previous_state = env.get_state()
  try: 
    obs, reward, done, debug_info = env.step(action)
    # pdb.set_trace()
    problem_file = os.path.join(path, f'temp.pddl')
    problem.write(problem_file, initial_state = obs.literals, fast_downward_order=True)
    task_finish = done
  except pddlgym.core.InvalidAction:
    invalid_action = True
    return env.get_state(), task_finish, problem_file, invalid_action, wrong_object, env_act_feedback
  if previous_state == obs:
    if 'load' in LLM_response and 'truck' in LLM_response:
      if 'city' in LLM_response:
        reason = f'Your assigned action {LLM_response} is not doable because inputs are not correct. The inputs to load action is object, vehicle(truck or airplane), and location.'
      else:
        reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. A package can be loaded into a truck only if the package and the truck are in the same specified location.'
    elif 'load' in LLM_response and 'airplane' in LLM_response:
      if 'city' in LLM_response:
        reason = f'Your assigned action {LLM_response} is not doable because inputs are not correct. The inputs to load action is object, vehicle(truck or airplane), and location.'
      else:
        reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. A package can be loaded into an airplane only if the package and the airplane are in the same specified location.'
    elif 'unload' in LLM_response and 'truck' in LLM_response:
      if 'city' in LLM_response:
        reason = f'Your assigned action {LLM_response} is not doable because inputs are not correct. The inputs to unload action is object, vehicle(truck or airplane), and location.'
      else:
        reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. A package can be unloaded from a truck only if the package is in the truck and the truck is at the specified location.'
    elif 'unload' in LLM_response and 'airplane' in LLM_response:
      if 'city' in LLM_response:
        reason = f'Your assigned action {LLM_response} is not doable because inputs are not correct. The inputs to unload action is object, vehicle(truck or airplane), and location.'
      else:
        reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. A package can be unloaded from an airplane only if the package in the airplane and the airplane is at the specified location.'
    elif 'drive' in LLM_response and 'truck' in LLM_response:
      reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. A truck can be driven from one location to another if the truck is at the from-location and both from-location and to-location are locations in the same specified city.'
    elif 'fly' in LLM_response and 'airplane' in LLM_response:
      if 'city' in LLM_response:
        reason = f'Your assigned action {LLM_response} is not doable because inputs are not correct. The inputs to fly action is airplane, from-location, and to-location.'
      else:
        reason = f'Your assigned action {LLM_response} is not doable because preconditions are not meet. An airplane can be flown from one city to another if the from-location and the to-location are airports and the airplane is at the from-location.'
    else:
      reason = f'Your assigned action {LLM_response} is not doable'
    print('-------noeffect action-------')
    env_act_feedback += f'{reason}'
  # pdb.set_trace()

  print('###########action from response############')

  # if LLM_response not in ['Move up', 'Move down', 'Move left', 'Move right', 'Pick goal']:
  #   env_act_feedback += f'Your assigned action {LLM_response} for the current robot is not in the doable action list.'

  if invalid_action:
    print('-------invalid action-------')
    env_act_feedback += f'Your assigned action {LLM_response} is not a valid action in current situation'
  # if noeffect_action:
  #   print('-------noeffect action-------')
  #   env_act_feedback += f'Your assigned action {LLM_response} is not doable because preconditions are not meet, '

  return obs, task_finish, problem_file, invalid_action, wrong_object, env_act_feedback
