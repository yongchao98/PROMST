# Grid World with multiple goals and obstacles
from prompt_env6 import *
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

class GridWorld:
    def __init__(self, dimension, num_obstacles=5, num_goals=5, input_board = False):
        if not input_board:
            self.dimension = dimension
            self.num_obstacles = num_obstacles
            self.num_goals = num_goals
            self.grid = [['empty' for _ in range(dimension)] for _ in range(dimension)]

            # Initialize Pygame
            #pygame.init()
            self.cell_size = 50
            self.screen_size = (dimension * self.cell_size, dimension * self.cell_size)
            #self.screen = pygame.display.set_mode(self.screen_size)
            #pygame.display.set_caption('GridWorld')

            # Colors
            self.colors = {
                'empty': (255, 255, 255),
                'agent': (0, 128, 255),
                'obstacle': (0, 0, 0),
                'goal': (255, 0, 0),
                'visited_goal': (0, 255, 0),
                'label': (255, 255, 255)
            }
            self.max_step = 50
            self.steps = 0
            self.goals_reached = 0

            self.goals = {}
            self.board = {}

            # Place obstacles randomly on the grid
            self.place_random_objects('obstacle', num_obstacles)

            # Place goals randomly on the grid
            self.place_random_objects('goal', num_goals)
            # print(self.goals)


            # Place the agent at a random position
            self.agent_position = self.get_random_empty_position()
            self.board[str(self.agent_position[0])+'_'+str(self.agent_position[1])] = ["Robot"]
            # print(self.board)
            #self.update_grid()
        else:
            self.dimension = dimension
            self.num_obstacles = 0
            self.num_goals = 0
            self.grid = [['empty' for _ in range(dimension)] for _ in range(dimension)]

            # Initialize Pygame
            #pygame.init()
            self.cell_size = 50
            self.screen_size = (dimension * self.cell_size, dimension * self.cell_size)
            #self.screen = pygame.display.set_mode(self.screen_size)
            #pygame.display.set_caption('GridWorld')

            # Colors
            self.colors = {
                'empty': (255, 255, 255),
                'agent': (0, 128, 255),
                'obstacle': (0, 0, 0),
                'goal': (255, 0, 0),
                'visited_goal': (0, 255, 0),
                'label': (255, 255, 255)
            }
            self.max_step = 50
            self.steps = 0
            self.goals_reached = 0

            self.goals = {}
            self.board = input_board

            self.place_objects(input_board)
            #self.update_grid()

    def place_objects(self, input_board):
        for key in input_board.keys():
            i = int(key.split('_')[0])
            j = int(key.split('_')[-1])
            value = input_board[key]
            if 'Robot' in value:
                self.agent_position = (i,j)
            elif 'obstacle' in value:
                self.grid[i][j] = 'obstacle'
                self.num_obstacles += 1
            else:
                self.grid[i][j] = 'goal'
                try:
                    self.goals[(i, j)] = value[0][1]
                except:
                    print(value)
                    raise error
                self.num_goals += 1


    def place_random_objects(self, obj_type, num_objects):
        for i in range(num_objects):
            position = self.get_random_empty_position()
            self.grid[position[0]][position[1]] = obj_type
            if obj_type == 'goal':
                self.goals[(position[0], position[1])] = i
                self.board[str(position[0])+'_'+str(position[1])] = [("box", i)]
            else:
                self.board[str(position[0])+'_'+str(position[1])] = ["obstacle"]

    def get_random_empty_position(self):
        empty_positions = [(i, j) for i in range(self.dimension) for j in range(self.dimension) if self.grid[i][j] == 'empty']
        return random.choice(empty_positions)

    def move_agent(self, action):
        i, j = self.agent_position
        if action == 'up':
            if i > 0 and self.grid[i - 1][j] != 'obstacle':
                self.agent_position = (i - 1, j)
            else:
                print("Game over.")
                #pygame.quit()
                if i <= 0:
                    return 'moveout_check'
                else:
                    return 'collision_check'
        elif action == 'down':
            if i < self.dimension - 1 and self.grid[i + 1][j] != 'obstacle':
                self.agent_position = (i + 1, j)
            else:
                print("Game over.")
                #pygame.quit()
                if i >= self.dimension - 1:
                    return 'moveout_check'
                else:
                    return 'collision_check'
        elif action == 'left':
            if j > 0 and self.grid[i][j - 1] != 'obstacle':
                self.agent_position = (i, j - 1)
            else:
                print("Game over.")
                #pygame.quit()
                if j <= 0:
                    return 'moveout_check'
                else:
                    return 'collision_check'
        elif action == 'right':
            if j < self.dimension - 1 and self.grid[i][j + 1] != 'obstacle':
                self.agent_position = (i, j + 1)
            else:
                print("Game over.")
                #pygame.quit()
                if j >= self.dimension - 1:
                    return 'moveout_check'
                else:
                    return 'collision_check'

        elif action == 'pick':
            if self.agent_position in self.get_goal_positions() and self.goals_reached == self.goals[self.agent_position]:
                self.goals_reached += 1
                self.grid[self.agent_position[0]][self.agent_position[1]] = 'visited_goal'
                if len(self.board[str(i)+'_'+str(j)]) > 1:
                    del self.board[str(i)+'_'+str(j)][0]
                else:
                    del self.board[str(i)+'_'+str(j)]
            elif self.agent_position not in self.get_goal_positions():
                return 'pick_goal_exist_check'
            elif self.goals_reached != self.goals[self.agent_position]:
                return 'pick_check'

        self.steps += 1
        self.update_board(i, j)
        #self.update_grid()
        return None

    def update_board(self, i, j):
        if len(self.board[str(i)+'_'+str(j)]) > 1:
            self.board[str(i)+'_'+str(j)].remove("Robot")
        else:
            del self.board[str(i)+'_'+str(j)]

        key = str(self.agent_position[0])+'_'+str(self.agent_position[1])
        if key in self.board.keys():
            self.board[key].append("Robot")
        else:
            self.board[key] = ["Robot"]
        # print(self.board)

    '''
    def update_grid(self):
        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        pygame.quit()
        #        sys.exit()
        for i in range(self.dimension):
            for j in range(self.dimension):
                if self.grid[i][j] == 'goal':
                    #pygame.draw.rect(self.screen, self.colors['goal'], (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                    if (i, j) in self.goals.keys():
                        #label_font = pygame.font.SysFont(None, 25)
                        label_text = label_font.render(str(self.goals[(i,j)]+1), True, self.colors['label'])
                        self.screen.blit(label_text, (j * self.cell_size + self.cell_size // 2 - 8, i * self.cell_size + self.cell_size // 2 - 10))
                elif self.grid[i][j] == 'visited_goal':
                    pygame.draw.rect(self.screen, self.colors['visited_goal'], (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                else:
                    pygame.draw.rect(self.screen, self.colors[self.grid[i][j]], (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))

        pygame.draw.rect(self.screen, self.colors['agent'], (self.agent_position[1] * self.cell_size, self.agent_position[0] * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()
    '''

    def run(self, sequence_of_actions=None):
        #clock = pygame.time.Clock()

        while self.goals_reached < self.num_goals:
            #clock.tick(10)  # Adjust the speed of visualization

            if sequence_of_actions:
                if len(sequence_of_actions) > 0:
                    action = sequence_of_actions.pop(0)
                    self.move_agent(action)
            '''
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]:
                        self.move_agent('up')
                    elif keys[pygame.K_DOWN]:
                        self.move_agent('down')
                    elif keys[pygame.K_LEFT]:
                        self.move_agent('left')
                    elif keys[pygame.K_RIGHT]:
                        self.move_agent('right')
                    elif keys[pygame.K_SPACE]:
                        self.move_agent('pick')
            '''

            # if self.agent_position in self.get_goal_positions() and self.goals_reached == self.goals[self.agent_position]:
            #     self.goals_reached += 1
            #     self.grid[self.agent_position[0]][self.agent_position[1]] = 'visited_goal'

            #self.update_grid()

            if self.steps > self.max_step:
                print('Maximized step nunmber reached! Game over')
                #pygame.quit()
                sys.exit()

        print("All goals reached! Game over.")
        #pygame.quit()

    def get_goal_positions(self):
        return [(i, j) for i in range(self.dimension) for j in range(self.dimension) if self.grid[i][j] == 'goal']


    def get_neighbors(self, position, pick=None):
        # pdb.set_trace()
        i, j = position
        neighbors = []

        # Check up
        if i > 0 and self.grid[i - 1][j] != 'obstacle':
            neighbors.append(('up', (i - 1, j)))

        # Check down
        if i < self.dimension - 1 and self.grid[i + 1][j] != 'obstacle':
            neighbors.append(('down', (i + 1, j)))

        # Check left
        if j > 0 and self.grid[i][j - 1] != 'obstacle':
            neighbors.append(('left', (i, j - 1)))

        # Check right
        if j < self.dimension - 1 and self.grid[i][j + 1] != 'obstacle':
            neighbors.append(('right', (i, j + 1)))

        if pick is not 'pick':
            if self.grid[i][j] == 'goal':
                neighbors.append(('pick', (i, j)))

        return neighbors

    def find_optimal_path(self, goal_position):
        open_set = []
        closed_set = set()

        start_node = (None, self.agent_position, None)

        heapq.heappush(open_set, (0, start_node))

        while open_set:
            # print(open_set)
            current_cost, current_node = heapq.heappop(open_set)
            # print(current_node)
            if current_node[1] == goal_position and current_node[0] == 'pick':
                return self.reconstruct_path(current_node)

            closed_set.add(current_node[1])

            for neighbor in self.get_neighbors(current_node[1], pick = current_node[0]):
                if neighbor[0] is not 'pick' and neighbor[1] in closed_set:
                    continue

                tentative_cost = current_cost + 1  # Uniform cost for each step

                if (tentative_cost, neighbor[1]) not in open_set:
                    heapq.heappush(open_set, (tentative_cost + self.heuristic(neighbor[1], goal_position), neighbor+ (current_node,)))

        return None

    def reconstruct_path(self, goal_node):
        current_node = goal_node
        path = []
        # print(current_node)

        while current_node[2] is not None: #self.agent_position:
            action, _, next_node = current_node
            path.append(action)
            current_node = next_node
            # pdb.set_trace()

        return path[::-1]  # Reverse the path to get the correct order

    def heuristic(self, current_node, goal_node):
        # Euclidean distance heuristic
        x1, y1 = current_node
        x2, y2 = goal_node
        # import pdb; pdb.set_trace()
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# This code is to update the current surrounding environment information to LLM, i.e., notice the goals and obstacles on the left, right, up, down for one step
def state_update_func(pg_row_num, pg_column_num, pg_dict):
  for key in pg_dict.keys():
    if 'Robot' in pg_dict[key]:
      i = int(key.split('_')[0])
      j = int(key.split('_')[-1])
      break
  state_update_prompt_start_overall_summary = 'The current environment information is as follows: \n\n The current environment is a {}x{} grid world. '.format(pg_row_num, pg_column_num)
  state_update_prompt_start_overall_summary += 'The robot yourself is located in square [{}, {}]. '.format(i, j)
  state_update_prompt = '\n\nBased on the above environment information, you are now in square [{}, {}]. '.format(i, j)


  if len(pg_dict[key]) > 1:
    obj = pg_dict[key][0]
    state_update_prompt += 'Your current location is goal_{}. '.format(obj[1]) #obj label
    state_update_prompt_start_overall_summary += f'The goal_{pg_dict[key][0][1]} is also located in square [{i}, {j}]. '
  else:
    state_update_prompt += 'Your current location does not have goals. '

  for key in pg_dict.keys():
    if 'Robot' not in pg_dict[key]:
        i_second = int(key.split('_')[0])
        j_second = int(key.split('_')[-1])
        if isinstance(pg_dict[key][0], list):
            state_update_prompt_start_overall_summary += f'The goal_{pg_dict[key][0][1]} is located in square [{i_second}, {j_second}]. '
        else:
            state_update_prompt_start_overall_summary += f'One obstacle is located in square [{i_second}, {j_second}]. '

  # Check left
  if j > 0:
    if str(i)+'_'+str(j-1) in pg_dict.keys():
      if 'obstacle' in pg_dict[str(i)+'_'+str(j-1)]:
        state_update_prompt += 'Your left is one obstacle. '
      else:
        obj = pg_dict[str(i)+'_'+str(j-1)][0]
        state_update_prompt += 'Your left is goal_{} '.format(obj[1]) #obj label
    else:
      state_update_prompt += 'Your left does not have object. '
  else:
    state_update_prompt += 'Your left does not have way because you are on the boundary. '

  # Check right
  if j < pg_column_num - 1:
    if str(i)+'_'+str(j+1) in pg_dict.keys():
      if 'obstacle' in pg_dict[str(i)+'_'+str(j+1)]:
        state_update_prompt += 'Your right is one obstacle. '
      else:
        obj = pg_dict[str(i)+'_'+str(j+1)][0]
        state_update_prompt += 'Your right is goal_{} '.format(obj[1]) #obj label
    else:
      state_update_prompt += 'Your right does not have object. '
  else:
    state_update_prompt += 'Your right does not have way because you are on the boundary. '

  # Check up
  if i > 0:
    if str(i-1)+'_'+str(j) in pg_dict.keys():
      if 'obstacle' in pg_dict[str(i-1)+'_'+str(j)]:
        state_update_prompt += 'Your upper is one obstacle. '
      else:
        obj = pg_dict[str(i-1)+'_'+str(j)][0]
        state_update_prompt += 'Your upper is goal_{} '.format(obj[1]) #obj label
    else:
      state_update_prompt += 'Your upper does not have object. '
  else:
    state_update_prompt += 'Your upper does not have way because you are on the boundary. '

  # Check down
  if i < pg_row_num - 1:
    if str(i+1)+'_'+str(j) in pg_dict.keys():
      if 'obstacle' in pg_dict[str(i+1)+'_'+str(j)]:
        state_update_prompt += 'Your down is one obstacle. '
      else:
        obj = pg_dict[str(i+1)+'_'+str(j)][0]
        state_update_prompt += 'Your down is goal_{} '.format(obj[1]) #obj label
    else:
      state_update_prompt += 'Your down does not have object. '
  else:
    state_update_prompt += 'Your down does not have way because you are on the boundary. '

  return state_update_prompt_start_overall_summary + state_update_prompt

def action_from_response(env, LLM_response, order_required):
  pg_dict_input = env.board

  box_count_input = 0
  for value in pg_dict_input.values():
      if any("box" in item for item in value if isinstance(item, list)):
          box_count_input += 1

  wrong_order_check = None
  if order_required:
    wrong_order_check = False
  collision_check = False
  moveout_check = False
  pick_goal_exist_check = False
  env_act_feedback = ''
  #pg_dict_updated = copy.deepcopy(pg_dict_input)
  ### Your Code Here, update pg_dict_input: location of robot, goals picked up should be deleted
  # If collision, collision_check = True
  # If current location is out of the whole field, moveout_check = True
  # If the robot pick up the goal with wrong order, wrong_order_check = True

  re = None
  if LLM_response == 'Move up':
    re = env.move_agent('up')
  if LLM_response == 'Move down':
    re = env.move_agent('down')
  if LLM_response == 'Move left':
    re = env.move_agent('left')
  if LLM_response == 'Move right':
    re = env.move_agent('right')
  if LLM_response == 'Pick goal':
    re = env.move_agent('pick')
  if re is not None:
    if re == 'collision_check':
      collision_check = True
    elif re == 'moveout_check':
      moveout_check = True
    elif re == 'pick_check':
      wrong_order_check = True
    elif re == 'pick_goal_exist_check':
      pick_goal_exist_check = True
  #else:
  pg_dict_updated = env.board

  box_count_output = 0
  for value in pg_dict_updated.values():
      if any("box" in item for item in value if isinstance(item, list)):
          box_count_output += 1

  print('###########action from response############')
  print('box_count_input', box_count_input)
  print('box_count_output', box_count_output)

  if LLM_response not in ['Move up', 'Move down', 'Move left', 'Move right', 'Pick goal']:
    env_act_feedback += f'Your assigned action {LLM_response} for the current robot is not in the doable action list.'

  if pick_goal_exist_check:
    print('-------pick_goal_exist_check!-------')
    env_act_feedback += f'You cannot pick up the goal right now since there is no goals in your square. '

  return pg_dict_updated, collision_check, moveout_check, wrong_order_check, env_act_feedback


def env_create(pg_row_num, goal_num, obs_num):
  # pg_dict records the items in each square over steps, here in the initial setting, we randomly assign items into each square
  # pg_dict = {}
  ### Your Code Here to assign goals and obstacles into dictionary pg_dict###
  # Output should look like {"1_4": ["Robot"], "1_2": ["box", 1], "1_3": ["box", 2],..., "2_1": ["obstacle"], "2_4": ["obstacle"]}
  # "1_2 are x, y locations", 1 is the index for boxes in case for ordered visiting
  env = GridWorld(pg_row_num, obs_num, goal_num)
  pg_dict = copy.deepcopy(env.board)

  paths = []
  start_goal_pairs = env.goals.items()
  #print(start_goal_pairs)
  for goal_position, goal_label in start_goal_pairs:
      path = env.find_optimal_path(goal_position)
      if path:
          #print(f"Optimal path to goal {goal_label + 1}: {path}")
          paths += path
          for action in path:
              # print(action)
              env.move_agent(action)
              #env.update_grid()
              # pdb.set_trace()
              # state_update_func(pg_row_num, pg_column_num, env.board)
              # pygame.time.delay(500)  # Delay to visualize each step (optional)
      else:
          #print(f"No path found to goal {goal_label + 1}.")
          paths = None
          break
  # pdb.set_trace()
  return pg_dict, paths

def create_env6(Saving_path, repeat_num = 10):
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  for pg_row_num, pg_column_num, goal_num, obs_num in [(4,4,3,4), (5,5,4,6), (6,6,5,10), (7,7,6,14)]:
    if not os.path.exists(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}_{goal_num}_{obs_num}'):
      os.makedirs(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}_{goal_num}_{obs_num}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}_{goal_num}_{obs_num}')
      os.makedirs(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}_{goal_num}_{obs_num}', exist_ok=True)

    iteration_num = 0
    while iteration_num < repeat_num:
      # Define the total row and column numbers of the whole playground, and the item number of each colored target and box
      pg_dict, paths = env_create(pg_row_num, goal_num, obs_num)
      if paths is not None:
        os.makedirs(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}_{goal_num}_{obs_num}/pg_state{iteration_num}', exist_ok=True)
        with open(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}_{goal_num}_{obs_num}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'w') as f:
          json.dump(pg_dict, f)
        with open(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}_{goal_num}_{obs_num}/pg_state{iteration_num}/Astarpath{iteration_num}.json', 'w') as f:
          json.dump(paths, f)
        iteration_num += 1
