from LLM import *
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000

def input_prompt_1_func_total(starting_prompt_task_explain, state_update_prompt, response_total_list,
                              pg_state_list, env_act_feedback_list):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(env_act_feedback_list) != 1:
    raise error('state and env act feedback list do not match')

  user_prompt_1 = f'''
  The previous state and action pairs at each step are:

  The current left boxes and agents are:
  {state_update_prompt}
  
  Now, plan the next step:
  '''
  token_num_count = len(enc.encode(starting_prompt_task_explain)) + len(enc.encode(user_prompt_1))

  state_action_prompt = ''
  for i in range(len(response_total_list) - 1, -1, -1):
    state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\nEnvironment Feedback{i + 1}: {env_act_feedback_list[i]}\n\n' + state_action_prompt
    if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
      state_action_prompt = state_action_prompt_next
    else:
      break

  user_prompt_1 = f'''
  The previous state and action pairs at each step are:
  {state_action_prompt}

  The current left boxes and agents are:
  {state_update_prompt}
  
  Now, plan the next step:
  '''
  return user_prompt_1

def prompt_to_promptLLM_func_total_APE(prompt_task_explain):

  prompt_to_promptLLM = f'Imagine you are a prompt optimizer. ' \
                        f'Generate a variation of the following prompt while keeping the semantic meaning.' \
                        f'Here is the original prompt:\n{prompt_task_explain}\n\n' \
                        f'Output the optimized prompt of task description without other texts: ' \

  return prompt_to_promptLLM

def error_string_func_APO(rear_prompt_1, response_1, env_act_feedback_1 = None, rear_prompt_2 = None, response_2 = None):
  if env_act_feedback_1 == None:
      prompt_to_promptLLM = f'\n\nHere is the prompt and response from previous round:\n\n' \
                            f'Here is the prompt of previous state/action pairs and the current state:\n{rear_prompt_1}\n\n' \
                            f'Here is the initial response generated by the task planning LLM agent based on the concatenated prompt of task description and current state:\n{response_1}\n\n' \

  else:
      prompt_to_promptLLM = f'\n\nHere is the prompt and response from previous two rounds:\n\n' \
                            f'Round1: \nHere is the prompt of previous state/action pairs and the current state:\n{rear_prompt_1}\n\n' \
                            f'Here is the initial response generated by the task planning LLM agent based on the concatenated prompt of task description and current state:\n{response_1}\n\n' \
                            f'Here is the environment feedback after executing the plan:\n{env_act_feedback_1}\n\n' \
                            f'Round2: \nHere is the prompt of previous state/action pairs and the current state:\n{rear_prompt_2}\n\n' \
                            f'Here is the initial response generated by the task planning LLM agent based on the concatenated prompt of task description and current state:\n{response_2}\n\n' \

  return prompt_to_promptLLM

def prompt_to_self_error_summarizer_APO(current_prompt, error_string):

  prompt_to_promptLLM = f'I’m writing prompts for a language model designed for a task. My current prompt of task specification is:' \
                        f'{current_prompt}' \
                        f'But this prompt gets the following examples wrong:' \
                        f'{error_string}' \
                        f'For each wrong example, carefully examine each question and wrong answer step by step, provide comprehensive and different reasons why the prompt leads to the wrong answer. ' \
                        f'At last, based on all these reasons, summarize and list all the aspects that can improve the prompt.'
  return prompt_to_promptLLM

def prompt_to_error_summarizer_PROMST(current_prompt, error_string):

  prompt_to_promptLLM = f'Imagine you are a prompt optimizer based on the feedback from the human and task execution feedback.' \
                        f'I’m writing prompts for a language model designed for a task. My current prompt of task specification is:' \
                        f'{current_prompt}' \
                        f'But this prompt gets the following examples wrong:' \
                        f'{error_string}' \
                        f'Based on all these errors and feedback, summarize the reasons and list all the aspects that can improve the prompt. Keep your summary concise and clear.'
  return prompt_to_promptLLM

def prompt_to_promptLLM_func_total_with_env_act_feedback_PROMST_APO(prompt_task_explain, error_feedback, trajectory_prompts):

  prompt_to_promptLLM = f'Imagine you are a prompt optimizer based on the feedback from the human and task execution feedback.' \
                        f'Here is the prompt of task description:\n{prompt_task_explain}\n\n' \
                        f'However, the response generated from the initial task description prompt owns some errors. Here are the error feedback from humans:\n{error_feedback}\n\n' \
                        f'There is a list of former prompts including the current prompt, and each prompt is modified from its former prompts:\n{trajectory_prompts}' \
                        f'Based on the feedback, think about why the task planning LLM agent makes the error and try to optimize the prompt of task description to avoid this error.\n\n' \
                        f'The new prompts should follow these guidelines: 1. The new prompts should solve the current prompt’s problems. 2. The new prompts should consider the list of prompts and evolve based on the current prompt.' \
                        f'Output the optimized prompt of task description without other texts: ' \

  return prompt_to_promptLLM

def feedback_to_promptLLM_func(rear_prompt_1, response_1, env_act_feedback_1,
                                                                                   rear_prompt_2, response_2, env_act_feedback_2,
                                                                                   error_feedback = ''):

  prompt_to_promptLLM = f'\n\nHere is the prompt and response from previous two rounds:\n\n' \
                        f'Round1: \nHere is the prompt of previous state/action pairs and the current state:\n{rear_prompt_1}\n\n' \
                        f'Here is the initial response generated by the task planning LLM agent based on the concatenated prompt of task description and current state:\n{response_1}\n\n' \
                        f'Here is the environment feedback after executing the plan:\n{env_act_feedback_1}\n\n' \
                        f'Round2: \nHere is the prompt of previous state/action pairs and the current state:\n{rear_prompt_2}\n\n' \
                        f'Here is the initial response generated by the task planning LLM agent based on the concatenated prompt of task description and current state:\n{response_2}\n\n' \
                        f'Here is the environment feedback after executing the plan:\n{env_act_feedback_2}\n\n' \
                        f'Based on previous rounds, here is the error feedback from humans:\n{error_feedback}\n\n' \

  return prompt_to_promptLLM

def message_construct_func(user_prompt_list, response_total_list):
  messages=[{"role": "system", "content": "You are a helpful assistant."}]
  #print('length of user_prompt_list', len(user_prompt_list))
  for i in range(len(user_prompt_list)):
    messages.append({"role": "user", "content": user_prompt_list[i]})
    if i < len(user_prompt_list)-1:
      messages.append({"role": "assistant", "content": response_total_list[i]})
  return messages