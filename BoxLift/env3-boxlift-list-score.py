import os
#import matplotlib.pyplot as plt

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")

def read_file_content(file_path):
    """Reads the content of a file given its path."""
    with open(file_path, 'r') as file:
        return file.read().strip()

def process_directory(base_dir):
    """Processes each 'prompt_{prompt_index}' subdirectory in the given base directory."""
    prompt_score_accumulator = {}
    prompt_count = {}
    dir_index = 1
    total_prompt_trial = 0
    while os.path.isdir(os.path.join(base_dir, f'prompt_{dir_index}')):
        dir_name = f'prompt_{dir_index}'
        prompt_dir = os.path.join(base_dir, dir_name)
        score_file = os.path.join(prompt_dir, 'total_score.txt')
        prompt_file = os.path.join(prompt_dir, 'prompt.txt')

        if os.path.exists(score_file) and os.path.exists(prompt_file):
            total_prompt_trial += 1
            score = float(read_file_content(score_file))
            prompt = read_file_content(prompt_file)
            print(f"Score: {score}")
            print('Prompt Length: ', len(enc.encode(prompt)))

            if prompt in prompt_score_accumulator:
                prompt_score_accumulator[prompt] += score
                prompt_count[prompt] += 1
            else:
                prompt_score_accumulator[prompt] = score
                prompt_count[prompt] = 1
        else:
            print('Missing files!')
        dir_index += 1

    return prompt_score_accumulator, prompt_count, total_prompt_trial


def process_multiple_directories(directories):
    """Processes multiple directories and merges their results."""
    prompt_score_accumulator = {}
    prompt_count = {}
    total_prompt_trial = 0
    for directory in directories:
        prompts_scores, prompts_counts, total_prompt_trial_sub = process_directory(directory)
        total_prompt_trial += total_prompt_trial_sub
        for prompt, score in prompts_scores.items():
            if prompt in prompt_score_accumulator:
                prompt_score_accumulator[prompt] += score
                prompt_count[prompt] += prompts_counts[prompt]
            else:
                prompt_score_accumulator[prompt] = score
                prompt_count[prompt] = prompts_counts[prompt]

    # Calculate average scores
    prompt_score_avg = {prompt: prompt_score_accumulator[prompt] / prompt_count[prompt] for prompt in prompt_score_accumulator}

    return prompt_score_avg, total_prompt_trial

def select_top_k_prompts(prompt_score_dict, k):
    """Selects the top k prompts based on their average score."""
    # Sorting the dictionary by score in descending order and selecting the top k items
    top_k_items = sorted(prompt_score_dict.items(), key=lambda x: x[1], reverse=True)[:k]

    # Creating a new dictionary from the top k items
    top_k_prompts = {item[0]: item[1] for item in top_k_items}
    return top_k_prompts


def ensure_directory_exists(directory):
    """Ensures the specified directory exists, creates it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_top_prompts_and_scores(top_prompts, save_path):
    """Saves the top prompts and their scores to the specified path."""
    ensure_directory_exists(save_path)

    for i, (prompt, score) in enumerate(top_prompts.items(), start=1):
        with open(os.path.join(save_path, f'prompt_{i}.txt'), 'w') as file:
            file.write(prompt)

        with open(os.path.join(save_path, f'score_{i}.txt'), 'w') as file:
            file.write(str(score))

def plot_scores(all_prompts_scores):
    """Plots the scores as a scatter plot."""
    scores = list(all_prompts_scores.values())
    prompts = range(len(scores))

    plt.scatter(prompts, scores)
    plt.title('Prompt Scores')
    plt.xlabel('Prompt Index')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Assuming scores range from 0 to 1
    plt.show()

# Change the bath_path to yout current directory
base_path = '../BoxLift/'

# Here is the path of the directory containing the prompt optimization trials, you can list multiple directories if needed
directories = [
   os.path.join(base_path, f'prompt_optimization_train_result/train_result_dir_trial_1_PromptLLM_gpt-4-1106-preview_TestLLM_gpt-3.5-turbo-16k-0613_PromptMethod_PROMST'),
   os.path.join(base_path, f'prompt_optimization_train_result/train_result_dir_trial_2_PromptLLM_gpt-4-1106-preview_TestLLM_gpt-3.5-turbo-16k-0613_PromptMethod_PROMST')
]

prompt_score_dict, total_prompt_trial = process_multiple_directories(directories)
print('Prompt total number: ', len(list(prompt_score_dict.values())))
print('Total prompt trial: ', total_prompt_trial)
print('Prompt unique ratio: ', len(list(prompt_score_dict.values())) / total_prompt_trial)
#plot_scores(prompt_score_dict)

for prompt, score in prompt_score_dict.items():
    print(f"Score: {score}")
    print(len(enc.encode(prompt)))

# Select top k prompts
k = 3  # Define k as needed
top_k_prompts = select_top_k_prompts(prompt_score_dict, k)
print("Top", k, "Prompts:")
for prompt, score in top_k_prompts.items():
    print(f"Score: {score}")

save_path = os.path.join(base_path, 'round1_gpt-4-1106-preview_gpt-3.5-turbo-16k-0613_prompt_method_PROMST_best_three')
save_top_prompts_and_scores(top_k_prompts, save_path)
