# PRompt Optimization in Multi-Step Tasks (PROMST): Integrating Human Feedback and Preference Alignment

This project is for automatic prompt optimization with a focus on multi-step tasks.

## Requirements
Please setup the environment with conda and pip as follows:
```
conda create -n PROMST python=3.10
conda activate PROMST
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::tiktoken
pip install openai
pip install pygame
conda install conda-forge::transformers
conda install anaconda::scikit-learn
pip install gym==0.26.2
pip install pddlgym
pip install tarski
```

## Usage

```
python env3-box-arrange-train_MCTS.py
-experiment_trial_num 1
-input_error_prompt_token_limit 15000
-model_name_promptLLM gpt-4-1106-preview
-model_name_testLLM gpt-3.5-turbo-16k-0613
-min_level 2
-prompt_method PROMST
-with_score_model 'False'
-Training_path ../BoxLift/train_set/
-Testing_path ../BoxLift/test_set/
-base_path ../BoxLift/
```

| Args to be set | Choices | Explanation |
| --------------- | --------------- | --------------- |
| experiment_trial_num | 1, 2, 3, ... | The trial number to record each trial |
| input_error_prompt_token_limit | 10000, 15000, 20000, ... | The prompt maximum token length for context sliding window |
| model_name_promptLLM | gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, gpt-4-1106-preview | LLM type of PromptLLM |
| model_name_testLLM | gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, gpt-4-1106-preview | LLM type of TestLLM |
| min_level | 1, 2, 3, ... | Minimum number of prompt levels to be explored |
| prompt_method | PROMST, APE, APO | Prompt optimziation methods |
| with_score_model | 'False', 'True' | Whether implementing socre model in PROMST |
| Training_path | ../BoxLift/train_set/ | The path to the created train_set path |
| Tesing_path | ../BoxLift/train_set/, ../BoxLift/test_set/ | The path to the created test_set path (be the same as the Training_path during optimization) |
| base_path | ../BoxLift/ | The base path to save the prompt optimization results |
