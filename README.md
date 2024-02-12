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

```
python env3-box-arrange-train_MCTS.py
-experiment_trial_num 1
-input_error_prompt_token_limit 15000
-model_name_promptLLM gpt-4-1106-preview
-model_name_testLLM gpt-3.5-turbo-16k-0613
-min_level 2
-prompt_method PROMST
-Training_path ../BoxLift/train_set/
-Testing_path ../BoxLift/test_set/
-base_path ../BoxLift/
```
