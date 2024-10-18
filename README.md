# PRompt Optimization in Multi-Step Tasks (PROMST): Integrating Human Feedback and Preference Alignment <span style="color: #8B0000;">(EMNLP 2024, Oral, Top 3.4%)</span>

# ([Website](https://yongchao98.github.io/MIT-REALM-PROMST/))
Paper Link: https://arxiv.org/pdf/2402.08702.pdf

This project is for automatic prompt optimization with a focus on multi-step tasks.

## Requirements
Please setup the environment with conda and pip as follows:
```
conda create -n PROMST python=3.10
conda activate PROMST
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::tiktoken
pip install openai --upgrade
pip install pygame
conda install conda-forge::transformers
conda install anaconda::scikit-learn
pip install gym==0.26.2
pip install pddlgym
pip install tarski
```
Note that when doing the **Logistics** and **BlocksWorld** tasks, we slightly modify the pddlgym files. **Thus, you need to substitute the original installed pddlgym directory in packages (e.g., /Users/Your user name/opt/anaconda3/envs/PROMST/lib/python3.10/site-packages/pddlgym) with the pddlgym directory in BlocksWorld/env_data_BlocksWorld/pddlgym. Like the following code:** 
```
cd /Users/Your user name/opt/anaconda3/envs/PROMST/lib/python3.10/site-packages
cp -r /Your path to PROMST/PROMST/BlocksWorld/env_data_BlocksWorld/pddlgym ./
```

## Usage
Your system does not need GPU to train the score model if you set with_score_model = 'False'. To run each task, you enter into each task directory and set up the OPENAI_API key (sk-..) in LLM.py line 11. Then you can run the following code. The args can be changed with your preference.
```
python env3-box-arrange-train_MCTS.py
-experiment_trial_num 1
-input_error_prompt_token_limit 15000
-model_name_promptLLM gpt-4-1106-preview
-model_name_testLLM gpt-3.5-turbo-16k-0613
-min_level 2
-n_children 8
-n_selected 2
-prompt_method PROMST
-with_score_model 'False'
-Training_path ../BoxLift/train_set/
-Testing_path ../BoxLift/test_set/
-base_path ../BoxLift/
```

```
python env3-box-arrange-train_MCTS.py -experiment_trial_num 3 -input_error_prompt_token_limit 15000 -model_name_promptLLM gpt-4-1106-preview -model_name_testLLM gpt-3.5-turbo-16k-0613 -min_level 2 -n_children 4 -n_selected 2 -prompt_method PROMST -with_score_model 'False' -with_SumLLM 'False' -Training_path ../BoxLift/train_set/ -Testing_path ../BoxLift/test_set/ -base_path ../BoxLift/
```

Command for Alfworld/Scienceworld/Webarena
```
python agentboard/env9-box-arrange-train_MCTS.py     --cfg-path eval_configs/main_results_all_tasks.yaml     --tasks alfworld    --log_path ./results/gpt-3.5-turbo-16k-0613     --project_name evaluate-gpt-4 --experiment_trial_num 1  --model_name_promptLLM gpt-4-1106-preview  --model_name_testLLM gpt-3.5-turbo-16k-0613 --min_level 2 --n_children 4 --n_selected 2 --prompt_method PROMST --with_score_model 'False' --base_path ./alfworld_result/ --max_num_steps 30
```

Command for the restart from training process for Alfworld/Scienceworld/Webarena
```
python agentboard/env9-box-arrange-train_MCTS_restart.py     --cfg-path eval_configs/main_results_all_tasks.yaml     --tasks webarena    --log_path ./results/gpt-3.5-turbo-16k-0613     --project_name evaluate-gpt-4 --experiment_trial_num 3  --model_name_promptLLM gpt-4-1106-preview  --model_name_testLLM gpt-3.5-turbo-16k-0613 --min_level 2 --n_children 4 --n_selected 2 --prompt_method PROMST --with_score_model 'False' --base_path ./webarena_result/ --max_num_steps 30
```

| Args to be set | Choices | Explanation |
| --------------- | --------------- | --------------- |
| experiment_trial_num | 1, 2, 3, ... | The trial number to record each trial |
| input_error_prompt_token_limit | 10000, 15000, 20000, ... | The prompt maximum token length for context sliding window |
| model_name_promptLLM | gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, gpt-4-1106-preview | LLM type of PromptLLM |
| model_name_testLLM | gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, gpt-4-1106-preview | LLM type of TestLLM |
| min_level | 1, 2, 3, ... | Minimum number of prompt levels to be explored |
| n_children | 4, 8, 12, ... | Number of prompts to be expanded in each level |
| n_selected | 2, 3, 4, ... | Number of best prompts to be further optimized in each level |
| prompt_method | PROMST, APE, APO | Prompt optimziation methods |
| with_score_model | 'False', 'True' | Whether implementing socre model in PROMST |
| Training_path | ../BoxLift/train_set/ | The path to the created train_set path |
| Tesing_path | ../BoxLift/train_set/, ../BoxLift/test_set/ | The path to the created test_set path (be the same as the Training_path during optimization) |
| base_path | ../BoxLift/ | The base path to save the prompt optimization results |

## 📝 Citation

```bibtex
@article{chen2024prompt,
  title={Prompt optimization in multi-step tasks (promst): Integrating human feedback and preference alignment},
  author={Chen, Yongchao and Arkin, Jacob and Hao, Yilun and Zhang, Yang and Roy, Nicholas and Fan, Chuchu},
  journal={arXiv preprint arXiv:2402.08702},
  year={2024}
}
```

## Recommended Work

[Large language models are human-level prompt engineers](https://arxiv.org/abs/2211.01910)

[Automatic Prompt Optimization with “Gradient Descent” and Beam Search](https://arxiv.org/abs/2305.03495)

[Scalable Multi-Robot Collaboration with Large Language Models: Centralized or Decentralized Systems?](https://yongchao98.github.io/MIT-REALM-Multi-Robot/)
