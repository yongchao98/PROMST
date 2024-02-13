import json
import pdb
import numpy as np
quries = {}
optimal_plans = {}
with open("Logistics/logistics_env/task_instance_info.json", "r") as f:
    info =  json.load(f)
for instance in info['instances']:
    index = instance['example_instance_id']
    query = instance['query'][:-32]
    ground_truth_plan = instance['ground_truth_plan']
    quries[index] = query
    optimal_plans[index] = ground_truth_plan
# pdb.set_trace()
np.save('Logistics/logistics_env/queries.npy', quries) 
np.save('Logistics/logistics_env/optimal_plans.npy', optimal_plans) 