from tarski.io import PDDLReader
import random
import yaml
import re

def get_problem(instance, domain):
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain(domain)
    return reader.parse_instance(instance)


def get_sorted(init_atoms):
    return sorted(init_atoms, key=lambda x: x.symbol.name + " " + " ".join([subterm.name for subterm in x.subterms]))


def parse_problem(problem, data, shuffle):
    def parse(init_goal_preds, OBJS):
        TEXT = ""
        predicates = []

        init_goal_preds = list(init_goal_preds)
        for atom in init_goal_preds:
            objs = []
            for subterm in atom.subterms:
                if 'obfuscated' in data["domain_name"]:
                    objs.append(subterm.name.replace('o','object_'))
                elif 'blocksworld' in data['domain_name']:
                    objs.append(OBJS[subterm.name])
                elif 'logistics' in data['domain_name']:
                    obj = subterm.name
                    objs.append(f"{OBJS[obj[0]].format(*[chr for chr in obj if chr.isdigit()])}")
                elif 'depots' in data['domain_name']:
                    objs.append(subterm.name)
                # ADD SPECIFIC TRANSLATION FOR EACH DOMAIN HERE
            try:
                pred_string = data['predicates'][atom.symbol.name].format(*objs)
                predicates.append(pred_string)
            except:
                # print("[-]: Predicate not found in predicates dict: {}".format(atom.symbol.name))
                pass
            
        if len(predicates) > 1:
            predicates = [item for item in predicates if item]
            TEXT += ", ".join(predicates[:-1]) + f" and {predicates[-1]}"
        else:
            TEXT += predicates[0]
        return TEXT

    OBJS = data['encoded_objects']

    init_atoms = get_sorted(problem.init.as_atoms())
    goal_preds = get_sorted(problem.goal.subformulas) if hasattr(problem.goal, 'subformulas') else [problem.goal]

    if shuffle:
        random.shuffle(init_atoms)
        random.shuffle(goal_preds)
    # print(shuffle,init_atoms)
    print(init_atoms)
    print(goal_preds)
    # ----------- INIT STATE TO TEXT ----------- #
    INIT = parse(init_atoms, OBJS)

    # ----------- GOAL TO TEXT ----------- #
    GOAL = parse(goal_preds, OBJS)

    return INIT, GOAL

def instance_to_text(problem, get_plan, data, shuffle=False):
    """
    Function to make an instance into human-readable format
    """

    OBJS = data['encoded_objects']

    # ----------- PARSE THE PROBLEM ----------- #
    INIT, GOAL = parse_problem(problem, data, shuffle)

    # ----------- PLAN TO TEXT ----------- #
    PLAN = ""
    plan_file = "sas_plan"
    if get_plan:
        PLAN = "\n"
        with open(plan_file) as f:
            plan = [line.rstrip() for line in f][:-1]

        for action in plan:
            action = action.strip("(").strip(")")
            act_name, objs = action.split(" ")[0], action.split(" ")[1:]
            if 'obfuscated' in data["domain_name"]:
                objs = [j.replace('o','object_') for j in objs]
            elif 'blocksworld' in data['domain_name']:
                objs = [OBJS[obj] for obj in objs]
            elif 'logistics' in data['domain_name']:
                objs = [f"{OBJS[obj[0]].format(*[chr for chr in obj if chr.isdigit()])}" for obj in objs]
            #elif 'depots' in data['domain_name']:  no formatting necessary
            # ADD SPECIFIC TRANSLATION FOR EACH DOMAIN HERE
        
            PLAN += data['actions'][act_name].format(*objs) + "\n"
        PLAN += "[PLAN END]\n"

    return INIT, GOAL, PLAN, data

def read_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def has_digit(string):
    return any(char.isdigit() for char in string)

def text_to_plan_logistics(text, action_set, plan_file, ground_flag=False):
    # import pdb; pdb.set_trace()
    raw_actions = [str(i).split('-')[0].lower() for i in list(action_set)]
    # ----------- GET PLAN FROM TEXT ----------- #
#     load package_0 into airplane_0 at location_0_0
# load package_1 into airplane_1 at location_1_0
# fly airplane_0 from location_0_0 to location_1_0
# fly airplane_1 from location_1_0 to location_0_0
# unload package_0 from airplane_0 at location_1_0
# unload package_1 from airplane_1 at location_0_0
    plan = ""
    readable_plan = ""
    lines = [line.strip().lower() for line in text.split("\n")]
    for line in lines:
        if not line:
            continue
        if '[COST]' in line:
            break
        
        if line[0].isdigit() and line[1]=='.':
            line = line[2:]
            line = line.replace(".", "")
        elif line[0].isdigit() and line[1].isdigit() and line[2]=='.':
            line = line[3:]
            line = line.replace(".", "")

        objs = [i[0]+'-'.join(i.split('_')[1:]) for i in line.split() if has_digit(i)]
        # objs = [i[0]+'-'.join(i.split('_')[1:]) for i in line.split()]
        
        
        if line.split()[0] in raw_actions:
            action = line.split()[0]
            print('action', action)
            print(objs)
            if 'load' in action or 'unload' in action:  
                to_check = objs[1]
            else:
                to_check = objs[0]
            if 'a' in to_check:
                action+='-airplane'
            elif 't' in to_check:
                action+='-truck'
            else:
                print(line, objs)
                raise ValueError
            if action=='drive-truck' and len(objs)==3:
                objs.append("c"+[i for i in objs[1] if i.isdigit()][0])


            readable_action = "({} {})".format(action, " ".join(objs))
            if not ground_flag:
                action = "({} {})".format(action, " ".join(objs))
            else:
                action = "({}_{})".format(action, "_".join(objs))
            plan += f"{action}\n"
            readable_plan += f"{readable_action}\n"
    # print(f"[+]: Saving plan in {plan_file}")
    # file = open(plan_file, "wt")
    # file.write(plan)
    # file.close()
    return plan, readable_plan