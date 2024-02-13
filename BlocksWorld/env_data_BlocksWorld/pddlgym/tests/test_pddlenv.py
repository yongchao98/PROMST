from pddlgym.core import PDDLEnv, InvalidAction
from pddlgym.structs import Predicate, Type, LiteralConjunction

import os


def test_pddlenv():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    domain_file = os.path.join(dir_path, 'pddl', 'test_domain.pddl')
    problem_dir = os.path.join(dir_path, 'pddl', 'test_domain')

    env = PDDLEnv(domain_file, problem_dir, raise_error_on_invalid_action=True,
                  dynamic_action_space=True)
    env2 = PDDLEnv(domain_file, problem_dir, raise_error_on_invalid_action=True,
                   dynamic_action_space=False)

    type1 = Type('type1')
    type2 = Type('type2')
    pred1 = Predicate('pred1', 1, [type1])
    pred2 = Predicate('pred2', 1, [type2])
    pred3 = Predicate('pred3', 3, [type1, type2, type2])
    action_pred = Predicate('actionpred', 1, [type1])

    obs, _ = env.reset()

    assert obs.literals == frozenset({ pred1('b2'), pred2('c1'), pred3('a1', 'c1', 'd1'), 
        pred3('a2', 'c2', 'd2') })

    # Invalid action
    action = action_pred('b1')

    try:
        env.step(action)
        assert False, "Action was supposed to be invalid"
    except InvalidAction:
        pass

    assert action not in env.action_space.all_ground_literals(obs), "Dynamic action space not working"
    env2.reset()
    assert action in env2.action_space.all_ground_literals(obs), "Dynamic action space not working"

    # Valid args
    action = action_pred('b2')

    obs, _, _, _ = env.step(action)

    assert obs.literals == frozenset({ pred1('b2'), pred3('b2', 'd1', 'c1'), 
        pred3('a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2') })

    assert isinstance(obs.goal, LiteralConjunction)
    assert set(obs.goal.literals) == {pred2('c2'), pred3('b1', 'c1', 'd1')}

    print("Test passed.")


def test_pddlenv_hierarchical_types():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    domain_file = os.path.join(dir_path, 'pddl', 'hierarchical_type_test_domain.pddl')
    problem_dir = os.path.join(dir_path, 'pddl', 'hierarchical_type_test_domain')

    env = PDDLEnv(domain_file, problem_dir)
    obs, _ = env.reset()

    ispresent = Predicate("ispresent", 1, [Type("entity")])
    islight = Predicate("islight", 1, [Type("object")])
    isfurry = Predicate("isfurry", 1, [Type("animal")])
    ishappy = Predicate("ishappy", 1, [Type("animal")])
    pet = Predicate("pet", 1, [Type("animal")])

    nomsy = Type("jindo")("nomsy")
    rover = Type("corgi")("rover")
    rene = Type("cat")("rene")
    block1 = Type("block")("block1")
    block2 = Type("block")("block2")
    cylinder1 = Type("cylinder")("cylinder1")

    assert obs.literals == frozenset({
        ispresent(nomsy),
        ispresent(rover),
        ispresent(rene),
        ispresent(block1),
        ispresent(block2),
        ispresent(cylinder1),
        islight(block1),
        islight(cylinder1),
        isfurry(nomsy),
    })

    obs, _, _, _ = env.step(pet('block1'))

    assert obs.literals == frozenset({
        ispresent(nomsy),
        ispresent(rover),
        ispresent(rene),
        ispresent(block1),
        ispresent(block2),
        ispresent(cylinder1),
        islight(block1),
        islight(cylinder1),
        isfurry(nomsy),
    })

    obs, _, _, _ = env.step(pet(nomsy))

    assert obs.literals == frozenset({
        ispresent(nomsy),
        ispresent(rover),
        ispresent(rene),
        ispresent(block1),
        ispresent(block2),
        ispresent(cylinder1),
        islight(block1),
        islight(cylinder1),
        isfurry(nomsy),
        ishappy(nomsy)
    })

    print("Test passed.")


def test_get_all_possible_transitions():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    domain_file = os.path.join(
        dir_path, 'pddl', 'test_probabilistic_domain.pddl')
    problem_dir = os.path.join(dir_path, 'pddl', 'test_probabilistic_domain')

    env = PDDLEnv(domain_file, problem_dir, raise_error_on_invalid_action=True,
                  dynamic_action_space=True)

    obs, _ = env.reset()
    action = env.action_space.all_ground_literals(obs).pop()
    transitions = env.get_all_possible_transitions(action)

    transition_list = list(transitions)
    assert len(transition_list) == 2
    state1, state2 = transition_list[0][0], transition_list[1][0]

    type1 = Type('type1')
    type2 = Type('type2')
    pred1 = Predicate('pred1', 1, [type1])
    pred2 = Predicate('pred2', 1, [type2])
    pred3 = Predicate('pred3', 3, [type1, type2, type2])

    state1, state2 = (state1, state2) if pred2(
        'c1') in state2.literals else (state2, state1)

    assert state1.literals == frozenset({pred1('b2'), pred3(
        'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')})
    assert state2.literals == frozenset({pred1('b2'), pred2('c1'), pred3(
        'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')})

    # Now test again with return_probs=True.
    transitions = env.get_all_possible_transitions(action, return_probs=True)

    transition_list = list(transitions)
    assert len(transition_list) == 2
    assert abs(transition_list[0][1]-0.3) < 1e-5 or abs(transition_list[0][1]-0.7) < 1e-5
    assert abs(transition_list[1][1]-0.3) < 1e-5 or abs(transition_list[1][1]-0.7) < 1e-5
    assert abs(transition_list[0][1]-transition_list[1][1]) > 0.3
    state1, state2 = transition_list[0][0][0], transition_list[1][0][0]

    type1 = Type('type1')
    type2 = Type('type2')
    pred1 = Predicate('pred1', 1, [type1])
    pred2 = Predicate('pred2', 1, [type2])
    pred3 = Predicate('pred3', 3, [type1, type2, type2])

    state1, state2 = (state1, state2) if pred2(
        'c1') in state2.literals else (state2, state1)

    assert state1.literals == frozenset({pred1('b2'), pred3(
        'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')})
    assert state2.literals == frozenset({pred1('b2'), pred2('c1'), pred3(
        'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')})

    print("Test passed.")

def test_get_all_possible_transitions_multiple_independent_effects():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    domain_file = os.path.join(
        dir_path, 'pddl', 'test_probabilistic_domain_alt.pddl')
    problem_dir = os.path.join(dir_path, 'pddl', 'test_probabilistic_domain_alt')

    env = PDDLEnv(domain_file, problem_dir, raise_error_on_invalid_action=True,
                  dynamic_action_space=True)

    obs, _ = env.reset()
    action = env.action_space.all_ground_literals(obs).pop()
    transitions = env.get_all_possible_transitions(action)

    transition_list = list(transitions)
    assert len(transition_list) == 4

    states = set({
        transition_list[0][0].literals, transition_list[1][0].literals,
        transition_list[2][0].literals, transition_list[3][0].literals
    })

    type1 = Type('type1')
    type2 = Type('type2')
    pred1 = Predicate('pred1', 1, [type1])
    pred2 = Predicate('pred2', 1, [type2])
    pred3 = Predicate('pred3', 3, [type1, type2, type2])

    expected_states = set({
        frozenset({pred1('b2'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2')}),
        frozenset({pred1('b2'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')}),
        frozenset({pred1('b2'), pred2('c1'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2')}),
        frozenset({pred1('b2'), pred2('c1'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')})
    })

    assert states == expected_states

    # Now test again with return_probs=True.
    transitions = env.get_all_possible_transitions(action, return_probs=True)

    transition_list = list(transitions)
    assert len(transition_list) == 4
    states_and_probs = {
        transition_list[0][0][0].literals: transition_list[0][1],
        transition_list[1][0][0].literals: transition_list[1][1],
        transition_list[2][0][0].literals: transition_list[2][1],
        transition_list[3][0][0].literals: transition_list[3][1]
    }

    type1 = Type('type1')
    type2 = Type('type2')
    pred1 = Predicate('pred1', 1, [type1])
    pred2 = Predicate('pred2', 1, [type2])
    pred3 = Predicate('pred3', 3, [type1, type2, type2])

    expected_states = {
        frozenset({pred1('b2'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2')}): 0.225,
        frozenset({pred1('b2'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')}): 0.075,
        frozenset({pred1('b2'), pred2('c1'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2')}): 0.525,
        frozenset({pred1('b2'), pred2('c1'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')}): 0.175
    }

    for s, prob in states_and_probs.items():
        assert s in expected_states
        assert prob - expected_states[s] < 1e-5

    print("Test passed.")

def test_get_all_possible_transitions_multiple_effects():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    domain_file = os.path.join(
        dir_path, 'pddl', 'test_probabilistic_domain_alt_2.pddl')
    problem_dir = os.path.join(dir_path, 'pddl', 'test_probabilistic_domain_alt_2')

    env = PDDLEnv(domain_file, problem_dir, raise_error_on_invalid_action=True,
                  dynamic_action_space=True)

    obs, _ = env.reset()
    action = env.action_space.all_ground_literals(obs).pop()
    transitions = env.get_all_possible_transitions(action)

    transition_list = list(transitions)
    assert len(transition_list) == 3

    states = set({
        transition_list[0][0].literals,
        transition_list[1][0].literals,
        transition_list[2][0].literals
    })

    type1 = Type('type1')
    type2 = Type('type2')
    pred1 = Predicate('pred1', 1, [type1])
    pred2 = Predicate('pred2', 1, [type2])
    pred3 = Predicate('pred3', 3, [type1, type2, type2])

    expected_states = set({
        frozenset({pred1('b2'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2')}),
        frozenset({pred2('c1'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2')}),
        frozenset({pred1('b2'), pred2('c1'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')})
    })

    assert states == expected_states

    # Now test again with return_probs=True.
    transitions = env.get_all_possible_transitions(action, return_probs=True)

    transition_list = list(transitions)
    assert len(transition_list) == 3
    states_and_probs = {
        transition_list[0][0][0].literals: transition_list[0][1],
        transition_list[1][0][0].literals: transition_list[1][1],
        transition_list[2][0][0].literals: transition_list[2][1]
    }

    type1 = Type('type1')
    type2 = Type('type2')
    pred1 = Predicate('pred1', 1, [type1])
    pred2 = Predicate('pred2', 1, [type2])
    pred3 = Predicate('pred3', 3, [type1, type2, type2])

    expected_states = {
        frozenset({pred1('b2'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2')}): 0.5,
        frozenset({pred2('c1'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2')}): 0.4,
        frozenset({pred1('b2'), pred2('c1'), pred3(
            'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')}): 0.1
    }

    for s, prob in states_and_probs.items():
        assert s in expected_states
        assert prob - expected_states[s] < 1e-5

    print("Test passed.")

def test_determinize():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    domain_file = os.path.join(
        dir_path, 'pddl', 'test_probabilistic_domain.pddl')
    problem_dir = os.path.join(dir_path, 'pddl', 'test_probabilistic_domain')

    env = PDDLEnv(domain_file, problem_dir, raise_error_on_invalid_action=True,
                  dynamic_action_space=True)
    env.domain.determinize()

    obs, _ = env.reset()
    action = env.action_space.all_ground_literals(obs).pop()
    transitions = env.get_all_possible_transitions(action, return_probs=True)

    transition_list = list(transitions)
    assert len(transition_list) == 1
    assert transition_list[0][1] == 1.0
    newstate = transition_list[0][0][0]

    type1 = Type('type1')
    type2 = Type('type2')
    pred1 = Predicate('pred1', 1, [type1])
    pred2 = Predicate('pred2', 1, [type2])
    pred3 = Predicate('pred3', 3, [type1, type2, type2])

    assert newstate.literals == frozenset({pred1('b2'), pred2('c1'), pred3(
        'a1', 'c1', 'd1'), pred3('a2', 'c2', 'd2'), pred3('b2', 'd1', 'c1')})

    print("Test passed.")


if __name__ == "__main__":
    test_pddlenv()
    test_pddlenv_hierarchical_types()
    test_get_all_possible_transitions()
    test_get_all_possible_transitions_multiple_effects()
    test_get_all_possible_transitions_multiple_independent_effects()
    test_determinize()
