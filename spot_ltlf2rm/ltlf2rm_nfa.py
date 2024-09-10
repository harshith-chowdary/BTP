import spot
import json
import buddy
import sys
import itertools

def ltlf_to_nfa(ltlf_formula):
    # modified from https://spot.lrde.epita.fr/tut12.html
    aut = spot.from_ltlf(ltlf_formula).translate('ba')
    # Remove "alive" atomic proposition
    rem = spot.remove_ap()
    rem.add_ap('alive')
    aut = rem.strip(aut)
    # Simplify result and print it. Use postprocess('ba', 'det')
    # if you always want a deterministic automaton.
    aut = spot.postprocess(aut, 'ba')
    # print(aut.to_str('hoa'))
    return aut

def terf_product(automata, rewards):
    # based on the automata product from https://spot.lrde.epita.fr/ipynb/product.html
    bdict = automata[0].get_dict()
    for automaton in automata:
        if automaton.get_dict() != bdict:
            raise RuntimeError("automata should share their dictionary")
        
    result = spot.make_twa_graph(bdict)
    # Copy the atomic propositions of the two input automata
    for automaton in automata:
        result.copy_ap_of(automaton)
    
    sdict = {}
    todo = []
    def dst(state_numbers):
        state_numbers_tuple = tuple(state_numbers)
        p = sdict.get(state_numbers_tuple)
        if p is None:
            p = result.new_state()
            sdict[state_numbers_tuple] = p
            todo.append((state_numbers_tuple, p))
        return p
    
    result.set_init_state(dst( [automata[k].get_init_state_number() for k in range(len(automata)) ]  ))
    
    while todo:
        tuple_rc, osrc = todo.pop()
        # cartesian product of automata[i].out(tuple_rc[i])
        lists_of_transitions = [ automata[i].out(tuple_rc[i]) for i in range(len(automata))  ]
        for element in itertools.product(*lists_of_transitions):
            cond = element[0].cond
            for j in range(1,len(element)):
                cond = cond & element[j].cond
            
            if cond != buddy.bddfalse:
                reward = sum( [rewards[k] for k in range(len(element)) if element[k].acc ])
                acc = spot.mark_t([reward])
                dest_numbers = [ element[k].dst for k in range(len(element)) ]
                result.new_edge(osrc, dst(dest_numbers), cond, acc)
    return result


class TERF_Parser():
    def __init__(self):
        self.ltlf_formulas = []
        self.rewards = []
    def parse_terf(self,terf):
        with open(terf,"r") as f:
            n_formulas = int(f.readline())
            for n in range(n_formulas):
                self.ltlf_formulas.append( f.readline().strip() )
                self.rewards.append( int(f.readline()) )


# STEP 0: Configure the output file (HOA, txt, json)
output_file_hoa = './out/rm' # output file in HOA format
output_file_txt = './out/rm.txt' # output file in txt format
output_file_json = './out/rm.json' # output file in txt format


# STEP 1: Input LTL formulas and rewards
# ltl_list = ['F (a & F (b & F c))'] # input all your LTL formulas in the list
# reward_list = [1]  # input the reward value for each LTL formula

ltl_list = ['F (a & F (b & F c))', 'F (f & g)'] 
reward_list = [1, 2]

# STEP 2: Transform LTLf formulas into nfa
automata = [ltlf_to_nfa(ltlf_formula) for ltlf_formula in ltl_list]

# STEP 3: Do the cross product automaton
product_nfa = terf_product(automata, reward_list)
# rm_hoa = product_nfa.to_str('hoa', 't')
rm_hoa = product_nfa.to_str('hoa') # HOA to string

# STEP 5: Dump HOA to output file
with open(output_file_hoa,"w") as f:
    f.write(rm_hoa) # transition-based output

# STEP 6: Dump to txt file that can be directly used by
# https://github.com/RodrigoToroIcarte/reward_machines for reinforcement learning algorithms with reward machines.
rm_txt = ''
rm_dict = {}
rm_dict['LTLs'] = ltl_list
rm_dict['Rewards'] = reward_list
rm_dict['Start'] = 0
rm_dict['Terminals'] = []
rm_dict['Transitions'] = []
hoa_split = rm_hoa.splitlines()
enter_body = False
state_visit_id = -1
for idx, line in enumerate(hoa_split):
    lst = line.split()
    if lst[0] == 'States:':
        number_states = int(lst[1]) # record number of states
        state_reward_list = [0]*number_states
        rm_dict['States'] = number_states
    if lst[0] == 'Start:':
        rm_dict['Start'] = int(lst[1]) # record initial state id
    if lst[0] == 'AP:':
        number_AP = int(lst[1]) # record number of APs
        AP_list = []
        for ap in lst[2:]:
            AP_list.append(ap.replace('\"','')) # record the list of symbal for AP
        rm_dict['AP'] = AP_list
    if lst[0] == '--BODY--':
        enter_body = True
    if enter_body:
        if lst[0] == 'State:':
            visit_state_id = int(lst[1])
            visit_state_reward = int(lst[2].replace('{', '').replace('}',''))
            state_reward_list[visit_state_id] = visit_state_reward
        if '[' in lst[0]:
            if 't' in lst[0]:
                rm_dict['Terminals'].append(visit_state_id)
                # continue
            next_state_id = int(lst[-1])
            letter = ''
            for s in lst[:-1]: letter += s
            letter = letter.replace('[', '').replace(']','')
            # print(letter)
            for i, ap in enumerate(AP_list):
                letter = letter.replace(str(i), ap)
            # print(letter)
            if letter == 't':
                letter = 'True'
            transition = [visit_state_id, next_state_id, letter]
            rm_dict['Transitions'].append(transition)
    if lst[0] == '--END--':
        enter_body = False

rm_txt += str(rm_dict['Start']) + ' # initial state' + '\n' \
        + str(rm_dict['Terminals']).replace(' ', '') + ' # terminal state'
for transition in rm_dict['Transitions']:
    if transition[2] == 'True':
        transition.append(0)
    else:
        transition.append(state_reward_list[transition[1]]) # add reward
    rm_txt += '\n'
    rm_txt += '(' + \
               str(transition[0]) + ','  + \
               str(transition[1]) + ',' + \
               '\'' + transition[2] + '\'' + ','  + \
               'ConstantRewardFunction({})'.format(transition[3]) + ')'

with open(output_file_txt, 'w') as f:
    f.write(rm_txt)

# STEP 7: Dump to json file
with open(output_file_json, "w") as outfile:
    json.dump(rm_dict, outfile, indent = 4)


# print the reward machine in HOA, txt, and Python dict, respectively
print('-----The reward machine in HOA-----')
print(rm_hoa)
print('\n')
print('-----The reward machine in txt-----')
print(rm_txt)
print('\n')
print('-----The reward machine in Python dictionary (json)-----')
print(rm_dict) # you can also get the reward machine in Python dict if needed
