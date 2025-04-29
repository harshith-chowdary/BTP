input_file = 'rm_hoa.txt'  # Input file containing LTL formulas and rewards
output_file_txt = 'rm.txt'  # Output file for the RM in txt format

# Read the entire file input_file as a string
with open(input_file, 'r', encoding='utf-8') as file:
    rm_hoa = file.read()

rm_txt = ''
rm_dict = {}
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
            try:
                visit_state_reward = int(lst[2].replace('{', '').replace('}',''))
            except:
                visit_state_reward = int(lst[3].replace('{', '').replace('}',''))
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