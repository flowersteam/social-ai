from gym_minigrid.minigrid import COLOR_TO_IDX, OBJECT_TO_IDX

def generate_text_obs(obs, info):

    text_observation = obs_to_text(info)

    llm_prompt = "Obs : "
    llm_prompt += "".join(text_observation)

    # add utterances
    if obs["utterance_history"] != "Conversation: \n":
        utt_hist = obs['utterance_history']
        utt_hist = utt_hist.replace("Conversation: \n","")
        llm_prompt += utt_hist

    return llm_prompt

def obs_to_text(info):
    image, vis_mask = info["image"], info["vis_mask"]
    carrying = info["carrying"]
    agent_pos_vx, agent_pos_vy = info["agent_pos_vx"], info["agent_pos_vy"]
    npc_actions_dict = info["npc_actions_dict"]

    # (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)
    # State, 0: open, 1: closed, 2: locked
    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

    list_textual_descriptions = []

    if carrying is not None:
        list_textual_descriptions.append("You carry a {} {}".format(carrying.color, carrying.type))

    # agent_pos_vx, agent_pos_vy = self.get_view_coords(self.agent_pos[0], self.agent_pos[1])

    view_field_dictionary = dict()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] != 0 and image[i][j][0] != 1 and image[i][j][0] != 2:
                if i not in view_field_dictionary.keys():
                    view_field_dictionary[i] = dict()
                    view_field_dictionary[i][j] = image[i][j]
                else:
                    view_field_dictionary[i][j] = image[i][j]

    # Find the wall if any
    #  We describe a wall only if there is no objects between the agent and the wall in straight line

    # Find wall in front
    add_wall_descr = False
    if add_wall_descr:
        j = agent_pos_vy - 1
        object_seen = False
        while j >= 0 and not object_seen:
            if image[agent_pos_vx][j][0] != 0 and image[agent_pos_vx][j][0] != 1:
                if image[agent_pos_vx][j][0] == 2:
                    list_textual_descriptions.append(
                        f"A wall is {agent_pos_vy - j} steps in front of you. \n")  # forward
                    object_seen = True
                else:
                    object_seen = True
            j -= 1
        # Find wall left
        i = agent_pos_vx - 1
        object_seen = False
        while i >= 0 and not object_seen:
            if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
                if image[i][agent_pos_vy][0] == 2:
                    list_textual_descriptions.append(
                        f"A wall is {agent_pos_vx - i} steps to the left. \n")  # left
                    object_seen = True
                else:
                    object_seen = True
            i -= 1
        # Find wall right
        i = agent_pos_vx + 1
        object_seen = False
        while i < image.shape[0] and not object_seen:
            if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
                if image[i][agent_pos_vy][0] == 2:
                    list_textual_descriptions.append(
                        f"A wall is {i - agent_pos_vx} steps to the right. \n")  # right
                    object_seen = True
                else:
                    object_seen = True
            i += 1

    # list_textual_descriptions.append("You see the following objects: ")
    # returns the position of seen objects relative to you
    for i in view_field_dictionary.keys():
        for j in view_field_dictionary[i].keys():
            if i != agent_pos_vx or j != agent_pos_vy:
                object = view_field_dictionary[i][j]

                # # don't show npc
                # if IDX_TO_OBJECT[object[0]] == "npc":
                #     continue

                front_dist = agent_pos_vy - j
                left_right_dist = i - agent_pos_vx

                loc_descr = ""
                if front_dist == 1 and left_right_dist == 0:
                    loc_descr += "Right in front of you "

                elif left_right_dist == 1 and front_dist == 0:
                    loc_descr += "Just to the right of you"

                elif left_right_dist == -1 and front_dist == 0:
                    loc_descr += "Just to the left of you"

                else:
                    front_str = str(front_dist) + " steps in front of you " if front_dist > 0 else ""

                    loc_descr += front_str

                    suff = "s" if abs(left_right_dist) > 0 else ""
                    and_ = "and" if loc_descr != "" else ""

                    if left_right_dist < 0:
                        left_right_str = f"{and_} {-left_right_dist} step{suff} to the left"
                        loc_descr += left_right_str

                    elif left_right_dist > 0:
                        left_right_str = f"{and_} {left_right_dist} step{suff} to the right"
                        loc_descr += left_right_str

                    else:
                        left_right_str = ""
                        loc_descr += left_right_str

                loc_descr += f" there is a "

                obj_type = IDX_TO_OBJECT[object[0]]
                if obj_type == "npc":
                    IDX_TO_STATE = {0: 'friendly', 1: 'antagonistic'}

                    description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} peer. "

                    # gaze
                    gaze_dir = {
                        0: "towards you",
                        1: "to the left of you",
                        2: "in the same direction as you",
                        3: "to the right of you",
                    }
                    description += f"It is looking {gaze_dir[object[3]]}. "

                    # point
                    point_dir = {
                        0: "towards you",
                        1: "to the left of you",
                        2: "in the same direction as you",
                        3: "to the right of you",
                    }

                    if object[4] != 255:
                        description += f"It is pointing {point_dir[object[4]]}. "

                    # last action
                    last_action = {v: k for k, v in npc_actions_dict.items()}[object[5]]

                    last_action = {
                        "go_forward": "foward",
                        "rotate_left": "turn left",
                        "rotate_right": "turn right",
                        "toggle_action": "toggle",
                        "point_stop_point": "stop pointing",
                        "point_E": "",
                        "point_S": "",
                        "point_W": "",
                        "point_N": "",
                        "stop_point": "stop pointing",
                        "no_op": ""
                    }[last_action]

                    if last_action not in ["no_op", ""]:
                        description += f"It's last action is {last_action}. "

                elif obj_type in ["switch", "apple", "generatorplatform", "marble", "marbletee", "fence"]:
                    # todo: this assumes that Switch.no_light == True
                    description = f"{IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                    assert object[2:].mean() == 0

                elif obj_type == "lockablebox":
                    IDX_TO_STATE = {0: 'open', 1: 'closed', 2: 'locked'}
                    description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                    assert object[3:].mean() == 0

                elif obj_type == "applegenerator":
                    IDX_TO_STATE = {1: 'square', 2: 'round'}
                    description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                    assert object[3:].mean() == 0

                elif obj_type == "remotedoor":
                    IDX_TO_STATE = {0: 'open', 1: 'closed'}
                    description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                    assert object[3:].mean() == 0

                elif obj_type == "door":
                    IDX_TO_STATE = {0: 'open', 1: 'closed', 2: 'locked'}
                    description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                    assert object[3:].mean() == 0

                elif obj_type == "lever":
                    IDX_TO_STATE = {1: 'activated', 0: 'unactivated'}
                    if object[3] == 255:
                        countdown_txt = ""
                    else:
                        countdown_txt = f"with {object[3]} timesteps left. "

                    description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} {countdown_txt}"

                    assert object[4:].mean() == 0
                else:
                    raise ValueError(f"Undefined object type {obj_type}")

                full_destr = loc_descr + description + "\n"

                list_textual_descriptions.append(full_destr)

    if len(list_textual_descriptions) == 0:
        list_textual_descriptions.append("\n")

    return list_textual_descriptions
