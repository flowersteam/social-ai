import streamlit as st
import copy
import streamlit.components.v1 as components
import streamlit.caching as caching
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import matplotlib.pyplot as plt
from gym_minigrid.social_ai_envs.socialaigrammar import SocialAIGrammar, SocialAIActions, SocialAIActionSpace

default_params = {
    "Pointing": 0,
    "Emulation": 1,
    "Language_grounding": 2,
    "Pragmatic_frame_complexity": 1,
}

class InteractiveACL:

    def choose(self, node, chosen_parameters):

        options = [n.label for n in node.children]

        box_name = f'{node.label} ({node.id})'
        ret = st.sidebar.selectbox(
            box_name,
            options,
            index=default_params.get(node.label, 0)
        )

        for ind, (c, c_lab) in enumerate(zip(node.children, options)):
            if c_lab == ret:
                return c

    def get_info(self):
        return {}

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_env():
    env = gym.make("SocialAI-SocialAIParamEnv-v1")
    env.curriculum=InteractiveACL()

    return env




st.title("SocialAI interactive demo")


env = load_env()

st.subheader("Primitive actions")

# moving buttons
columns = st.columns([1]*(len(SocialAIActions)+1))
action_names = [a.name for a in list(SocialAIActions)] + ["no_op"]
# keys = ["Left arrow", "Right arrow", "Up arrow", "t", "q", "Shift"]
keys = ["a", "d", "w", "t", "q", "Shift"]

# actions = [st.button(a.name) for a in list(SocialAIActions)] + [st.button("none")]
actions = []
for a_name, col, key in zip(action_names, columns, keys):
    with col:
        actions.append(st.button(a_name+f" ({key})", help=f"Shortcut: {key}"))


st.subheader("Speaking actions")
# talking buttons
t, w, b = st.columns([1, 1, 1])

changes = [False, False]

with t:
    templ = st.selectbox("Template", options=SocialAIGrammar.templates, index=1)
with w:
    word = st.selectbox("Word", options=SocialAIGrammar.things, index=0)

speak = st.button("Speak (s)", help="Shortcut s")

# utterance change detection
utt_changed = False

if "template" in st.session_state:
    utt_changed = st.session_state.template != templ

if "word" in st.session_state:
    utt_changed = utt_changed or st.session_state.word != word

st.session_state["template"] = templ
st.session_state["word"] = word

st.sidebar.subheader("Select the parameters:")

play = st.button("Play (Enter)", help="Generate the env. Shortcut: Enter")

components.html(
    """
<script>
const doc = window.parent.document;
buttons = Array.from(doc.querySelectorAll('button[kind=primary]'));

const left_button = buttons.find(el => el.innerText === 'left (a)');
const right_button = buttons.find(el => el.innerText === 'right (d)');
const forward_button = buttons.find(el => el.innerText === 'forward (w)');
const toggle_button = buttons.find(el => el.innerText === 'toggle (t)');
const none_button = buttons.find(el => el.innerText === 'no_op (Shift)');
const done_button = buttons.find(el => el.innerText === 'done (q)');
const play_button = buttons.find(el => el.innerText === 'Play (Enter)');
const speak_button = buttons.find(el => el.innerText === 'Speak (s)');

doc.addEventListener('keydown', function(e) {
switch (e.keyCode) {
    case 65: // (65 = a )
        left_button.click();
        break;
    case 68: // (68 = d )
        right_button.click();
        break;
    case 87: // (87 = w )
        forward_button.click();
        break;
    case 84: // (84 = t)
        toggle_button.click();
        break;
    case 16: // (16 = shift)
        none_button.click();
        break;
    case 81: // (81 = q)
        done_button.click();
        break;
    case 13: // (13 = enter)
        play_button.click();
        break;
    case 83: // (83 = s)
        speak_button.click();
        break;
}

});
</script>
""",
    height=0,
    width=0,
)

# no action
done_ind = len(actions) - 2
actions[done_ind] = False

# was agent controlled
no_action = not any(actions) and not speak

done = False
info = None

if not no_action or play or utt_changed:
    # agent is controlled
    if any(actions):
        p_act = np.argmax(actions)
        if p_act == len(actions) - 1:
            p_act = np.nan

        action = [p_act, np.nan, np.nan]

    elif speak:
        templ_ind = SocialAIGrammar.templates.index(templ)
        word_ind = SocialAIGrammar.things.index(word)
        action = [np.nan, templ_ind, word_ind]

    else:
        action = None

    if action:
        obs, reward, done, info = env.step(action)

    env.render(mode='human')
    st.pyplot(env.window.fig)


# if done or no_action:
if done or (no_action and not play and not utt_changed):
    env.reset()

else:
    env.parameter_tree.sample_env_params(ACL=env.curriculum)


with st.expander("Parametric tree", True):
    # draw tree
    current_param_labels = env.current_env.parameters if env.current_env.parameters else {}
    folded_nodes = [
        "Information_seeking",
        "Collaboration",
        "OthersPerceptionInference"
    ]
    # print(current_param_labels["Env_type"])
    folded_nodes.remove(current_param_labels["Env_type"])
    env.parameter_tree.draw_tree(
        filename="viz/streamlit_temp_tree",
        ignore_labels=["Num_of_colors"],
        selected_parameters=current_param_labels,
        folded_nodes=folded_nodes,
        # save=False
    )
    # st.graphviz_chart(env.parameter_tree.tree)
    st.image("viz/streamlit_temp_tree.png")

# if not no_action or play or utt_changed:
#     # agent is controlled
#     if any(actions):
#         p_act = np.argmax(actions)
#         if p_act == len(actions) - 1:
#             p_act = np.nan
#
#         action = [p_act, np.nan, np.nan]
#
#     elif speak:
#         templ_ind = SocialAIGrammar.templates.index(templ)
#         word_ind = SocialAIGrammar.things.index(word)
#         action = [np.nan, templ_ind, word_ind]
#
#     else:
#         action = None
#
#     if action:
#         obs, reward, done, info = env.step(action)
#
#     env.render(mode='human')
#     st.pyplot(env.window.fig)
