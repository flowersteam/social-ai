from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify
from PIL import Image
import io
import base64
import time

import gym
import gym_minigrid
import numpy as np
from gym_minigrid.window import Window

import os

app = Flask(__name__)

env_types = ["Information_seeking", "Collaboration", "AppleStealing"]

env_label_to_env_name = {
    "Full SocialAI environment": "SocialAI-SocialAIParamEnv-v1",  # all
    "Pointing (Train)": "SocialAI-EPointingHeldoutDoorsTrainInformationSeekingParamEnv-v1",  # Pointing Train
    "Pointing (Test)": "SocialAI-EPointingBoxesTestInformationSeekingParamEnv-v1",  # Pointing Test
    "Role Reversal Single Role B (Pretrain - experimental)": "SocialAI-MarblePassBCollaborationParamEnv-v1",
    "Role Reversal Single Asocial (Pretrain - control)": "SocialAI-AsocialMarbleCollaborationParamEnv-v1",
    "Role Reversal Group Role B (Pretrain - experimental)": "SocialAI-RoleReversalGroupExperimentalCollaborationParamEnv-v1",
    "Role Reversal Group Asocial (Pretrain - control)": "SocialAI-RoleReversalGroupControlCollaborationParamEnv-v1",
    "Role Reversal Role A (Finetune - test)": "SocialAI-MarblePassACollaborationParamEnv-v1",
    "Imitation (Train)": "SocialAI-EEmulationNoDistrInformationSeekingParamEnv-v1",
    "Imitation (Test)": "SocialAI-EEmulationNoDistrDoorsInformationSeekingParamEnv-v1",
    "Language Color (Train)": "SocialAI-ELangColorHeldoutDoorsTrainInformationSeekingParamEnv-v1",
    "Language Color (Test)": "SocialAI-ELangColorDoorsTestInformationSeekingParamEnv-v1",
    "Language Feedback (Train)": "SocialAI-ELangFeedbackHeldoutDoorsTrainInformationSeekingParamEnv-v1",
    "Language Feedback (Test)": "SocialAI-ELangFeedbackDoorsTestInformationSeekingParamEnv-v1",
    "Joint Attention Language Color (Train)": "SocialAI-ELangColorHeldoutDoorsTrainInformationSeekingParamEnv-v1",
    "Joint Attention Language Color (Test)": "SocialAI-ELangColorDoorsTestInformationSeekingParamEnv-v1",
    "Apple stealing": "SocialAI-AppleStealingObst_NoParamEnv-v1",
    "Apple stealing (Occlusions)": "SocialAI-AppleStealingObst_MediumParamEnv-v1",
    "AsocialBox (textworld)": "SocialAI-AsocialBoxInformationSeekingParamEnv-v1",
    "ColorBoxes (textworld)": "SocialAI-ColorBoxesLLMCSParamEnv-v1",
    "Scaffolding (train - scaf_8: Phase 1)": "SocialAI-AELangFeedbackTrainScaffoldingCSParamEnv-v1",
    "Scaffolding/Formats (test)":"SocialAI-AELangFeedbackTrainFormatsCSParamEnv-v1",
}

# env = gym.make(args.env, **env_args_str_to_dict(args.env_args))
global env_name
global env_label
env_label = list(env_label_to_env_name.keys())[0]
env_name = env_label_to_env_name[env_label]

global mask_unobserved
mask_unobserved = False

env = gym.make(env_name)

def update_tree():
    selected_parameters = env.current_env.parameters
    selected_env_type = selected_parameters["Env_type"]

    assert selected_env_type in env_types, f"Env_type {selected_env_type} not in {env_types}"

    folded_nodes = [e for e in env_types if e  != selected_env_type]

    env.parameter_tree.draw_tree(
        filename="./web_demo/static/current_tree",
        ignore_labels=["Num_of_colors"],
        selected_parameters=selected_parameters,
        folded_nodes=folded_nodes

    )

update_tree()


def np_img_to_base64(np_image):
    image = Image.fromarray(np_image)
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode('utf-8')


def format_bubble_text(text):
    lines = text.split("\n")

    if len(lines) > 10:
        # Keep the first line, add "....", and then append the last 8 lines
        lines = [lines[0], "...."] + lines[-8:]

    return "\n".join(lines)


@app.route('/set_env', methods=['POST'])
def set_env():
    global env_name  # Declare the variable as global to modify it
    global env_label  # Declare the variable as global to modify it
    env_label = request.form.get('env_label')  # Get the selected env_name from the form

    env_name = env_label_to_env_name[env_label]

    global env  # Declare the env variable as global to modify it
    env = gym.make(env_name)  # Initialize the environment with the new name
    update_tree()  # Update the tree for the new environment
    return redirect(url_for('index'))  # Redirect back to the main page


@app.route('/set_mask_unobserved', methods=['POST'])
def set_mask_unobserved():
    global mask_unobserved
    mask_unobserved_value = request.form.get('mask_unobserved')
    mask_unobserved = bool(mask_unobserved_value)

    return redirect(url_for('index'))



@app.route('/update_image', methods=['POST'])
def update_image():
    action_name = request.form.get('action')


    if action_name == 'done':
        # reset the env and update the tree image
        obs = env.reset()
        update_tree()

    else:
        if action_name == "speak":
            action_template = request.form.get('template')
            action_word = request.form.get('word')

            temp_ind, word_ind = env.grammar.get_action(action_template, action_word)
            action = [np.nan, temp_ind, word_ind]

        elif action_name == 'left':
            action = [int(env.actions.left), np.nan, np.nan]
        elif action_name == 'right':
            action = [int(env.actions.right), np.nan, np.nan]
        elif action_name == 'forward':
            action = [int(env.actions.forward), np.nan, np.nan]
        elif action_name == 'toggle':
            action = [int(env.actions.toggle), np.nan, np.nan]
        elif action_name == 'noop':
            action = [np.nan, np.nan, np.nan]
        else:
            action = [np.nan, np.nan, np.nan]

        obs, reward, done, info = env.step(action)

    image = env.render('rgb_array', tile_size=32, mask_unobserved=mask_unobserved)
    image_data = np_img_to_base64(image)


    bubble_text = format_bubble_text(env.current_env.full_conversation)

    return jsonify({'image_data': image_data, "bubble_text": bubble_text})


@app.route('/', methods=['GET', 'POST'])
def index():
    image = env.render('rgb_array', tile_size=32, mask_unobserved=mask_unobserved)
    image_data = np_img_to_base64(image)

    bubble_text = format_bubble_text(env.current_env.full_conversation)

    available_env_labels = env_label_to_env_name.keys()

    grammar_templates = env.grammar.templates
    grammar_words = env.grammar.things

    return render_template(
        'index.html',
        image_data=image_data,
        bubble_text=bubble_text,
        mask_unobserved=mask_unobserved,
        timestamp=time.time(),
        available_env_labels=available_env_labels,
        current_env_label=env_label,
        grammar_templates=grammar_templates,
        grammar_words=grammar_words,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
