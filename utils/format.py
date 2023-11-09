import os
import json
import numpy
import re
import torch
import torch_ac
import gym

import utils


def get_obss_preprocessor(obs_space, text=None, dialogue_current=None, dialogue_history=None, custom_image_preprocessor=None, custom_image_space_preprocessor=None):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            assert custom_image_preprocessor is None
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["image"]:

        assert (custom_image_preprocessor is None) == (custom_image_space_preprocessor is None)

        image_obs_space = obs_space.spaces["image"].shape

        if custom_image_preprocessor:
            image_obs_space = custom_image_space_preprocessor(image_obs_space)

        obs_space = {"image": image_obs_space, "text": 100}

        # must be specified in this case
        if text is None:
            raise ValueError("text argument must be specified.")
        if dialogue_current is None:
            raise ValueError("dialogue current argument must be specified.")
        if dialogue_history is None:
            raise ValueError("dialogue history argument must be specified.")

        vocab = Vocabulary(obs_space["text"])
        def preprocess_obss(obss, device=None):
            if custom_image_preprocessor is None:
                D = {
                    "image": preprocess_images([obs["image"] for obs in obss], device=device)
                }
            else:
                D = {
                    "image": custom_image_preprocessor([obs["image"] for obs in obss], device=device)
                }

            if dialogue_current:
                D["utterance"] = preprocess_texts([obs["utterance"] for obs in obss], vocab, device=device)

            if dialogue_history:
                D["utterance_history"] = preprocess_texts([obs["utterance_history"] for obs in obss], vocab, device=device)

            if text:
                D["text"] = preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)


            return torch_ac.DictList(D)

        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss

def ride_ref_image_space_preprocessor(image_space):
    return image_space

def ride_ref_image_preprocessor(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array

    images = numpy.array(images)

    # grid dimensions
    size = images.shape[1]
    assert size == images.shape[2]

    # assert that 1, 2 are absolute cooridnates
    # assert images[:,:,:,1].max() <= size
    # assert images[:,:,:,2].max() <= size
    assert images[:,:,:,1].min() >= 0
    assert images[:,:,:,2].min() >= 0
    #
    # # 0, 1, 2 -> door state
    # assert all([e in set([0, 1, 2]) for e in numpy.unique(images[:, :, :, 4].reshape(-1))])
    #
    # only keep the (obj id, colors, state) -> multiply others by 0
    # print(images[:, :, :, 1].max())

    images[:, :, :, 1] *= 0
    images[:, :, :, 2] *= 0

    assert images.shape[-1] == 5

    return torch.tensor(images, device=device, dtype=torch.float)

def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
