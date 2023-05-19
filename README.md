# Lightweight fine-tuning of Language model for micro tasks

This repository contains the code to replicate the results found in **paper name**. 
To generate the results run the Jupyter notebook in the root directory. 

## Summary
In this paper we examine the use of Hopfield networks augmenting the last layers of a Large Language model. 
We evaluate the ability of this network to remember new information about a very specific set of information.
Only the Hopfield layers are fine-tuned. There is a two-step fine-tuning process. The model is fine-tuned on a set
of movie reviews from multiple users. We randomly select a subset of users to use as a testing set. For each user
we create a new instance of the Hopfield Layers. We then fine-tune these Hopfield layers on only the one user. To test 
the model's ability to remember the ratings of movies from each user.

## Model Training

The model training was preformed in two parts. We used the pretrained GPT2 model from [Hugging Face](
https://huggingface.co/). We augment this model with our Hopfield Layers in the final layers. This hurts the performance,
so we retrain the model on a subset of the openwebtext that GPT2 was originally trained on. 

The next step of training was to train it as a Chat Bot. To do this we fine-tune the model on Open Assistant's 
dataset also found on Hugging Face. We create a custom set up of the dataset. Currently, it is stored in trees. We 
flatten the dataset such that each full conversation is a single entry. We trained for 16 epochs.

We find that this second round of training helps the small GPT2 model to have good perfomance as a Chat Bot when 
evaluated by a human.

## Usage

The model will be uploaded to 