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