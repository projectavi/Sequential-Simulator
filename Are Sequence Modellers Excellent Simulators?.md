# Are Sequence Modellers Excellent Simulators?

For this project I need to encode each commentary snippet into an embedding to represent the 'state' of the system at that time.

Then using this state, and a sequence of them - we use a sequence modelling technique to run a masked self-supervised learning paradigm (maybe) or time series forcasting to find out how to predict masked states in a sequence of states.

This should allow the sequence modeller to understand the dynamics of the state.

Then this can be applied to predicting the outcome of anything, but primarily my focus is on games.

In this process, we need the sequence of states to have a masked dataloader - so that a single or multiple states together get masked out.

Keep note that player names will have to be separately endocoded using a vocabulary to avoid different players with the same name having similar embeddings. We can use a hash function for this - where there should theoretically be no noticeable pattern and hence each player will be treated as their own separate entity.

Using TimesFM to sequence model these games. This will require fine-tuning on the simulation dataset: [timesfm/notebooks/finetuning.ipynb at master · google-research/timesfm · GitHub](https://github.com/google-research/timesfm/blob/master/notebooks/finetuning.ipynb)

Masked Modeling Self-Supervised Learning

https://github.com/Lupin1998/Awesome-MIM?tab=readme-ov-file#mim-for-transformers

If TimesFM doesn't work altogether or needs adjustment I can train a masked language/sequence model for this task.

Now the biggest issue I can see here would be in the decoder architecture. The state encodings that we provide are not discrete and thus we cannot output a distribution over all of the discrete states. What we require is a continuous distribution modeled over the state space.

Look into how to make a decode output a continuous probability distribution. Or consider that the number of possibilities for these states is finite in some sense, but will have to be computed. Ie. all the possibilities will have to be computed and assigned as potential outputs - this feels infeasible and it wont transfer to new tasks.

The only way to work with a discrete set of states would be to ensure that the state is composable. This can be done. Each player name and team name will be renamed to a numbering and there will be some assumptions input about that player so that we can get player specific information. This will be standard cricket metrics? at the time.

- We can compose the state space by having the decoder have multiple output heads which each provide a discrete distribution over the possibilities for different components
  
  - Batsman, 2nd Batsman, Bowler, Runs Scored, Wicket? - for cricket

- For the first instance of the project we can discard player specific and maybe team specific information just to get a simulation

- For this task, we will adjust each player name into a simple numbering according to their appearance in the lineup

[GitHub - RitaRamo/remote-sensing-images-caption](https://github.com/RitaRamo/remote-sensing-images-caption/tree/master) This repository contains code for a paper that produces continuous output neural encoder-decoder models. They do this by targeting to generate sequences of tokens. This could be a stand in for the entire architecture and could work well for this purpose.

We could also just do a masked sequence modeling instance where the output is the $d$ dimensional state representation and use the MSE loss.

This is another possibility: https://boyuan.space/diffusion-forcing/

- Improvements on using diffusion models for sequence modelling

Consider Mamba/S4 - [GitHub - state-spaces/mamba: Mamba SSM architecture](https://github.com/state-spaces/mamba)

[Generative Marginalization Models](https://proceedings.mlr.press/v235/liu24az.html)

- This is an interesting paper that makes me think I need to look into AutoRegressive models more for this type of sequential modeling - specifically rhose that ahve been good for protein modeling

- It requires that within the $D$ dimensions of the input vector $x$, each of the $D$ dimensions has a discrete set of values. I wonder if chess can be transformed to this.
  
  - Something like [piece, colour, original position, ending position, piece captured]

- The work here seems to mask out portions of the input $x$ and predict the marginals of the non-masked by summing over the masked
  
  - since these are used for protein synthesis, its possible that the $x$ vector is the sequence itself - in which case this could be good for state space simulation such as this





Initial tests with a reworked version of RoBERTa had some success. Modelled a game for a bit and then lost consistency. EIther undertraining or the model isn't powerful enough.



2 things to try:

- Train more (seems to overfit after 900 ish steps)

- Change how the ratings are added as conditionals to the embeddings
  
  - Idea: add them in somewhere else so the moves are embedded conditional on the ratings of the players
  
  - Idea: transform the summed embeddings through one more layer



So i trained more to 1200 steps and the simulator played a coherent game but the conditionals on the ratings did not work very well as an 800 rating white player demolished a 1700 rating black player.



I need to change how the ratings are added for conditionals. I can try it for the encoder soon but first I will try changing how the embeddings get conditioned on the player ratings.

- Making the embeddings conditional on the player ratings may not be the best call because it means that the actual sequences of vectors modelled depend on the ratings rather than which sequence is chosen next which is why it would be better to put it in the encoder - but i will try the embeddings one more time
