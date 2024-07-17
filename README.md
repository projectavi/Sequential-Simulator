# Sequential-Simulator

This is a project on testing the capacities of modern sequence modeling algorithms that power the abundance of Large Language Models (LLMs) that are growing in popularity each day.

In this codebase I ask the question: Can we imbue positional embeddings with additional information about the state of some environment that allows it to learn the transition function of an arbitrary environment, and subsequently, can it provide a simulation of that environment under specified conditions?

Currently, the repository contains the code to train a Karpov model, aptly named after Anatoly Karpov, which is a Causal model based off of the Roberta model, with the capacity to sequentially model chess games conditioned on a set of moves already played, and the ratings of the players in the game.