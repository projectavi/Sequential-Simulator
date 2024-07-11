# Are Sequence Modellers Excellent Simulators?

For this project I need to encode each commentary snippet into an embedding to represent the 'state' of the system at that time.

Then using this state, and a sequence of them - we use a sequence modelling technique to run a masked self-supervised learning paradigm to find out how to predict masked states in a sequence of states.

This should allow the sequence modeller to understand the dynamics of the state.

Then this can be applied to predicting the outcome of anything, but primarily my focus is on games.



The first attempt at this will use this paper as a backbone: "Sefl-Supervised Relational Reasoning for Representation Learning", Patacchiola, M., and Storkey, A., *Advances in Neural Information Processing Systems (NeurIPS)
