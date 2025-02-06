
# Stock Market Forecaster

Analysing past stock market price fluctuation can allow for more informed investments. This project utilises a Genetic Algorithm paired with a Neural Network (along with different "Techncal Indicators") to measure asset price movement.

The Genetic Algorithm runs (by default) 500 generations to optimise a portfolio for a given list of pre-downloaded stock assets. The given defaults are Amazon. Apple, Google, Microsoft and Sony.

The portfolio is stored as an array, where each index represents the % of an investment to each stock asset. This is a calculated "optimised" portfolio.


This is then fed into a Neural Network, where the optimised portfolio contributes to the modelling of the price movement.



## Demo

The terminal contains a series of commands, allowing for quick and easy experimentation with parameters. 

This could include changing the number of generations for the Genetic Algorithm to run, or the activation function for the Neural Network. 

To see more information for a specific command, simply run "help <command>" (e.g. "help gens").

A command can also be executed with no arguments, this will display it's current value.


#### Changing the "gens" parameter

![help_demo](https://github.com/user-attachments/assets/68070523-20ad-4a3b-8d61-1334c4316f9a)


#### Running the model

![run_demo-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/d6f71752-9617-4bc1-ab6e-795b9e1a5546)

## Output Data

The results are stored in the "plots" folder, each asset will be given it's own image (PNG). 

A graph (made with matplotlib) will display the actual and predicted price movement.

Here are some examples (note that the model was run with default settings):


![Screenshot from 2025-02-06 18-53-21](https://github.com/user-attachments/assets/7cb09d59-68d5-4f30-b768-bfbeec0721e4)
![Screenshot from 2025-02-06 18-52-56](https://github.com/user-attachments/assets/bb200bc3-8061-4d25-b8e9-3ac414d6f7ea)
![Screenshot from 2025-02-06 18-53-04](https://github.com/user-attachments/assets/2033ce47-4899-4926-85a3-cac02ca46dd6)
![Screenshot from 2025-02-06 18-53-09](https://github.com/user-attachments/assets/d1596bde-932c-44f7-81b8-2c60a58cd82b)
![Screenshot from 2025-02-06 18-53-15](https://github.com/user-attachments/assets/5e8607da-1358-4df7-aa8d-b579a1d561ea)
