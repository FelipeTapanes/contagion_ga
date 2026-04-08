# Ireland Contagion Optimiser

A network-based contagion model using SIR dynamics and multi-objective optimisation (NSGA-II and SPEA2).

## Requirements

- Python 3.9–3.11
- pip

## Features

- Disease spread modelling
- Marketing diffusion modelling
- Pareto front optimisation
- Interactive Streamlit UI

## How to Run (use the following commands in terminal)
```bash

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

```

## Overview

The application is designed to figure out the most cost effective way to intervene in a disease spread. The intervention types are represented by percentage reductions in the different aspects of the spread in each counties. how these reductions are done are not defined so the cost is a number that is relative to the different solutions. For example: a 50 percent reduction in 2 counties is more expensive than a 25 percent reduction.

the intervention method is optimised using genetic algorithms. these are the Non-Dominated Sorting Genetic Algorithm 2 (NSGA2) and the Strength Pareto Evolutionary Algorithm (SPEA 2). There is also an experimental marketing solution where the idea is to treat the contagion spread as marketing and the intervention is to maximise the amount of time that the disease remains. The created app allows the user to change the variables of the disease spread (the SIR model) and the Genetic Algorithms. further details about these variables are provided below. check the app section to see a brief explanations on how the app works without the background detail.

## Scripts
This folder contains the scripts to get the distance data. The scripts uses an api call to openroute service, providing the longitude and latitude information for the 32 county capitals in ireland which then returns a matrix on their distances. From this an edge matrix is created using minutes as the travel distance for a more realistic depiction of spread (traveling 25km through rural is not the same as 25km through motorways). these can be changed to seconds or km in code but for this case minutes were used.

The towns are not intended to be fully interconnected so a k nearest neighbour node choice was used where k = 4 in this case. what this means is that we only save the 4 closest towns to each town in a network
these are not used after the data is collected and the network is created.

## Data
this is where the data files and the network is stored.

## SRC
This is the backend of the application. This is Where the SIR model is run and the Genetic algorithms

### SIR
this is the model used for the simulation of a disease spread. it is broken down into 3 aspects. the S which is the people susceptible to the disease, the I which is the people who are infected and the R which is the people who have recovered. the idea is that S becomes I which becomes R. from S to I we have 2 varables called beta and tau. beta is the infection rate within a town and tau is the incoming infection rates from other towns. gamma represents the the infection to recovery rate. the intervention methods are used to improve these by lowering beta and tau and increasing gamma. these are rates of improvement in each county. the model also takes in a starting location and a starting number of people infected.

### GA
these are the genetic algorithms. they will create a list of intervention solutions at random and then improve on them using genetic algorithms. there are a lot of variables here which can be checked in the code description but the main idea is the following: population of potential solutions which are called individuals, are scored and the best solutions combine to create a new solution. these new solutions represent the new population and can mutate which means that they change slightly.

This is done over and over until a set of the best solutions through all populations are obtained. This set is called the pareto front. the issue is there is no one best solution because both intervention cost and infection are trying to be minimised and there is always a trade off. In the case of marketing the infection is trying to be maximised. the difference between the two genetic algorithms is that the solutions are scored using different techniques.

## The App
First is deciding if it is being used for marketing or is it being used for disease. Then the variables need to be selected. the default variables are there if the user just wants to run the app and test it.

Disease model:

- beta: rate of spread within a town
- gamma: rate of recovery
- tau: rate of spread between towns
- simulation days: the number of days the model will simulate
- seed county: where the infection starts
- starting infected number: the number of infected people in the seed county

GA Model:

- choose which algorithm to run
- population size: choose the number of interventions that will be tested in each run. (larger is better but careful because its a computationally expensive model due to needing to simulate the SIR model for each individual)
- generations: the number of times the ga will run (careful like above)
- seed number: the algorithm creates random initial population. the seed number ensures that the randomly generated population can be the same for reproducibility and testing. changing the number would give a different randomly produced population.

the result is a graph that shows the initial spread compared to a spread with a selected solution. below that is a graph with the pareto front (which are the set of best possible solutions) and the currently selected solution in the front. to the right of that is the solution itself which would be the percentage reduction of beta and tau and the percentage increase of gamma in each county to achieve the desired results. for marketing it is reversed (increase - decrease)

## Future Work
There is a lot of potential future work for this model. The accuracy of the interventions could be improved by making them more specific types, each with costs on different scales. The accuracy of the model itself could also be improved by including more towns, adding incubation periods, or modelling disease mutation, which could allow recovered individuals to become infected again. Other algorithms could also be tested, or the model could be run with larger variable sets using a more powerful computer.




