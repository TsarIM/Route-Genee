import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from deap import base,creator,tools,algorithms
import mplcyberpunk
from mappingLocs import mapping

st. set_page_config(layout="wide")

def evaluate(route):

  total_distance=0
  truck_distances=[]

  for i in range(n_trucks):
    truck_route = [depot]+[locs[route[j]] for j in range(i,len(route),n_trucks)]+[depot]
    truck_dist=sum(np.linalg.norm(np.array(truck_route[k+1]) - np.array(truck_route[k])) for k in range(len(truck_route)-1))
    total_distance+=truck_dist
    truck_distances.append(truck_dist)

  imbalance=np.std(truck_distances)
  return total_distance,imbalance

def plot_routes(route):
    
    plt.style.use("cyberpunk")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(depot[0], depot[1], 'gs')
    ax.text(depot[0], depot[1], "Depot", fontsize=12, ha='left')

    for loc_name in locations:
        x, y = mapping[loc_name]
        ax.plot(x, y, 'ro')
        ax.text(x, y, loc_name, fontsize=12, ha='left')

    for i in range(n_trucks):
        truck_route = [depot] + [locs[route[j]] for j in range(i, len(route), n_trucks)] + [depot]
        ax.plot(*zip(*truck_route), '-')

    mplcyberpunk.add_glow_effects()
    ax.set_title('Optimized Routes')
    ax.set_xlabel('x-coordinates')
    ax.set_ylabel('y-coordinates')
    st.pyplot(fig)
  
def run_ga(generations,pop_size,cxpb,mutpb,bool_value):
  random.seed(96)#for reproducibility
  pop=toolbox.population(n=pop_size)
  optimal=tools.HallOfFame(1)#to get the best individual
  algorithms.eaSimple(pop,toolbox,cxpb,mutpb,generations,halloffame=optimal,verbose=bool_value)
  plot_routes(optimal[0])
  optimal_paths = [[] for _ in range(n_trucks)]
  for i in range(n_locs):
      truck_idx = i % n_trucks
      loc_idx = optimal[0][i]
      optimal_paths[truck_idx].append(locations[loc_idx])
  
  return optimal_paths

st.title("RouteGenee : Genetic Algorithm Route Optimizer")


col1, col2 = st.columns(2)

with col1:
    st.text("")
    st.text("")
    
    locations = st.multiselect("Choose delivery locations",
            list(mapping.keys())
            )
    st.text("")
    st.text("")
    n_vehicles = st.slider("Pick number of vehicles",1,10)
    st.button("Reset")
     
with col2:
    
    if st.button("Find Routes",type="primary"):
        depot=(47,55)
        locs=[]
        for i in locations:
          locs.append(mapping[i])
          
        n_locs= len(locs)
        if n_locs>=2:
          n_trucks= n_vehicles
            
          toolbox=base.Toolbox()

          creator.create('FitnessMin',base.Fitness,weights=(-1,-1))
          creator.create('Route',list,fitness=creator.FitnessMin)

          toolbox.register("ordering", random.sample, range(n_locs),n_locs) #function to generate orderings of numbers from 0-(n_locs-1) (non repeating elements)
          toolbox.register("route", tools.initIterate, creator.Route, toolbox.ordering)  #function to create an individual-route by shuffling location indeces list
          toolbox.register("population", tools.initRepeat, list, toolbox.route) #function to create a population of routes

          toolbox.register('evaluate',evaluate)
          toolbox.register('mate',tools.cxPartialyMatched)
          toolbox.register('mutate',tools.mutShuffleIndexes,indpb=0.1)
          toolbox.register('select',tools.selTournament,tournsize=3)
          optimal_paths = run_ga(200,300,0.7,0.2,0)
          st.subheader(f"Optimal Routes for all {n_vehicles} vehicle:")
          optimal_paths=pd.DataFrame(optimal_paths)
          st.write(optimal_paths)
          st.text("Each row corresponds to a vehicle's route")

        else:
          st.write('choose more locations')
        
    else:
        st.write("The Vehicle Routing Problem (VRP) involves finding the best routes for a fleet of vehicles to deliver goods or services to customers. Our version of VRP focuses on two main goals: minimizing the total distance traveled and balancing the workload so that no single vehicle gets worn out more than the others. This helps reduce overall travel costs and maintenance expenses while ensuring that all vehicles last longer and remain reliable. Optimizing VRP in logistics is crucial for businesses, as it leads to significant cost savings, better customer service, and a more efficient and sustainable transportation system.")
        st.write("This project focuses on optimizing Vehicle Routes using a genetic algorithm implemented with the DEAP (Distributed Evolutionary Algorithms in Python) library")
        st.subheader("Genetic Algorithm")
        st.write("Genetic algorithm (GA) is an artificial intelligence search method that uses the process of evolution and natural selection theory and is under the umbrella of evolutionary computing algorithms. It is an efficient tool for solving optimization problems.")
        st.write("In a genetic algorithm, a population of candidate solutions (called individuals, creatures, organisms, or phenotypes) to an optimization problem is evolved toward better solutions. Each candidate solution has a set of properties (its chromosomes or genotype) which can be mutated and altered.The evolution usually starts from a population of randomly generated individuals, and is an iterative process, with the population in each iteration called a generation. In each generation, the fitness of every individual in the population is evaluated; the fitness is usually the value of the objective function in the optimization problem being solved. The more fit individuals are stochastically selected from the current population, and each individual's genome is modified (recombined and possibly randomly mutated) to form a new generation. The new generation of candidate solutions is then used in the next iteration of the algorithm. Commonly, the algorithm terminates when either a maximum number of generations has been produced, or a satisfactory fitness level has been reached for the population")
