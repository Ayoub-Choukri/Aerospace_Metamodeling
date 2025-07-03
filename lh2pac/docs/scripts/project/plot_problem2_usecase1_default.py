"""
Problem 2 Use Case 1 on x_default
Analyzing the sensitivity of the aircraft design problem 
"""
#%% 
# Importing Librairies

from gemseo.disciplines.auto_py import AutoPyDiscipline

import numpy as np

from gemseo import configure_logger
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo import generate_coupling_graph


from gemseo_oad_training.models import aerodynamic
from gemseo_oad_training.models import approach
from gemseo_oad_training.models import battery
from gemseo_oad_training.models import climb
from gemseo_oad_training.models import engine
from gemseo_oad_training.models import fuel_tank
from gemseo_oad_training.models import geometry
from gemseo_oad_training.models import mass
from gemseo_oad_training.models import mission
from gemseo_oad_training.models import operating_cost
from gemseo_oad_training.models import take_off
from gemseo_oad_training.models import total_mass

from gemseo_oad_training.unit import convert_from
from gemseo_oad_training.unit import convert_to
from numpy import array

from gemseo import from_pickle
from gemseo import sample_disciplines
from gemseo import to_pickle
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.surrogate import SurrogateDiscipline

configure_logger()

#%% 
# Creating The Disciplines
discipline_aerodynamic = AutoPyDiscipline(aerodynamic)
discipline_approach = AutoPyDiscipline(approach)
discipline_battery = AutoPyDiscipline(battery)
discipline_climb = AutoPyDiscipline(climb)
discipline_engine = AutoPyDiscipline(engine)
discipline_fuel_tank = AutoPyDiscipline(fuel_tank)
discipline_geometry = AutoPyDiscipline(geometry)
discipline_mass = AutoPyDiscipline(mass)
discipline_mission = AutoPyDiscipline(mission)
discipline_operating_cost = AutoPyDiscipline(operating_cost)
discipline_take_off = AutoPyDiscipline(take_off)
discipline_total_mass = AutoPyDiscipline(total_mass)

List_Disciplines = [
    discipline_aerodynamic,
    discipline_approach,
    discipline_battery,
    discipline_climb,
    discipline_engine,
    discipline_fuel_tank,
    discipline_geometry,
    discipline_mass,
    discipline_mission,
    discipline_operating_cost,
    discipline_take_off,
    discipline_total_mass
]

GENERATE_GRAPH_DISCIPLINES = True

if GENERATE_GRAPH_DISCIPLINES:
    generate_coupling_graph(List_Disciplines)


for i in range(len(List_Disciplines)):
    print(List_Disciplines[i].input_grammar.defaults)


#%% 
# Creating the surrogate discipline

##  Creating Domain Space for training the surrogate discipline
class MyDesignSpace(DesignSpace):
    def __init__(self):
        super().__init__(name="foo")
        self.add_variable("aef", lower_bound=0.99, upper_bound=1.03, value=1.0)
        self.add_variable("cef", lower_bound=0.99, upper_bound=1.03, value=1.0)
        self.add_variable("sef", lower_bound=0.99, upper_bound=1.03, value=1.0)


DesignSpace_Surrogates = MyDesignSpace()

training_dataset = sample_disciplines(List_Disciplines, DesignSpace_Surrogates, ["mtom", "tofl", "vz", "span","length", "fm", "vapp"], algo_name="OT_MONTE_CARLO", n_samples=20, seed = 38)

test_dataset = sample_disciplines(List_Disciplines, DesignSpace_Surrogates, ["mtom", "tofl", "vz", "span","length", "fm", "vapp"], algo_name="OT_FULLFACT", n_samples=200, seed =318)
to_pickle(training_dataset, "training_dataset.pkl")

# %%  
##  Training the Surrogate Model
# surrogate_discipline = SurrogateDiscipline("MLPRegressor", training_dataset,hidden_layer_sizes = [400,2000,400])
surrogate_discipline = SurrogateDiscipline("RBFRegressor", training_dataset,epsilon = 100)
surrogate_discipline.execute({"aef": array([1.]),
 "cef": array([1.]),
 "sef": array([1.])})


# %% 
## Evaluating the surrohate discipline
r2 = surrogate_discipline.get_error_measure("R2Measure")
print(r2.compute_learning_measure(as_dict=True))
print(r2.compute_test_measure(test_dataset, as_dict=True))


def Plot_R2(Dict_R2_Train, Dict_R2_Test):
    import matplotlib.pyplot as plt
    import numpy as np
    for key in Dict_R2_Train.keys():
        Dict_R2_Train[key] = float(Dict_R2_Train[key])
        Dict_R2_Test[key] = float(Dict_R2_Test[key])
        
    Keys = list(Dict_R2_Train.keys())
    Values_Train = list(Dict_R2_Train.values())
    Values_Test = list(Dict_R2_Test.values())
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    bars_train = axes[0].bar(Keys, Values_Train, color='blue', alpha=0.6, label='Training R2')
    bars_test = axes[1].bar(Keys, Values_Test, color='orange', alpha=0.6, label='Test R2')
    axes[0].set_title('Training R2 Values')
    axes[1].set_title('Test R2 Values')
    axes[0].set_ylabel('R2 Value')
    axes[1].set_ylabel('R2 Value')

    for ax, bars, values in zip(axes, [bars_train, bars_test], [Values_Train, Values_Test]):
        ax.set_xticklabels(Keys, rotation=45, ha='right')
        ax.set_ylim(0, 1.3)
        ax.legend()
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


Plot_R2(r2.compute_learning_measure(as_dict=True), r2.compute_test_measure(test_dataset, as_dict=True))



# Sobol Indices
#%% 
## Creating Uncertainty Space 
from gemseo.algos.parameter_space import ParameterSpace


class MyUncertainSpace(ParameterSpace):
    def __init__(self):
        super().__init__()
        self.add_random_variable("cef", "OTTriangularDistribution", minimum=0.99, mode=1, maximum=1.03)
        self.add_random_variable("aef", "OTTriangularDistribution", minimum=0.99, mode=1, maximum=1.03)
        self.add_random_variable("sef", "OTTriangularDistribution", minimum=0.99, mode=1, maximum=1.03)


Uncertain_space = MyUncertainSpace()

data_simulation = sample_disciplines([surrogate_discipline], Uncertain_space, ["mtom", "tofl", "vz", "span","length", "fm", "vapp"], algo_name="OT_MONTE_CARLO", n_samples=10000, seed = 328)



#%% 
## Visualizing the distribution of the Data

import seaborn as sns
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 3, figsize=(20, 12))
variable_names = ["cef", "sef", "aef", "mtom", "tofl", "vz", "span", "fm", "vapp"]
axes = axes.flatten()
for ax, name in zip(axes, variable_names):
    data = data_simulation.get_view(variable_names=name)
    sns.histplot(data, ax=ax, kde=True, bins=10, color="skyblue", edgecolor="black")
    ax.set_title(name)
    ax.set_xlabel(name)
    ax.set_ylabel('Density')
    ax.grid(True)
plt.tight_layout()
plt.show()


#%% 
## Computing General Statistics
from matplotlib import pyplot as plt

from gemseo import sample_disciplines
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.uncertainty.statistics.empirical_statistics import EmpiricalStatistics

statistics = EmpiricalStatistics(data_simulation, ["mtom", "tofl", "vz", "span", "length", "fm", "vapp"]+["cef","aef","sef"])



## Mean
mean = statistics.compute_mean()

print("Mean Values:")
## Variance 
for key, value in mean.items():
    mean[key] = float(value)  # Convert numpy.float64 to python float for better readability
    print(f"{key}: {mean[key]}")

variance = statistics.compute_variance()

for key, value in variance.items():
    variance[key] = float(value)  # Convert numpy.float64 to python float for better readability
    print(f"{key}: {variance[key]}")


## Coefficient of Variation and IQ

def Coefficient_Variation(mean, var):
    Result={}

    for variable in mean.keys():
        Mean_Variable = mean[variable]
        Std_Variable = np.sqrt(var[variable])
        Coefficient_Variation_Variable = Std_Variable / Mean_Variable if Mean_Variable else 0
        Result[variable] = float(Coefficient_Variation_Variable)
    return Result

def IQ(data_simulation):
    q25 = {}
    q75 = {}
    IQ = {}
    for var in ["mtom", "tofl", "vz", "span", "length", "fm", "vapp"]:
        values = data_simulation.get_view(variable_names=var).values.flatten()
        q25[var] = np.percentile(values, 25)
        q75[var] = np.percentile(values, 75)
        IQ[var] = (q75[var]-q25[var])/(q75[var]+q25[var])
    return IQ

CoVV = Coefficient_Variation(mean, variance)
print(f"Coefficient of Variation: {CoVV}")

IQQ = IQ(data_simulation)
print(f"Interquartile Ratio: {IQQ}")



#%% 
# Plotting the Statistics

def Plot_Means_And_Variance(Dict_Mean, Dict_Variance):
    import matplotlib.pyplot as plt
    import numpy as np

    keys = list(Dict_Mean.keys())
    means = list(Dict_Mean.values())
    variances = list(Dict_Variance.values())

    x = np.arange(len(keys))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, means, width, label='Mean', color='skyblue')
    bars2 = ax.bar(x + width/2, variances, width, label='Variance', color='orange')

    ax.set_ylabel('Values')
    ax.set_title('Means and Variances of Variables')
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.legend()

    plt.tight_layout()
    plt.show()

Plot_Means_And_Variance(mean, variance)


#%% Plotting the Statistics
## Visualizing the data

import matplotlib.pyplot as plt

def Plot_Means_Over_Sd(Dict_Means_Over_Sd, figsize=(10, 6)):
    # Extraire les clés et les valeurs
    Keys = list(Dict_Means_Over_Sd.keys())
    Values = list(Dict_Means_Over_Sd.values())

    # Créer la figure
    plt.figure(figsize=figsize)
    
    # Tracer le bar plot
    plt.bar(Keys, Values, color='skyblue')

    # Ajouter des titres et labels
    plt.title("Coeficients de variation des variables")
    plt.xlabel("Catégories")
    plt.ylabel("Coefficient de variation")

    # Rotation des labels de l'axe x si nécessaire
    plt.xticks(rotation=45)

    # Afficher le graphique
    plt.tight_layout()
    plt.show()



Plot_Means_Over_Sd(CoVV)

#%% Plotting the Statistics
## Visualizing the data


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def Plot_Heatmap_From_Scores(score_dict, Output_Variables, Input_Variables, figsize=(10, 6), title="Heatmap des scores"):
    # Construire une matrice (DataFrame) de taille (len(Output_Variables), len(Input_Variables))
    matrix = []

    for output in Output_Variables:
        row = []
        for input_var in Input_Variables:
            # Calcul du ratio : output/input
            numerator = score_dict.get(output, np.nan)
            denominator = score_dict.get(input_var, np.nan)
            ratio = numerator / denominator if denominator not in [0, np.nan] else np.nan
            row.append(ratio)
        matrix.append(row)
    
    df = pd.DataFrame(matrix, index=Output_Variables, columns=Input_Variables)

    # Tracer la heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Variables d'entrée")
    plt.ylabel("Variables de sortie")
    plt.tight_layout()
    plt.show()

Plot_Heatmap_From_Scores(
    CoVV,
    Output_Variables=["mtom", "tofl", "vz", "span", "length", "fm", "vapp"],
    Input_Variables=["cef", "aef", "sef"],
    title="Heatmap du ratio du coefficient de variation des variables de sortie sur les variables d'entrée"
)


#%% # Sobol Analysis
## Computing Sobol Analysis

import pprint

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis
from numpy import pi


sobol = SobolAnalysis()
sobol.compute_samples([surrogate_discipline], Uncertain_space, 8000)
sobol.compute_indices(output_names=["mtom", "tofl", "vz", "fm", "vapp"])


sobol.compute_indices(output_names=["mtom", "tofl", "vz","span" ,"fm", "vapp"])


pprint.pprint(sobol.indices.first)
pprint.pprint(sobol.indices.total)


#%% 
## Plotting the Sobol Indices

sobol.plot("mtom", save=False, show=True)
plt.show()
sobol.plot("tofl", save=False, show=True)
plt.show()
sobol.plot("vz", save=False, show=True)
plt.show()
sobol.plot("fm", save=False, show=True)
plt.show()
sobol.plot("span", save=False, show=True)
plt.show()
sobol.plot("vapp", save=False, show=True)
plt.show()
