"""
Problem 1 Use Case 1
Plotting the problem with the OAD training models
"""
#%%
###Importing the necessary libraries
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

#%% Creating the disciplines from the OAD training models

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


#%% Plotting the coupling graph of the disciplines

generate_coupling_graph(List_Disciplines)


#%% Creating the design space

class MyDesignSpace(DesignSpace):
    def __init__(self):
        super().__init__(name="foo")
        self.add_variable("slst", lower_bound=100000., upper_bound=200000., value=150000.)
        self.add_variable("n_pax", lower_bound=120., upper_bound=180., value=150.)
        self.add_variable("area", lower_bound=100., upper_bound=200., value=180.)
        self.add_variable("ar", lower_bound=5., upper_bound=20., value=9.)


fly_test_space = MyDesignSpace()
fly_test_space



#%% Creating the training and test datasets

training_dataset = sample_disciplines(List_Disciplines, fly_test_space, ["mtom", "tofl", "vz", "span","length", "fm", "vapp"], algo_name="OT_OPT_LHS", n_samples=20)
test_dataset = sample_disciplines(List_Disciplines, fly_test_space, ["mtom", "tofl", "vz", "span","length", "fm", "vapp"], algo_name="OT_FULLFACT", n_samples=100)

#%% Creating the surrogate discipline and executing it

surrogate_discipline = SurrogateDiscipline("RBFRegressor", training_dataset)
surrogate_discipline.execute({"ar": array([5]),
 "area": array([100]),
 "n_pax": array([120]), "slst": array([100000])})

r2 = surrogate_discipline.get_error_measure("R2Measure")
print(r2.compute_learning_measure(as_dict=True))

#%%
print(r2.compute_test_measure(test_dataset, as_dict=True))

#%%
rmse = surrogate_discipline.get_error_measure("RMSEMeasure")
rmse.compute_learning_measure(as_dict=True)

#%%
rmse.compute_test_measure(test_dataset, as_dict=True)


#%% Creating the MDO scenario from the surrogate and executing it

scenario = MDOScenario([surrogate_discipline],"mtom", fly_test_space, formulation_name="MDF")

scenario.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900)
scenario.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135))
scenario.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300))
scenario.add_constraint("span", constraint_type="ineq", positive=False, value=40)
scenario.add_constraint("length", constraint_type="ineq", positive=False, value=45)
scenario.add_constraint("fm", constraint_type="ineq", positive=True, value=0)

scenario.execute(algo_name="NLOPT_COBYLA", max_iter=200)


#%% Plotting the optimization history and displaying the results

scenario.post_process(post_name="OptHistoryView", save=False, show=True)

input = scenario.optimization_result.x_opt_as_dict
print(input)


#%% Creating the input array for the direct model and executing it to verify the results

input_array = array([input["slst"], input["n_pax"], input["area"], input["ar"]])
input_array = input_array.reshape(1, -1)

scenario_direct_model = MDOScenario(List_Disciplines,"mtom", fly_test_space, formulation_name="MDF")

scenario_direct_model.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900)
scenario_direct_model.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135))
scenario_direct_model.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300))
scenario_direct_model.add_constraint("span", constraint_type="ineq", positive=False, value=40)
scenario_direct_model.add_constraint("length", constraint_type="ineq", positive=False, value=45)
scenario_direct_model.add_constraint("fm", constraint_type="ineq", positive=True, value=0)

scenario_direct_model.execute(algo_name = "CustomDOE", samples = input_array)

#%% Displaying the differents aircrafts

from gemseo_oad_training.utils import draw_aircraft, AircraftConfiguration

area = input["area"]
ar = input["ar"]
length = np.sqrt(ar * area)

configDefault = AircraftConfiguration(
    name="Default",
    area=180,
    ar=9,
    length=np.sqrt(9*180),
    n_pax=150,
    slst=150000,
    color="blue",)

configOptimized = AircraftConfiguration(
    name="Optimized",
    area=area,
    ar=ar,
    length=length,
    n_pax=120,
    slst=100000,
    color="red",)

draw_aircraft(configDefault, configOptimized)


#%% Resoudre le probleme avec le modele direct

# On optimise aussi le probleme avec le modele direct. On veut voir si le surrogate nous donne vraiment une solution proche de la solution du modele direct. 
scenario_direct_model = MDOScenario(List_Disciplines,"mtom", fly_test_space, formulation_name="MDF")

scenario_direct_model.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900)
scenario_direct_model.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135))
scenario_direct_model.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300))
scenario_direct_model.add_constraint("span", constraint_type="ineq", positive=False, value=40)
scenario_direct_model.add_constraint("length", constraint_type="ineq", positive=False, value=45)
scenario_direct_model.add_constraint("fm", constraint_type="ineq", positive=True, value=0)

scenario_direct_model.execute(algo_name="NLOPT_COBYLA", max_iter=50)

#%% Post-processing the direct model results
scenario_direct_model.post_process(post_name="OptHistoryView", save=False, show=True)
#%% Displaying the results of the direct model
input_direct = scenario_direct_model.optimization_result.x_opt_as_dict
print(input_direct)
#%% Displaying the differents aircrafts with the direct model results
configDirect = AircraftConfiguration(
    name="Direct",
    area=input_direct["area"],
    ar=input_direct["ar"],
    length=np.sqrt(input_direct["ar"] * input_direct["area"]),
    n_pax=int(input_direct["n_pax"]),
    slst=input_direct["slst"],
    color="green",)
draw_aircraft(configOptimized, configDirect)

#%%
draw_aircraft(configOptimized, configDirect, configDefault)