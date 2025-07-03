"""
Problem 1 Use Case 2
Plotting the results of the optimization problem
"""
#%%
### Importing the necessary libraries

from numpy import array

from gemseo import from_pickle
from gemseo import sample_disciplines
from gemseo import to_pickle
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.surrogate import SurrogateDiscipline
from gemseo.disciplines.auto_py import AutoPyDiscipline


from gemseo import configure_logger
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo import generate_coupling_graph

from gemseo_oad_training.unit import convert_from
from gemseo_oad_training.unit import convert_to
import numpy as np


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

from lh2pac.utils import update_default_inputs

configure_logger()
#%%
Dict_Parametres_To_Change = {}

Dict_Parametres_To_Change["fuel_type"] = "liquid_h2"
Dict_Parametres_To_Change["engine_type"] = "turbofan"


#%%
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

update_default_inputs(List_Disciplines, Dict_Parametres_To_Change)

for i in range(len(List_Disciplines)):
    print(List_Disciplines[i].input_grammar.defaults)


#%% Creating the design space

class MyDesignSpace(DesignSpace):
    def __init__(self):
        super().__init__(name="foo")
        self.add_variable("slst", lower_bound=100000., upper_bound=200000., value=150000.)
        self.add_variable("n_pax", lower_bound=120., upper_bound=180., value=150.)
        self.add_variable("area", lower_bound=100., upper_bound=200., value=180.)
        self.add_variable("ar", lower_bound=5., upper_bound=20., value=9.)

Design_Space = MyDesignSpace()

#%%Creation the testing and training datasets
## Creation des training and testing datasets


training_dataset = sample_disciplines(List_Disciplines, Design_Space, ["mtom"]+["tofl","vapp","vz","span","length","fm"], algo_name="OT_OPT_LHS", n_samples=20, seed=42)

test_dataset = sample_disciplines(List_Disciplines, Design_Space, ["mtom"]+["tofl","vapp","vz","span","length","fm"], algo_name="OT_OPT_LHS", n_samples=110, seed=43)


#%%Creation of the surrogate model

surrogate_discipline = SurrogateDiscipline("MOERegressor", training_dataset)
surrogate_discipline.execute({"slst": np.array([150000.]),
                             "n_pax": np.array([150.]),
                             "area": np.array([180.]),
                             "ar": np.array([9.])})

r2 = surrogate_discipline.get_error_measure("R2Measure")

#%%
r2.compute_learning_measure(as_dict=True)
r2.compute_test_measure(test_dataset, as_dict=True)

#%%
rmse = surrogate_discipline.get_error_measure("RMSEMeasure")
rmse.compute_learning_measure(as_dict=True)

#%%
rmse.compute_test_measure(test_dataset, as_dict=True)

#%%
## Optimization of the surrogate model

eps_pource = 0
scenario = MDOScenario([surrogate_discipline],"mtom", Design_Space, formulation_name="DisciplinaryOpt")

scenario.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900 -eps_pource * 1900)
scenario.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135) - eps_pource * convert_from("kt", 135))
scenario.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300) + eps_pource * convert_from("ft/min", 300))
scenario.add_constraint("span", constraint_type="ineq", positive=False, value=40 - eps_pource * 40)
scenario.add_constraint("length", constraint_type="ineq", positive=False, value=45 - eps_pource * 45)
scenario.add_constraint("fm", constraint_type="ineq", positive=True, value=0+ eps_pource * 0)

scenario.execute(algo_name="NLOPT_COBYLA", max_iter=200)

#%%
scenario.optimization_result

#%%
scenario.post_process(post_name = "OptHistoryView", save=False, show=False)
input = scenario.optimization_result.x_opt_as_dict
print(input)


#%%
## Verification of the respect of the constraints
Scenario_Direct_Model = MDOScenario(List_Disciplines, "mtom", Design_Space, formulation_name="MDF")

Scenario_Direct_Model.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900)
Scenario_Direct_Model.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135))
Scenario_Direct_Model.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300))
Scenario_Direct_Model.add_constraint("span", constraint_type="ineq", positive=False, value=40)
Scenario_Direct_Model.add_constraint("length", constraint_type="ineq", positive=False, value=45)
Scenario_Direct_Model.add_constraint("fm", constraint_type="ineq", positive=True, value=0)

slst, n_pax, area, ar = scenario.optimization_result.x_opt

Scenario_Direct_Model.execute(algo_name="CustomDOE",samples = np.array([[slst, n_pax, area, ar]]))

#%%
Scenario_Direct_Model.post_process(post_name = "OptHistoryView", save=False, show=False)


#%%
## Displaying the different aircraft configurations
from gemseo_oad_training.utils import draw_aircraft, AircraftConfiguration

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
scenario_direct_model = MDOScenario(List_Disciplines,"mtom", Design_Space, formulation_name="MDF")

scenario_direct_model.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900)
scenario_direct_model.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135))
scenario_direct_model.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300))
scenario_direct_model.add_constraint("span", constraint_type="ineq", positive=False, value=40)
scenario_direct_model.add_constraint("length", constraint_type="ineq", positive=False, value=45)
scenario_direct_model.add_constraint("fm", constraint_type="ineq", positive=True, value=0)

scenario_direct_model.execute(algo_name="NLOPT_COBYLA", max_iter=50)

#%% Post-processing the direct model results
scenario_direct_model.post_process(post_name="OptHistoryView", save=False, show=False)
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