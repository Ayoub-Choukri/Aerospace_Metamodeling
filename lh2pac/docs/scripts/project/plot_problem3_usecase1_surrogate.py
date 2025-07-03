"""
Problem 3 Use Case 1 Iterative Surrogate Method
This script creates one surrogate model for each iteration of the aircraft design problem and optimizes it using a surrogate-based approach.
"""
#%% 
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo import configure_logger
from gemseo.algos.design_space import DesignSpace
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
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario
from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings


configure_logger()
#%% 
#### Creating direct Disciplines
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

Disciplines_direct = [
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

#%%
#### Creating design space and incertain space
class AirCraftDesignSpace(DesignSpace):
    def __init__(self):
        super().__init__(name="foo")
        self.add_variable("slst", lower_bound=100000., upper_bound=200000., value=150000.)
        self.add_variable("n_pax", lower_bound=120., upper_bound=180., value=150.)
        self.add_variable("area", lower_bound=100., upper_bound=200., value=180.)
        self.add_variable("ar", lower_bound=5., upper_bound=20., value=9.)

class AirCraftUncertainSpace(ParameterSpace):
    def __init__(self):
        super().__init__()
        self.add_random_variable("cef", "OTTriangularDistribution", minimum=0.99, mode=1, maximum=1.03)
        self.add_random_variable("aef", "OTTriangularDistribution", minimum=0.99, mode=1, maximum=1.03)
        self.add_random_variable("sef", "OTTriangularDistribution", minimum=0.99, mode=1, maximum=1.03)

fly_uncertain_space = AirCraftUncertainSpace()
fly_desgin_space = AirCraftDesignSpace()


#%%
#### Creating Scenario 

scenario = UMDOScenario(
    Disciplines_direct,
    "mtom",
    fly_desgin_space,
    fly_uncertain_space,
    "Mean",
    formulation_name="DisciplinaryOpt",
    statistic_estimation_settings=Surrogate_Settings(
    n_samples=30)
)

scenario.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900, statistic_name="Margin", factor = 2.0)
scenario.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135), statistic_name="Margin", factor = 2.0)
scenario.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300), statistic_name="Margin", factor = -2.0)
scenario.add_constraint("span", constraint_type="ineq", positive=False, value=40, statistic_name="Margin", factor = 2.0)
scenario.add_constraint("length", constraint_type="ineq", positive=False, value=45, statistic_name="Margin", factor = 2.0)
scenario.add_constraint("fm", constraint_type="ineq", positive=True, value=0, statistic_name="Margin", factor = -2.0)

scenario.execute(algo_name="NLOPT_COBYLA", max_iter=200)
scenario.post_process(post_name="OptHistoryView", save=False, show=False)

#%%
result = scenario.optimization_result
(result.x_opt, result.constraint_values, result.f_opt)



#%%
#### Verifier le contraint
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
scenario_direct_model = UMDOScenario(
    Disciplines_direct,
    "mtom",
    fly_desgin_space,
    fly_uncertain_space,
    "Mean",
    formulation_name="DisciplinaryOpt",
    statistic_estimation_settings=Sampling_Settings()
)

scenario_direct_model.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900, statistic_name="Margin", factor = 2.0)
scenario_direct_model.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135), statistic_name="Margin", factor = 2.0)
scenario_direct_model.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300), statistic_name="Margin", factor = -2.0)
scenario_direct_model.add_constraint("span", constraint_type="ineq", positive=False, value=40, statistic_name="Margin", factor = 2.0)
scenario_direct_model.add_constraint("length", constraint_type="ineq", positive=False, value=45, statistic_name="Margin", factor = 2.0)
scenario_direct_model.add_constraint("fm", constraint_type="ineq", positive=True, value=0, statistic_name="Margin", factor = -2.0)

import numpy as np
input = result.x_opt_as_dict
print(input)
input_array = np.array([input["slst"], input["n_pax"], input["area"], input["ar"]])
print(input_array)
scenario_direct_model.execute(algo_name = "CustomDOE", samples = input_array.reshape(1,4))


#%% Visualize the aircraft optimized by surrogate model vs. true optimized model
slst, n_pax, area, ar = result.x_opt
ls = [slst, n_pax, area, ar]
for i in ls: 
    print(i)

from gemseo_oad_training.utils import AircraftConfiguration, draw_aircraft

config_opt = AircraftConfiguration(
    name="Exact", 
    area=109.18, 
    length= 45-12.9987477067113612, 
    n_pax=120, 
    slst=100024.611,
    color="blue"
)

config_opt2 = AircraftConfiguration(
    name="Opt_iterative_Surrogate", 
    area=area, 
    length=32, 
    n_pax=np.round(n_pax), 
    slst=slst,
    color="gray"
)
draw_aircraft(config_opt, config_opt2)
#%%
