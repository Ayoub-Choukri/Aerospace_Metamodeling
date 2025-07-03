"""
Problem 3 Use Case 2 PolyTaylor Method
This script creates surrogate models for the aircraft design problem and optimizes it with the Taylor Expression Method
"""
#%% Import libraries and useful packages
from gemseo.disciplines.surrogate import SurrogateDiscipline
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.algos.design_space import DesignSpace
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
import numpy as np

from gemseo import configure_logger
from gemseo import generate_coupling_graph
from gemseo import sample_disciplines

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

configure_logger()


#%% Create disciplines 

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
generate_coupling_graph(List_Disciplines)

#%% Create design space and uncertain space 
class MyDesignSpace(DesignSpace):
    def __init__(self):
        super().__init__(name="DOE")
        self.add_variable("slst", lower_bound=100000.0, upper_bound=200000.0, value=150000.0)
        self.add_variable("n_pax", lower_bound=120, upper_bound=180, value=150)
        self.add_variable("area", lower_bound=100.0, upper_bound=200.0, value=180.0)
        self.add_variable("ar", lower_bound=5.0, upper_bound=20.0, value=9.0)

class MyUncertainSpace(ParameterSpace):
    def __init__(self):
        super().__init__(name="MDO")
        self.add_random_variable(
            "gi", "OTTriangularDistribution", minimum=0.35, mode=0.4, maximum=0.405
        )
        self.add_random_variable(
            "vi", "OTTriangularDistribution", minimum=0.755, mode=0.8, maximum=0.805
        )
        self.add_random_variable(
            "aef", "OTTriangularDistribution", minimum=0.99, mode=1., maximum=1.03
        )
        self.add_random_variable(
            "cef", "OTTriangularDistribution", minimum=0.99, mode=1., maximum=1.03
        )
        self.add_random_variable(
            "sef", "OTTriangularDistribution", minimum=0.99, mode=1, maximum=1.03
        )


uncertain_space = MyUncertainSpace()
design_space = MyDesignSpace()
design_space, uncertain_space

#%% Create scenario 

from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings
from gemseo_umdo.formulations.taylor_polynomial_settings import TaylorPolynomial_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario
from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)

settings = Surrogate_Settings(n_samples=30)

scenario1 = UMDOScenario(
    List_Disciplines, 
    ["mtom"], 
    design_space, 
    uncertain_space, 
    "Mean", 
    formulation_name="DisciplinaryOpt", 
    statistic_estimation_settings=TaylorPolynomial_Settings(), 
)

def add_constraint_to_scenario(scenario):
    scenario.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900, statistic_name="Margin", factor = 2.0)
    scenario.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135), statistic_name="Margin", factor = 2.0)
    scenario.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300), statistic_name="Margin", factor = 2.0)
    scenario.add_constraint("span", constraint_type="ineq", positive=False, value=40, statistic_name="Margin", factor = 2.0)
    scenario.add_constraint("length", constraint_type="ineq", positive=False, value=45, statistic_name="Margin", factor = 2.0)
    scenario.add_constraint("fm", constraint_type="ineq", positive=True, value=0, statistic_name="Margin", factor = 2.0)
    return scenario

scenario1 = add_constraint_to_scenario(scenario1)
scenario1.execute(algo_name="NLOPT_COBYLA", max_iter=200)

#%% Check resultats and optimisation evolution 
result1 = scenario1.optimization_result
scenario1.post_process(post_name="OptHistoryView", save=False, show=False)

#%% Check the constraints 
scenario_direct_model = UMDOScenario(
    List_Disciplines,
    "mtom",
    design_space,
    uncertain_space,
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

input = result1.x_opt_as_dict
print(input)
input_array = np.array([input["slst"], input["n_pax"], input["area"], input["ar"]])
print(input_array)
scenario_direct_model.execute(algo_name = "CustomDOE", samples = input_array.reshape(1,4))
#%% Visualize the aircraft optimized by surrogate model vs. true optimized model
slst1, n_pax1, area1, ar1 = result1.x_opt

ls = [slst1, n_pax1, area1, ar1 ]
for i in ls: 
    print(i)

from gemseo_oad_training.utils import AircraftConfiguration, draw_aircraft

config_opt = AircraftConfiguration(
    name="Exact", 
    area=111.335, 
    length= 45-9.1047, 
    n_pax=127, 
    slst=145902.985,
    color="blue"
)
config_opt1 = AircraftConfiguration(
    name="Opt_Taylor", 
    area=area1, 
    length=32, 
    n_pax=np.round(n_pax1), 
    slst=slst1,
    color="red"
)

draw_aircraft(config_opt, config_opt1)
#%% 