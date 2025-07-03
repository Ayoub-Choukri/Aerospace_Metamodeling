"""
Problem 3 Use Case 1 Direct Surrogate Method
This script creates ONE surrogate model for the aircraft design problem and optimizes it using a sampling-based approach.
"""
#%% 
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
from gemseo.algos.parameter_space import ParameterSpace
from gemseo_umdo.formulations.sampling_settings import Sampling_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario


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
class SurrogateDesignSpace(DesignSpace):
    def __init__(self):
        super().__init__(name="foo")
        self.add_variable("slst", lower_bound=100000., upper_bound=200000., value=150000.)
        self.add_variable("n_pax", lower_bound=120., upper_bound=180., value=150.)
        self.add_variable("area", lower_bound=100., upper_bound=200., value=180.)
        self.add_variable("ar", lower_bound=5., upper_bound=20., value=9.)
        self.add_variable("cef", lower_bound=0.99, value=1, upper_bound=1.03)
        self.add_variable("aef", lower_bound=0.99, value=1, upper_bound=1.03)
        self.add_variable("sef",  lower_bound=0.99, value=1, upper_bound=1.03)

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

surrogate_desgin_space = SurrogateDesignSpace()
fly_uncertain_space = AirCraftUncertainSpace()
fly_desgin_space = AirCraftDesignSpace()
#%%
#### Creating big surrogate model
training_dataset = sample_disciplines(Disciplines_direct, surrogate_desgin_space, ["mtom", "tofl", "vz", "span","length", "fm", "vapp"], algo_name="OT_OPT_LHS", n_samples=30)
test_dataset = sample_disciplines(Disciplines_direct, surrogate_desgin_space, ["mtom", "tofl", "vz", "span","length", "fm", "vapp"], algo_name="OT_OPT_LHS", n_samples=100)
surrogate_discipline = SurrogateDiscipline("MLPRegressor", training_dataset,hidden_layer_sizes=(100,200,500,200,100))

#%%
#### Show r2 erreur
r2 = surrogate_discipline.get_error_measure("R2Measure")
r2.compute_learning_measure(as_dict=True)
r2.compute_test_measure(test_dataset, as_dict=True)
print(r2.compute_test_measure(test_dataset, as_dict=True))

#%%
#### Creating Scenario 

scenario = UMDOScenario(
    [surrogate_discipline],
    "mtom",
    fly_desgin_space,
    fly_uncertain_space,
    "Mean",
    formulation_name="DisciplinaryOpt",
    statistic_estimation_settings=Sampling_Settings()
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

#%% 
##### We find that the tofl constraint is not satisfied, so we will not plot the results here.