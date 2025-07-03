"""
Problem 3 Use Case 2 Direct Surrogate Method
This script creates ONE surrogate model for the aircraft design problem and optimizes it with the sampling-based approach
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

class SurrogateDesignSpace(DesignSpace):
    def __init__(self):
        super().__init__(name="foo")
        self.add_variable("slst", lower_bound=100000., upper_bound=200000., value=150000.)
        self.add_variable("n_pax", lower_bound=120., upper_bound=180., value=150.)
        self.add_variable("area", lower_bound=100., upper_bound=200., value=180.)
        self.add_variable("ar", lower_bound=5., upper_bound=20., value=9.)
        self.add_variable("gi", lower_bound=0.35, value=0.4, upper_bound=0.405)
        self.add_variable("vi", lower_bound=0.755, value=0.8, upper_bound=0.805)
        self.add_variable("cef", lower_bound=0.99, value=1, upper_bound=1.03)
        self.add_variable("aef", lower_bound=0.99, value=1, upper_bound=1.03)
        self.add_variable("sef",  lower_bound=0.99, value=1, upper_bound=1.03)


uncertain_space = MyUncertainSpace()
design_space = MyDesignSpace()
surrogate_space = SurrogateDesignSpace()
design_space, uncertain_space, surrogate_space


#%% Creating big surrogate model
training_dataset = sample_disciplines(List_Disciplines, surrogate_space, ["mtom", "tofl", "vz", "span","length", "fm", "vapp"], algo_name="OT_OPT_LHS", n_samples=30)
test_dataset = sample_disciplines(List_Disciplines, surrogate_space, ["mtom", "tofl", "vz", "span","length", "fm", "vapp"], algo_name="OT_OPT_LHS", n_samples=100)
surrogate_discipline = SurrogateDiscipline("GaussianProcessRegressor", training_dataset)

#%% Show r2 erreur
r2 = surrogate_discipline.get_error_measure("R2Measure")
r2.compute_learning_measure(as_dict=True)
r2.compute_test_measure(test_dataset, as_dict=True)

#%% Create scenario 

from gemseo_umdo.formulations.surrogate_settings import Surrogate_Settings
from gemseo_umdo.formulations.taylor_polynomial_settings import TaylorPolynomial_Settings
from gemseo_umdo.scenarios.umdo_scenario import UMDOScenario
from gemseo_umdo.formulations.sequential_sampling_settings import (
    SequentialSampling_Settings,
)

settings = Surrogate_Settings(n_samples=30)

scenario2 = UMDOScenario(
    [surrogate_discipline], 
    ["mtom"], 
    design_space, 
    uncertain_space, 
    "Mean", 
    formulation_name="DisciplinaryOpt", 
    # statistic_estimation_settings=TaylorPolynomial_Settings(), 
    # statistic_estimation_settings=settings, 
    statistic_estimation_settings=Sampling_Settings(), 
    # statistic_estimation_settings=SequentialSampling_Settings(
    #     n_samples=50,
    #     initial_n_samples=20,
    #     n_samples_increment=5,
    # ),
)

def add_constraint_to_scenario(scenario):
    scenario.add_constraint("tofl", constraint_type="ineq", positive=False, value=1900, statistic_name="Margin", factor = 2.0)
    scenario.add_constraint("vapp", constraint_type="ineq", positive=False, value=convert_from("kt", 135), statistic_name="Margin", factor = 2.0)
    scenario.add_constraint("vz", constraint_type="ineq", positive=True, value=convert_from("ft/min", 300), statistic_name="Margin", factor = -2.0)
    scenario.add_constraint("span", constraint_type="ineq", positive=False, value=40, statistic_name="Margin", factor = 2.0)
    scenario.add_constraint("length", constraint_type="ineq", positive=False, value=45, statistic_name="Margin", factor = 2.0)
    scenario.add_constraint("fm", constraint_type="ineq", positive=True, value=0, statistic_name="Margin", factor = -2.0)
    return scenario

scenario2 = add_constraint_to_scenario(scenario2)

scenario2.execute(algo_name="NLOPT_COBYLA", max_iter=200)

#%% Check resultats and optimisation evolution 

result2 = scenario2.optimization_result
(result2.x_opt, result2.constraint_values, result2.f_opt)
scenario2.post_process(post_name="OptHistoryView", save=False, show=False)


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

import numpy as np
input = result2.x_opt_as_dict
print(input)
input_array = np.array([input["slst"], input["n_pax"], input["area"], input["ar"]])
print(input_array)
scenario_direct_model.execute(algo_name = "CustomDOE", samples = input_array.reshape(1,4))


#%% 
##### We find that the tofl constraint is not satisfied, so we will not plot the results here.