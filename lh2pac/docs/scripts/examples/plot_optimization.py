r"""
# Solve an optimization problem

We want to minimize the Rosenbrock function $f(x,y)=(1-x)^2+100(y-x**2)^2$
over the domain $[-2,2]^2$.
"""
from gemseo import configure_logger
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.scenarios.mdo_scenario import MDOScenario

# %%
# Before starting,
# we activate the logger as an optimization process logs meaningful information.
configure_logger()

# %%
# Firstly,
# we define the discipline computing the Rosenbrock function
# and the Euclidean distance to the optimum:
discipline = AnalyticDiscipline(
    {"z": "(1-x)**2+100*(y-x**2)**2", "c": "((x-1)**2+(y-1)**2)**0.5"},
    name="Rosenbrock"
)

# %%
# Then, we create the design space:
design_space = DesignSpace()
design_space.add_variable("x", lower_bound=-2., upper_bound=2., value=0.)
design_space.add_variable("y", lower_bound=-2., upper_bound=2., value=0.)

# %%
# Thirdly,
# we put these elements together in a scenario
# to minimize the Rosenbrock function
# under the constraint that the distance
# between the design point and the solution of the unconstrained problem
# is greater or equal to 1.
scenario = MDOScenario([discipline], "z", design_space, formulation_name="DisciplinaryOpt")
scenario.add_constraint("c", constraint_type="ineq", positive=True, value=1.)

# %%
# !!! note
#
#     GEMSEO is a Python library
#     dedicated to multidisciplinary design optimization (MDO)
#     based on the notion of MDO formulation.
#     This is why the second positional argument `formulation` is mandatory.
#     But when using the scenario with a unique discipline,
#     don't bother and consider `"DisciplinaryOpt"`.

# %%
# before executing it with a gradient-free optimizer:
scenario.execute(algo_name="NLOPT_COBYLA", max_iter=100)

# %%
# Lastly,
# we can plot the optimization history:
scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# !!! seealso "More than one discipline"
#     If you have more than one discipline,
#     the MDF formulation could be helpful.
#     You can have a look to [this exemple](https://gemseo.readthedocs.io/en/stable/examples/formulations/plot_sobieski_mdf_example.html#sphx-glr-examples-formulations-plot-sobieski-mdf-example-py).