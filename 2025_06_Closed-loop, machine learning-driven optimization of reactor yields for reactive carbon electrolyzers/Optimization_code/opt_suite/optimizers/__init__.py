import importlib

OPTIMISERS = {
    "Ax": 'opt_ax',
    "BayesianOptimizer": 'opt_bayesian_optimizer',
    "Botorch": 'opt_botorch',
    "EHVI": 'opt_ehvi',
    "Grid": 'opt_grid',
    "Laniakea": 'opt_skgp',
    "List": 'opt_list',
    "NelderMead": "opt_nelder_mead",
    "Phoenics": 'opt_phoenics',
    "Random": 'opt_random',
    "SKForest": 'opt_skforest',
    "SKGP": 'opt_skgp',
    "Sobol": 'opt_sobol',
}


def get_optimizer(optimizer):
    """ returns the desired Optimizer class given by name, desirable to avoid importing all optimizers """
    opt_file = importlib.import_module(f"opt_suite.optimizers.{OPTIMISERS[optimizer]}")
    opt_class = getattr(opt_file, optimizer)
    return opt_class
