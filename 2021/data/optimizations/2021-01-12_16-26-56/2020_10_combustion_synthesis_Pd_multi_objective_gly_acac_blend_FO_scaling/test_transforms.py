import importlib
transformations_config = importlib.import_module("configuration.experiments.2020_10_combustion_synthesis_Pd_multi_objective_gly_acac_blend_FO_scaling.transformations_config")
from tests.test_parameter_transformations import generate_test_sets, test_transformations

MAN_TEST_FORWARD = [{'input': {'acac_amount': 0.0,
                               'anneal_temperature': 140.0,
                               'fuel_oxidizer_ratio': 0.5,
                               'concentration': 0.006},
                     'output': {'Pd_ACN': 0.23479155892521153,
                                'acac_H2O': 0.010627630406188715,
                                'anneal_temperature': 140.0,
                                'gly_H2O': 0.02124747733526647,
                                'H2O': 0.13333333333333336}}
                    ]

Man_TEST_REVERSE = [{'input': {'acac_amount': 0.5,
                               'anneal_temperature': 140.0,
                               'fuel_oxidizer_ratio': 0.5,
                               'concentration': 0.008,
                               'realized_parameters': {'Pd_ACN': 0.23479155892521153,
                                                       'acac_H2O': 0.010627630406188715,
                                                       'anneal_temperature': 140.0,
                                                       'gly_H2O': 0.02124747733526647,
                                                       'H2O': 0.13333333333333336}},
                     'output': {'acac_amount': 0.5,
                                'anneal_temperature': 140.0,
                                'fuel_oxidizer_ratio': 0.5,
                                'concentration': 0.008}}
                    ]


if __name__ == '__main__':
    # can be used to generate the structure needed to test different parameter sets
    # TEST_FORWARD, TEST_REVERSE = generate_test_sets(transformations_config, steps=5)
    # test_transformations(transformations_config, forward_set=TEST_FORWARD, reverse_set=TEST_REVERSE)

    test_transformations(transformations_config, MAN_TEST_FORWARD, Man_TEST_REVERSE)
