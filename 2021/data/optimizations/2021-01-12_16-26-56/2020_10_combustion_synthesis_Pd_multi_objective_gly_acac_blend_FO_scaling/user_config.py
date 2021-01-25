acac_amount = [0, 0.1, 0.5, 0.7, 0.9, 1.0] * 4
anneal_temperature = [180, 200, 220, 260] * 6
fuel_oxidizer_ratio = [0.5, 1.0, 2.0] * 8
concentration = [0.006, 0.012] * 12

# choices = [2, 3, 5, 6, 7, 9, 10, 11, 15, 16, 19]

choices = [2, 3, 5, 6, 7, 9, 10, 11]

control_config = {
    # data options, primarily for testing purposes
    "CAMPAIGN": {
        "NAME": "combustion_synthesis_30072020",
        "OWNER": "BenM",
        "OPTIMIZER": {
            "EHVI": {
                "points_to_test": {
                    'acac_amount': [acac_amount[x] for x in choices],
                    'anneal_temperature': [anneal_temperature[x] for x in choices],
                    'fuel_oxidizer_ratio': [fuel_oxidizer_ratio[x] for x in choices],
                    'concentration': [concentration[x] for x in choices],
                },
                "rel_noise": {"condxrfman_avg": 0.2, "anneal_temperature": 0},
                "sobol_iters": 0,
                "ignore_points": [],
            },
        },
        "DATA_PROCESSOR": {
            "OBJECTIVE": [
                {
                    "name": "condxrfman_avg",
                    "reference_point": 1e-9,
                    "minimize": False,
                    "dynamic": "5% of max"
                },
                {
                    "name": "anneal_temperature",
                    "reference_point": 280,
                    "minimize": True,
                },
            ],
            "RESPONSE": "linear",
        },
        "CONTINUE_CAMPAIGN": r"",
    },
    "MONITORING": (
        "RECORD_VIDEO",
        "STREAM_VIDEO",
        "MONGODB",
    ),
    "EMULATE": False,
    "CONTROL": (
        # Order is important for some operations!
        "MIX_CHEMICALS",
        "DROP_CAST",
        "FLIR_CAMERA",
        "ANNEAL",
        "XRF",
        "FLIR_CAMERA",
        "CONDUCTIVITY",
    )
}


DROP_0 = 0
DROP_1 = -25.5
DROP_2 = 25.5

# create a distribution of roughly 100 points in a square grid but truncated by a 9 mm diameter circle
SLIDE_CENTER_X = 12.85
SLIDE_CENTER_Y = 37.9

SLIDE_START_X = SLIDE_CENTER_X - 10
SLIDE_START_Y = SLIDE_CENTER_Y - 10

X_POINTS = [(x * 1.4) + SLIDE_CENTER_X for x in range(-10, 10)]
Y_POINTS = [(y * 1.4) + SLIDE_CENTER_Y for y in range(-10, 10)]

circle_points = []
for i in range(0, 20):
    for j in range(0, 20):
        x = X_POINTS[i]
        y = Y_POINTS[j]
        xdistl = x - SLIDE_CENTER_X - 1.5
        xdistr = x - SLIDE_CENTER_X + 1.5
        xdistc = x - SLIDE_CENTER_X
        ydist = y - SLIDE_CENTER_Y
        if (xdistl ** 2 + ydist ** 2 <= 9 ** 2) and (xdistr ** 2 + ydist ** 2 <= 9 ** 2) and (xdistc ** 2 + ydist ** 2 <= 8 ** 2):
            circle_points.append({'x': x, 'y': y})

# Contains parameters required for operation, required values left blank will be set from defaults
parameter_config = {
    # "POINTS_OF_INTEREST": [
    #     {'x': SLIDE_CENTER_X - 6, 'y': SLIDE_CENTER_Y - 5}, {'x': SLIDE_CENTER_X - 6, 'y': SLIDE_CENTER_Y - 4},
    #     {'x': SLIDE_CENTER_X - 6, 'y': SLIDE_CENTER_Y - 3}, {'x': SLIDE_CENTER_X - 2, 'y': SLIDE_CENTER_Y - 1},
    #     {'x': SLIDE_CENTER_X - 6, 'y': SLIDE_CENTER_Y}, {'x': SLIDE_CENTER_X - 6, 'y': SLIDE_CENTER_Y + 1},
    #     {'x': SLIDE_CENTER_X - 6, 'y': SLIDE_CENTER_Y + 2}, {'x': SLIDE_CENTER_X - 6, 'y': SLIDE_CENTER_Y + 3}
    # ],
    # "POINTS_OF_INTEREST": [
    # {'x': SLIDE_CENTER_X - 2, 'y': SLIDE_CENTER_Y + 2}, {'x': SLIDE_CENTER_X + 2, 'y': SLIDE_CENTER_Y + 2},
    #                     {'x': SLIDE_CENTER_X, 'y': SLIDE_CENTER_Y},
    # {'x': SLIDE_CENTER_X - 2, 'y': SLIDE_CENTER_Y - 2}, {'x': SLIDE_CENTER_X + 2, 'y': SLIDE_CENTER_Y - 2},
    # ],
    "POINTS_OF_INTEREST": circle_points,
    "ROBOT_PARAM_DEFAULTS": {
        # ANNEAL
        "anneal_cool_temperature": 60,
        "anneal_hold_time": 450,  # s
        "anneal_ramp_rate": 0.75,  # C/seconds
        # DROP
        "drop_height": 1.5,  # mm
        "drop_volume": 0.098,  # ml
        "drop_dispense_velocity": 0.3,  # ml/min
    },
    # "XRF": {
    #     "IMAGE_ONLY": True,
    #     "IMAGE_ONLY_KWARGS": dict(image_types=['10x'], autofocus='once'),
    #     "POINTS": [(0, -18.75), (0, 0), (0, 18.75), (0, -18.75 - 3.75), (0, 0 - 3.75), (0, 18.75 - 3.75)],
    # },
    "UV_CLEANING": False,
    "PLASMA_CLEANING": False,
    "DROP_CAST": {
        "OFFSETS": [  # from the center of the slide
            # {'y': DROP_1},
            {'y': DROP_0},
            # {'y': DROP_2},  # closest to handler
        ],
    },
    "CONDUCTIVITY": {
        "SOURCED": "CURRENT",
        "KEITHLEY_SETTINGS": {
            "script_name": r'20200728_updated_currentsweep',
            "settling_delay": 0.1,  # 0.25,
            "i_start": 0.0e-3,
            "i_end": 1e-3,
            "i_step": 0.2e-3,
            "nplc": 2,  # 5,
        },
    },
    "MIX_CHEMICALS": {
        "INITIAL_PROMPT": False,
    },
}
