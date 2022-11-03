from collections import OrderedDict


control_config = {
    # data options, primarily for testing purposes
    "CAMPAIGN": {
        "NAME": "combustion_synthesis_30072020",
        "OWNER": "BenM",
        "OPTIMIZER": [
            {
                "name": "Random",
                "number_of_runs": 30,
                "settings": {'replicates': 2,
                             'seed': 2,
                }
            },
            {
                "name": "Botorch",
                "number_of_runs": 0,
                "settings": {'ignore_points': [86],
                    'replicates': 2,
                }
            },
        ],
        "DATA_PROCESSOR": {
            "OBJECTIVE": [
                {
                    "name": "conductivity_avg",
                    "reference_point": 50,
                    "minimize": False,
                    "dynamic": "5% of max"
                },
            ],
            "RESPONSE": "linear",
        },
        "CONTINUE_CAMPAIGN": r"" # \\ALFRED\ada-nas\data\experiments\2022-06-29_22-33-42"
    },
    "MONITORING": (
        # "RECORD_VIDEO",
        # "STREAM_VIDEO",
        "MONGODB",
    ),
    "EMULATE": False,
    "CONTROL": (
        # Order is important for some operations!
        "MIX_CHEMICALS",
        "SPRAY_COAT",
        "XRF",
        "MICROSCOPE",
        "CONDUCTIVITY",
        "FLIR_CAMERA",

    )
}

# create a distribution of roughly 100 points in a square grid but truncated by a 9 mm diameter circle
SLIDE_CENTER_X = 12.85
SLIDE_CENTER_Y = 37.9

MEASUREMENT_CENTER_X = SLIDE_CENTER_X
MEASUREMENT_CENTER_Y = 50.8

X_POINTS = [(x * 3.5) + MEASUREMENT_CENTER_X for x in range(-2, 3)]
Y_POINTS = [(y * 2) + MEASUREMENT_CENTER_Y for y in range(-2, 3)]  # changing to 5 points instead of 10

# Contains parameters required for operation, required values left blank will be set from defaults
parameter_config = {
    "POINTS_OF_INTEREST": [{'x': MEASUREMENT_CENTER_X, 'y': y} for y in Y_POINTS],  # 5 points in a vertical line
    "ROBOT_PARAM_DEFAULTS": {
        # Spray
        "spray_volume": 0.30,  # mL
        "sample_draw_volume": 0.33,  # mL
        "combustion_time": 30,  # s
        "num_lines": 12,
        "spray_length": {  # the length of the pattern in mm this is likely fixed
            "x": 0,
            "y": -50,
            "units": "mm"
        },
        "spray_width": {
            "x": 25,
            "y": 0,
            "units": "mm"
        },
    },
    "UV_CLEANING": False,
    "PLASMA_CLEANING": False,
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
    "SPRAY_COAT": {
        "spray_generator_power": 50,  # %
        "move_up_during_pause": True,  # moves the nozzle out of the way during combustion pause time
        "starting_point_offset": {  # Use this to add an offset to slide_loc.
            "x": 0,  # x offset added due to incorrect default position
            "y": -1,  # y offset added so that the spray goes more over the middle of the slide
            "z": 0,
            "units": "mm"
        },
        "sample_draw_location": {
            "x": 439.5,
            "y": -256,
            "z": -71,
            "units": "mm"
        },

    },
    "MIX_CHEMICALS": {
        "INITIAL_PROMPT": False,
        "CAPPED_MIX_VIALS": False,
    },
    "MICROSCOPY": {
        "IMAGE_CONFIGS": {
            "MAG_2.5x": {
                "objective": "2.5x",
                "reflector": "BF&DF",
                "graphics": "scalebar",
                "export": ".png",
                "can_skip": True,
            },
            "MAG_20x": {
                "objective": "20x",
                "reflector": "BF&DF",
                "graphics": "scalebar",
                "export": ".png",
                "can_skip": True,
            }
        },
        "POINT_CONFIGS": [
            {
                "config": "MAG_2.5x",
                "points": [1, 3]
            },
            {
                "config": "MAG_20x",
                "points": [1, 3]  # list of indices of points of interest
            },
        ]
    },
    "XRF": {
        'X_RES': 40,  # The number of points to take in the x dimension. The points in the y-dim are calculated from this number.
        'ELEMENT': 'pd',
        'XRF_ORIGIN_X': 57.74,  # 69.63  # The position of the bottom-right corner of the slide, as reported by the XRF instrument. Elija's 57.74
        'XRF_ORIGIN_Y': 0.94,  # -2.93  # The position of the bottom-right corner of the slide, as reported by the XRF instrument. Elija's 0.94

        # XRF collection parameters
        'TIME_PER_POINT': 0.2,  # Time per measurement point in s.
        'MAP_WIDTH': 16,
        'MAP_HEIGHT': 16,
        'CENTRE_MAP_X': MEASUREMENT_CENTER_X,
        'CENTRE_MAP_Y': MEASUREMENT_CENTER_Y,
    },
    "ITERATIONS": {
        "MAX_CONC_SAMPLES": 4,  # maximum number of concurrent samples
    },
    "VALIDATION": {
        "conductivity_avg": lambda x: x >= 0.0
    },
    # "DUPLICATE_CRITERIA": [
    #     lambda x: x['sample'] == 50
    # ],
      # "COMPOUND_VALIDATION": [
      #     lambda x: abs (100 - (x['feCO'] +  x['feH2'])) >= 10.0
      # ],
}

