from collections import OrderedDict

ROBOT_PARAMS = {
    "Pd_ACN": {
        'stock_tray_name': 'stock_tray',
        'index': [0, 1, 2, 3, 4, 15],
        'chemical_name': 'Pd(NO3)2_ACN',
        'concentration': {'value': 30., 'units': 'mg/ml'},
        'molar_weight': {'value': 1., 'units': 'mg'},
        'solvent': 'ACN',
        'density': {'value': 0.816, 'units': 'g/cm^3'}
    },
    "acac_ACN": {
        'stock_tray_name': 'stock_tray',
        'index': [8, 9],
        'chemical_name': 'acetylacetone_acetonitrile',
        'concentration': {'value': 30, 'units': 'mg/ml'},
        'molar_weight': {'value': 1., 'units': 'mg'},
        'solvent': 'ACN',
        'density': {'value': 0.816, 'units': 'g/cm^3'}
    },
    "ACN": {
        'stock_tray_name': 'stock_tray',
        'index': [10, 11, 12, 13, 14],
        'chemical_name': 'acetonitrile',
        'concentration': {'value': 1, 'units': 'mg/ml'},
        'molar_weight': {'value': 1., 'units': 'mg'},
        'solvent': 'ACN',
        'density': {'value': 0.786, 'units': 'g/cm^3'}
    },
    "DMSO": {
        'stock_tray_name': 'stock_tray',
        'index': [5, 6, 7],
        'chemical_name': 'dimethyl sulfoxide',
        'concentration': {'value': 1, 'units': 'mg/ml'},
        'molar_weight': {'value': 1., 'units': 'mg'},
        'solvent': 'H2O',
        'density': {'value': 1.1, 'units': 'g/cm^3'}
    },
}

OPTIMIZER_PARAMS = OrderedDict([
    ('concentration', {  # g/mL
        'display_name': '',
        'minimum': 0.010,
        'maximum': 0.020,
    }),
    ('DMSO_content', {  # v/v
        'display_name': '',
        'minimum': 0,
        'maximum': 0.3,
    }),
    ('combustion_temp', {  # degrees C
        'display_name': '',
        'minimum': 220,
        'maximum': 300,
    }),
    ('air_flow_rate', {  # percentage valve open
        'display_name': '',
        'minimum': 65,
        'maximum': 100,
    }),
    ('spray_flow_rate', {  # mL/s
        'display_name': '',
        'minimum': 0.002,
        'maximum': 0.008,
    }),
    ('spray_height', {  # mm
        'display_name': '',
        'minimum': 10,
        'maximum': 25,
    }),
    ('num_passes', {  # unitless
        'display_name': '',
        'minimum': 0.51,
        'maximum': 10.49,
    }),
])

# constants
MM_pd = 230.43
MM_acac = 100.13
VOL = 0.380
C_precursors = 0.030

val_pd = 10
val_urea = 6
val_acac = 24

def forward_transformations(concentration, DMSO_content, combustion_temp, air_flow_rate, spray_flow_rate, spray_height, num_passes):

    fuel_oxidizer_ratio = 1

    # oxidizer
    vol_pd = 1  # fix pd volume, but we will scale back to VOL at end of transformation
    mass_pd = C_precursors * vol_pd
    mols_pd = mass_pd / MM_pd
    redox_pd = mols_pd * val_pd

    # fuels
    redox_acac = (fuel_oxidizer_ratio * redox_pd)
    mols_acac = redox_acac / val_acac
    mass_acac = mols_acac * MM_acac
    vol_acac = (mass_acac / C_precursors)

    # normalize to VOL
    vol_total = vol_pd + vol_acac

    vol_pd = (vol_pd / vol_total) * VOL
    vol_acac = (vol_acac / vol_total) * VOL

    vol_dilutant = (VOL * (C_precursors / concentration)) - VOL

    # renormalize to vol
    total_vol = VOL + vol_dilutant
    vol_ratio = VOL / total_vol

    real_vol_Pd = vol_pd*vol_ratio
    real_vol_acac = vol_acac*vol_ratio
    real_vol_dilutant = vol_dilutant*vol_ratio

    real_vol_DMSO = VOL*DMSO_content
    real_vol_ACN = real_vol_dilutant - real_vol_DMSO

    # keys must match ROBOT_PARAMS
    return {
        "Pd_ACN": real_vol_Pd,
        "acac_ACN": real_vol_acac,
        "ACN": real_vol_ACN,
        "DMSO": real_vol_DMSO,
        'combustion_temp': combustion_temp,
        'air_flow_rate': air_flow_rate,
        'spray_flow_rate': spray_flow_rate,
        'spray_height': spray_height,
        'num_passes': int(num_passes + 0.5),
    }


def reverse_transformations(realized_parameters, concentration, DMSO_content, combustion_temp, air_flow_rate, spray_flow_rate, spray_height, num_passes):

    vol_pd = realized_parameters['Pd_ACN']
    vol_acac = realized_parameters['acac_ACN']
    vol_ACN = realized_parameters['ACN']
    vol_DMSO = realized_parameters['DMSO']
    total_vol = + vol_acac + vol_pd + vol_ACN + vol_DMSO

    mass_pd = vol_pd * C_precursors
    mass_acac = vol_acac * C_precursors

    concentration = (mass_pd + mass_acac)/total_vol
    DMSO_content = (vol_DMSO) / total_vol


    # keys must match OPTIMIZER_PARAMS
    return {
        'concentration': concentration,
        'DMSO_content': DMSO_content,
        'combustion_temp': realized_parameters['combustion_temp'],
        'air_flow_rate': realized_parameters['air_flow_rate'],
        'spray_flow_rate': realized_parameters['spray_flow_rate'],
        'spray_height': realized_parameters['spray_height'],
        'num_passes': realized_parameters['num_passes'],
    }

