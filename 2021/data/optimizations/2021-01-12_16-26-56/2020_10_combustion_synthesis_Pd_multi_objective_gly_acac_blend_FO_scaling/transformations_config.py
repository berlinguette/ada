# translates a list of n dimensionless parameters (0 to 0.99999...) to a list of n + 1 simplex coordinates
def simplex_from_cartesian(parameter_list):
    x = parameter_list  # rename parameters as x[0:n]
    coordinates = [1]  # initialize list of output coord
    for i in range(len(x)):  # recursively build the coordinates based on some fun algebra
        for j in range(len(coordinates)):
            coordinates[j] = coordinates[j] * (1 + i) * (1 - x[i]) / (1 + i - i * x[i])
        coordinates.insert(0, (x[i] / (1 + i - i * x[i])))
    return coordinates


# translates a list of n simplex coordinates to n - 1 dimensionless (0 to 1) cartesian parameters
# cartesian_from_simplex( simplex_from_cartesian( list ) ) == list
# does not require normalization of simplex coordinates (accepts non-unit simplexes)
def cartesian_from_simplex(coordinate_list):
    s = coordinate_list[::-1]  # rename coordinate list as s[0:n]
    parameters = []
    for i in range(1, len(s)):  # define each parameter as a ratio of simplex coordinates​
        if sum(s[0:i + 1]) == 0:
            x = 0
        else:
            x = i * s[i] / (sum(s[0:i]) + i * s[i])
        parameters.append(x)
    return parameters


ROBOT_PARAMS = {
    "Pd_ACN": {
        'stock_tray_name': 'stock_tray',
        'index': [5, 6, 7, 13],
        'chemical_name': 'Pd(NO3)2_ACN',
        'concentration': {'value': 12., 'units': 'mg/ml'},
        'molar_weight': {'value': 1., 'units': 'mg'},
        'solvent': 'ACN',
        'density': {'value': 0.798, 'units': 'g/cm^3'}
    },
    "gly_H2O": {
        'stock_tray_name': 'stock_tray',
        'index': [0, 1, 2],
        'chemical_name': 'glycine_H2O',
        'concentration': {'value': 12, 'units': 'mg/ml'},
        'molar_weight': {'value': 1., 'units': 'mg'},
        'solvent': 'H2O',
        'density': {'value': 1.012, 'units': 'g/cm^3'}
    },
    "acac_H2O": {
        'stock_tray_name': 'stock_tray',
        'index': [8, 9, 10],
        'chemical_name': 'acetylacetone_H2O',
        'concentration': {'value': 12, 'units': 'mg/ml'},
        'molar_weight': {'value': 1., 'units': 'mg'},
        'solvent': 'H2O',
        'density': {'value': 1.012, 'units': 'g/cm^3'}
    },
    "dilutant": {
        'stock_tray_name': 'stock_tray',
        'index': [3, 4, 11, 12],
        'chemical_name': 'H2O',
        'concentration': {'value': 1, 'units': 'mg/ml'},
        'molar_weight': {'value': 1., 'units': 'mg'},
        'solvent': 'H2O',
        'density': {'value': 1.0, 'units': 'g/cm^3'}
    },
}

OPTIMIZER_PARAMS = {
    'fuel_oxidizer_ratio': {
        'display_name': '',
        'minimum': 0,
        'maximum': 2,
    },
    'acac_amount': {  # amount of acac in mix from 0 to 1
        'display_name': '',
        'minimum': 0.,
        'maximum': 1.,
    },
    'concentration': {  # fuel + oxidizer in g/ mL
        'display_name': '',
        'minimum': 0.006,
        'maximum': 0.012,
    },
    'anneal_temperature': {
        'display_name': '',
        'minimum': 180.,
        'maximum': 280.,
    },
}

# constants
MM_pd = 230.43
MM_gly = 75.07
MM_acac = 100.13
VOL = 0.200
C_precursors = 0.012
fuel_oxidizer_ratio = 0.5

val_pd = 10
val_gly = 9
val_acac = 24

REDOX = True  # uses volume based transfrom if false


def forward_transformations(fuel_oxidizer_ratio, acac_amount, concentration, anneal_temperature):
    if REDOX:
        return forward_transformations_redox(fuel_oxidizer_ratio, acac_amount, concentration, anneal_temperature)
    else:
        return forward_transformations_volume(fuel_oxidizer_ratio, acac_amount, concentration, anneal_temperature)


def forward_transformations_redox(fuel_oxidizer_ratio, acac_amount, concentration, anneal_temperature):
    # oxidizer
    vol_pd = 1  # fix pd volume, but we will scale back to VOL at end of transformation
    mass_pd = C_precursors * vol_pd
    mols_pd = mass_pd / MM_pd
    redox_pd = mols_pd * val_pd

    # fuels
    redox_gly = (fuel_oxidizer_ratio * redox_pd)
    mols_gly = redox_gly / val_gly
    mass_gly = mols_gly * MM_gly
    vol_gly = (mass_gly / C_precursors) * (1 - acac_amount)

    redox_acac = (fuel_oxidizer_ratio * redox_pd)
    mols_acac = redox_acac / val_acac
    mass_acac = mols_acac * MM_acac
    vol_acac = (mass_acac / C_precursors) * acac_amount

    # normalize to VOL
    vol_total = vol_pd + vol_gly + vol_acac

    vol_pd = (vol_pd / vol_total) * VOL
    vol_gly = (vol_gly / vol_total) * VOL
    vol_acac = (vol_acac / vol_total) * VOL

    vol_dilutant = (VOL * (C_precursors / concentration)) - VOL

    # renormalize to vol
    total_vol = VOL + vol_dilutant
    vol_ratio = VOL / total_vol

    # keys must match ROBOT_PARAMS
    return {
        "Pd_ACN": vol_pd * vol_ratio,
        "gly_H2O": vol_gly * vol_ratio,
        "acac_H2O": vol_acac * vol_ratio,
        "dilutant": vol_dilutant * vol_ratio,
        "anneal_temperature": anneal_temperature,
    }


def forward_transformations_does_not_work(fuel_oxidizer_ratio, acac_amount, anneal_temperature):
    # showing the work
    # fuel_oxidizer_ratio = (redox_gly + redox_acac) / redox_pd
    # redox_pd  = (redox_gly + redox_acac) / fuel_oxidizer_ratio
    # ((vol_pd * C_precursors / MM_pd) * val_pd)) = (((vol_gly * C_precursors / MM_gly) * val_gly) + ((vol_acac * C_precursors / MM_acac) * val_acac)) / fuel_oxidizer_ratio
    # vol_pd= MM_pd * (((vol_gly * C_precursors / MM_gly) * val_gly) + ((vol_acac * C_precursors / MM_acac) * val_acac)) / (fuel_oxidizer_ratio * C_precursors * val_pd)
    #
    # vol_pd + vol_fuel = VOL
    # vol_fuel = (VOL - vol_pd)
    # vol_acac = (acac_amount * (VOL - vol_pd))
    # vol_gly = ((1 - acac_amount) * (VOL - vol_pd))

    # vol_pd = MM_pd * (((((1 - acac_amount) * (VOL - vol_pd)) * C_precursors / MM_gly) * val_gly) + (((acac_amount * (VOL - vol_pd)) * C_precursors / MM_acac) * val_acac)) / (fuel_oxidizer_ratio * C_precursors * val_pd)
    # vol_pd = -MM_pd*VOL*(-acac_amount*C_precursors*val_gly*MM_acac*fuel_oxidizer_ratio*val_pd + acac_amount*MM_gly*val_acac + C_precursors*val_gly*MM_acac*fuel_oxidizer_ratio*val_pd) / (MM_pd*acac_amount*C_precursors*val_gly*MM_acac*fuel_oxidizer_ratio*val_pd - MM_pd*acac_amount*MM_gly*val_acac - MM_pd*C_precursors*val_gly*MM_acac*fuel_oxidizer_ratio*val_pd - MM_gly*MM_acac*fuel_oxidizer_ratio*val_pd)

    if acac_amount != 0:
        numerator = -MM_pd * VOL * (-acac_amount * C_precursors * val_gly * MM_acac * fuel_oxidizer_ratio * val_pd + acac_amount * MM_gly * val_acac + C_precursors * val_gly * MM_acac * fuel_oxidizer_ratio * val_pd)
        denominator = (MM_pd * acac_amount * C_precursors * val_gly * MM_acac * fuel_oxidizer_ratio * val_pd - MM_pd * acac_amount * MM_gly * val_acac - MM_pd * C_precursors * val_gly * MM_acac * fuel_oxidizer_ratio * val_pd - MM_gly * MM_acac * fuel_oxidizer_ratio * val_pd)
        if denominator != 0:
            vol_pd = numerator / denominator
        else:
            vol_pd = 0

        vol_gly = ((1 - acac_amount) * (VOL - vol_pd))
        vol_acac = (acac_amount * (VOL - vol_pd))
    else:
        # oxidizer
        vol_pd = 1  # fix pd volume, but we will scale back to VOL at end of transformation
        mass_pd = C_precursors * vol_pd
        mols_pd = mass_pd / MM_pd
        redox_pd = mols_pd * val_pd

        # fuels
        redox_gly = (fuel_oxidizer_ratio * redox_pd)
        mols_gly = redox_gly / val_gly
        mass_gly = mols_gly * MM_gly
        vol_gly = (mass_gly / C_precursors)

        # normalize to VOL
        vol_total = vol_pd + vol_gly

        vol_pd = (vol_pd / vol_total) * VOL
        vol_gly = (vol_gly / vol_total) * VOL
        vol_acac = 0

    # keys must match ROBOT_PARAMS
    return {
        "Pd_ACN": vol_pd,
        "gly_H2O": vol_gly,
        "acac_H2O": vol_acac,
        "anneal_temperature": anneal_temperature,
    }


def forward_transformations_volume(fuel_oxidizer_ratio, acac_amount, concentration, anneal_temperature):

    # treat mix of fuels as single entity
    vol_fuel = 1
    vol_gly = vol_fuel * (1 - acac_amount)
    vol_acac = vol_fuel * acac_amount

    mass_gly = vol_gly * C_precursors
    mass_acac = vol_acac * C_precursors

    mols_gly = mass_gly / MM_gly
    mols_acac = mass_acac / MM_acac

    redox_gly = mols_gly * val_gly
    redox_acac = mols_acac * val_acac
    redox_fuel = redox_gly + redox_acac

    # oxidizer
    try:
        redox_pd = redox_fuel / fuel_oxidizer_ratio
        mols_pd = redox_pd / val_pd
        mass_pd = mols_pd * MM_pd
        vol_pd = mass_pd / C_precursors
    except ZeroDivisionError:
        vol_pd = 1
        vol_gly = 0
        vol_acac = 0

    # normalize to VOL
    vol_total = vol_pd + vol_gly + vol_acac

    vol_pd = (vol_pd / vol_total) * VOL
    vol_gly = (vol_gly / vol_total) * VOL
    vol_acac = (vol_acac / vol_total) * VOL

    vol_dilutant = (VOL * (C_precursors / concentration)) - VOL

    # renormalize to vol
    total_vol = VOL + vol_dilutant
    vol_ratio = VOL / total_vol


    # keys must match ROBOT_PARAMS
    return {
        "Pd_ACN": vol_pd * vol_ratio,
        "gly_H2O": vol_gly * vol_ratio,
        "acac_H2O": vol_acac * vol_ratio,
        "dilutant": vol_dilutant * vol_ratio,
        "anneal_temperature": anneal_temperature,
    }


def reverse_transformations(realized_parameters, fuel_oxidizer_ratio, acac_amount, concentration, anneal_temperature):
    vol_fuel = realized_parameters['gly_H2O'] + realized_parameters['acac_H2O']
    vol_pd = realized_parameters['Pd_ACN']
    vol_gly = realized_parameters['gly_H2O']
    vol_acac = realized_parameters['acac_H2O']
    vol_dilutant = realized_parameters['dilutant']

    mass_pd = vol_pd * C_precursors
    mass_gly = vol_gly * C_precursors
    mass_acac = vol_acac * C_precursors

    mols_pd = mass_pd / MM_pd
    mols_gly = mass_gly / MM_gly
    mols_acac = mass_acac / MM_acac

    redox_pd = mols_pd * val_pd
    redox_gly = mols_gly * val_gly
    redox_acac = mols_acac * val_acac

    try:
        fuel_oxidizer_ratio = (redox_gly + redox_acac) / redox_pd
    except ZeroDivisionError:
        fuel_oxidizer_ratio = 99.

    try:
        redox_acac_amount = redox_acac / (redox_acac + redox_gly)
    except ZeroDivisionError:
        redox_acac_amount = acac_amount
    try:
        volume_acac_amount = realized_parameters['acac_H2O'] / vol_fuel
    except ZeroDivisionError:
        volume_acac_amount = acac_amount

    total_vol = vol_fuel + vol_pd + vol_dilutant
    vol_ratio = (vol_fuel + vol_pd) / total_vol

    calc_concentration = C_precursors * VOL / (vol_fuel + vol_pd + vol_dilutant)

    # keys must match OPTIMISER_PARAMS
    return {
        'fuel_oxidizer_ratio': fuel_oxidizer_ratio,
        'acac_amount': redox_acac_amount if REDOX else volume_acac_amount,
        'concentration': calc_concentration * vol_ratio,
        'anneal_temperature': anneal_temperature,
    }


def compare_reverse(realized_parameters, expected_parameters, acac_amount):
    vol_fuel = realized_parameters['gly_H2O'] + realized_parameters['acac_H2O']
    vol_pd = realized_parameters['Pd_ACN']
    vol_gly = realized_parameters['gly_H2O']
    vol_acac = realized_parameters['acac_H2O']

    mass_pd = vol_pd * C_precursors
    mass_gly = vol_gly * C_precursors
    mass_acac = vol_acac * C_precursors

    mols_pd = mass_pd / MM_pd
    mols_gly = mass_gly / MM_gly
    mols_acac = mass_acac / MM_acac

    redox_pd = mols_pd * val_pd
    redox_gly = mols_gly * val_gly
    redox_acac = mols_acac * val_acac

    fuel_oxidizer_ratio = (redox_gly + redox_acac) / redox_pd
    try:
        actual_acac_amount = realized_parameters['acac_H2O'] / vol_fuel
    except ZeroDivisionError:
        actual_acac_amount = acac_amount

    exp_vol_fuel = expected_parameters['gly_H2O'] + expected_parameters['acac_H2O']
    exp_vol_pd = expected_parameters['Pd_ACN']
    exp_vol_gly = expected_parameters['gly_H2O']
    exp_vol_acac = expected_parameters['acac_H2O']

    exp_mass_pd = exp_vol_pd * C_precursors
    exp_mass_gly = exp_vol_gly * C_precursors
    exp_mass_acac = exp_vol_acac * C_precursors

    exp_mols_pd = exp_mass_pd / MM_pd
    exp_mols_gly = exp_mass_gly / MM_gly
    exp_mols_acac = exp_mass_acac / MM_acac

    exp_redox_pd = exp_mols_pd * val_pd
    exp_redox_gly = exp_mols_gly * val_gly
    exp_redox_acac = exp_mols_acac * val_acac

    exp_fuel_oxidizer_ratio = (exp_redox_gly + exp_redox_acac) / exp_redox_pd
    try:
        exp_actual_acac_amount = expected_parameters['acac_H2O'] / exp_vol_fuel
    except ZeroDivisionError:
        exp_actual_acac_amount = acac_amount

    output = {
        'volumes': [(vol_pd, exp_vol_pd), (vol_gly, exp_vol_gly), (vol_acac, exp_vol_acac), (vol_fuel, exp_vol_fuel)],
        'masses': [(mass_pd, exp_mass_pd), (mass_gly, exp_mass_gly), (mass_acac, exp_mass_acac)],
        'mols': [(mols_pd, exp_mols_pd), (mols_gly, exp_mols_gly), (mols_acac, exp_mols_acac)],
        'redox': [(redox_pd, exp_redox_pd), (redox_gly, exp_redox_gly), (redox_acac, exp_redox_acac)],
        'fuel_oxidizer ratio': [(fuel_oxidizer_ratio,exp_fuel_oxidizer_ratio)],
        'acac_amount': [(actual_acac_amount, exp_actual_acac_amount)]
    }
    output_labels = {
        'volumes': ['pd', 'gly', 'acac', 'fuel'],
        'masses': ['pd', 'gly', 'acac'],
        'mols': ['pd', 'gly', 'acac'],
        'redox': ['pd', 'gly', 'acac'],
        'fuel_oxidizer ratio': [''],
        'acac_amount': ['']
    }

    for k, v in output.items():
        print(f'{k}\n')
        for i in range(0, len(v)):
            print(f'{output_labels[k][i]}: expected = {v[i][1]}\tactual = {v[i][0]}\tdifference = {v[i][0] - v[i][1]}\tdifference_ratio = {(v[i][0] / v[i][1]) * 100}\n')
        print('\n\n')


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    realized_parameters = {
        'acac_H2O': 0.03980583,
        'gly_H2O': 0.03669903,
        'Pd_ACN': 0.173652,
    }

    expected_parameters = {
        'acac_H2O': 0.0,
        'gly_H2O': 0.0,
        'Pd_ACN': 0.25,
    }

    # compare_reverse(realized_parameters, expected_parameters, 0.5)

    trans = forward_transformations(0.75, 0.5, 0)
    rtrans_exp = reverse_transformations(expected_parameters, 1.5, 0.5, 0)
    rtrans = reverse_transformations(realized_parameters, 1.5, 0.5, 0)
    arra = np.empty(shape=(300, 1), dtype=float)
    arrg = np.empty(shape=(300, 1), dtype=float)
    arrp = np.empty(shape=(300, 1), dtype=float)
    arrt = np.empty(shape=(300, 1), dtype=float)
    arrf = np.empty(shape=(300, 1), dtype=float)
    xline = []
    for i in range(0, 300):
        trans = forward_transformations(fuel_oxidizer_ratio=i / 100.0, acac_amount=0.5, anneal_temperature=0)
        arra[i] = trans['acac_H2O']
        arrg[i] = trans['gly_H2O']
        arrp[i] = trans['Pd_ACN']
        arrt[i] = arra[i] + arrg[i] + arrp[i]
        arrf[i] = arra[i] + arrg[i]
        xline.append(i / 100.0)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_title('acac-gly 0')
    ax1.scatter(xline, arra)
    ax1.scatter(xline, arrg)
    ax1.scatter(xline, arrp)
    ax1.scatter(xline, arrf)
    ax1.scatter(xline, arrt)
    plt.xlabel('fuel to ox ratio')
    plt.ylabel('volume (mL)')
    ax1.legend(['acac', 'gly', 'pd', 'fuel', 'total'])

    fox = np.empty(shape=(300, 1), dtype=float)
    fox2 = np.empty(shape=(300, 1), dtype=float)
    for i in range(0, 300):
        realized_parameters = {
            'acac_H2O': arra[i],
            'gly_H2O': arrg[i],
            'Pd_ACN': arrp[i],
        }
        rtrans = reverse_transformations(realized_parameters, 1.5, 0.5, 0)
        fox[i] = rtrans['fuel_oxidizer_ratio']
        fox2[i] = arra[i] + arrg[i] / arrp[i] if arrp[i] > 0 else 0

    ax2 = fig.add_subplot(122)
    ax2.set_title('acac-gly 0')
    ax2.scatter(fox2, fox)

    plt.xlabel('fuel to ox volume ratio')
    plt.ylabel('fuel to ox ratio')
    # sn1 = sns.heatmap(arra, ax=ax1)

    asum = np.sum(arra)

    print(f'tacac: {asum}')

    fig.tight_layout()
    # plt.imshow(arr, cmap='hot', interpolation='nearest')
    plt.show()
    print('done')
