"""Script for running hyperopt to optimize hyper-parameters."""

import pprint
from subprocess import call
from collections import OrderedDict
import numpy as np

import hyperopt
from hyperopt import fmin, tpe

base_params_files =  ['best-params.dat']
base_params_files += ['lf-params.dat']
base_params_files += ['iter-params.dat']
base_params_files += ['crazy-fat-sunset-params.dat']

run_args = []
run_args += ['--with-cells']
run_args += ['--layer-functions', 'layer_functions']
run_args += ['--layer-functions', 'layer_functions-pair']
run_args += ['--layer-functions', 'layer_functions-dp']
run_args += ['-p', 'post__nonphys_odd=True']
run_args += ['--events', '1001,1002,1003']
#run_args += ['--events', '1001,1004,1006,1039']
#run_args += ['--events', '1004']
#run_args += ['--events', '1000,1004']

# define an objective function
def objective(args):
    params_filename = 'params.dat'
    results_filename = 'results.dat'
    params = OrderedDict()
    for filename in base_params_files:
        with open(filename) as file:
            params.update(eval(file.read()))
    params.update(args)
    with open(params_filename, 'w') as file:
        pprint.pprint(params, stream=file)
    with open(results_filename, 'w') as file:
        pprint.pprint({'status': hyperopt.STATUS_FAIL}, stream=file)
    call(['run.bat'] + run_args + ['--params-file', params_filename, '--results-file', results_filename])
    with open(results_filename) as file:
        results = eval(file.read())
    return results

# define a search space
from hyperopt import hp
space = {
#    'rank__ntop'           : hp.qlognormal('rank__ntop'          , np.log(200000.0), 2.0, 1000.0),  # was 33000
#    'rank__ntop'           : 200000,
    'rank__ntop_qu'        : hp.lognormal('rank__ntop_qu', np.log(8.117e-6), 0.10),
    'rank__ntop_li'        : hp.lognormal('rank__ntop_li', np.log(0.143   ), 0.10),
#    'value__hit_bonus'     : hp.lognormal('value__hit_bonus'     , np.log(0.0000550), 0.2), # was 0.0001437, 0.0001398, 0.0001355, 0.0001278
#    'value__cross_bonus'   : hp.lognormal('value__cross_bonus'   , np.log(5.357e-05), 0.2),   # was 0.0
#    'value__bayes_weight'  : hp.lognormal('value__bayes_weight'  , np.log(1.603e-05) , 0.2),
#    'value__p0'            : hp.lognormal('value__p0'            , np.log(0.01)     , 1.0),
#    'value__fit_weight_hcs': hp.lognormal('value__fit_weight_hcs', np.log(0.0006197) , 0.2),
#    'value__ploss_weight': hp.lognormal('value__ploss_weight', np.log(8.449e-7) , 0.2),
#    'value__ploss_bias'  : hp.lognormal('value__ploss_bias'  , np.log(18.31)   , 0.2),
#    'value__cells_weight': hp.lognormal('value__cells_weight', np.log(0.008654) , 0.2),
#    'value__cells_bias'  : hp.uniform('value__cells_bias', 0.90, 1.0),
#    'commit__niter'        : 20,
#    'commit__nmax'         : hp.quniform('commit__nmax', 2000, 5000, 100),
#    'commit__max_loss_fraction': hp.uniform('commit__max_loss_fraction', 0.20, 0.50),
#    'commit__max_nloss'    : 20,
#    'commit__max_nloss'    : hp.quniform('commit__max_nloss', 2, 5, 1),
#    'follow__niter'        : hp.quniform('follow__niter', 11, 13, 1),
#    'follow__drop_start'   : hp.quniform('follow__drop_start', 3, 5, 1),
#    'sunset__nb__cut_factor' : hp.lognormal('sunset__nb__cut_factor', np.log(7.379), 0.3),
#    'sunset__follow__niter'      : hp.quniform('sunset__follow__niter', 8, 13, 1),
#    'sunset__follow__drop_start' : hp.quniform('sunset__follow__drop_start', 2, 5, 1),
#    'commit__max_nloss'        : hp.quniform('commit__max_nloss', 0, 10, 1),
#    'upto2__commit__min_nhits' : hp.quniform('upto2__commit__min_nhits', 3, 10, 1),
#    'upto0__ADD__commit__min_nhits' : hp.quniform('upto0__ADD__commit__min_nhits', 0, 10, 1),
#    'nb__cyl_first_dz_0'  : hp.lognormal('nb__cyl_first_dz_0'  , np.log(1000.0   ), 0.1),   # was 1000.0
#    'nb__cyl_first_dz_1'  : hp.lognormal('nb__cyl_first_dz_1'  , np.log(200.0    ), 0.1),   # was 200.0
#    'nb__cyl_first_dz_2'  : hp.lognormal('nb__cyl_first_dz_2'  , np.log(200.0    ), 1.0),
#    'nb__cyl_first_dz_3'  : hp.lognormal('nb__cyl_first_dz_3'  , np.log(200.0    ), 1.0),
#    'nb__cyl_first_dz_4'  : hp.lognormal('nb__cyl_first_dz_4'  , np.log(400.0    ), 0.1),   # was 400.0
#    'nb__cyl_first_dz_5'  : hp.lognormal('nb__cyl_first_dz_5'  , np.log(400.0    ), 1.0),   # was 400.0
#    'nb__cap_first_dr2_0' : hp.lognormal('nb__cap_first_dr2_0' , np.log(70.0     ), 0.1),   # was 100.0
#    'nb__cap_first_dr2_1' : hp.lognormal('nb__cap_first_dr2_1' , np.log(200.0    ), 0.1),   # was 200.0
#    'nb__cap_first_dr2_2' : hp.lognormal('nb__cap_first_dr2_2' , np.log(200.0    ), 1.0),
#    'sunset__nb__cyl_first_dz_0'  : hp.lognormal('sunset__nb__cyl_first_dz_0'  , np.log(1000.0   ), 0.1),
#    'sunset__nb__cyl_first_dz_1'  : hp.lognormal('sunset__nb__cyl_first_dz_1'  , np.log(200.0    ), 0.1),
#    'sunset__nb__cyl_first_dz_2'  : hp.lognormal('sunset__nb__cyl_first_dz_2'  , np.log(200.0    ), 1.0),
#    'sunset__nb__cyl_first_dz_3'  : hp.lognormal('sunset__nb__cyl_first_dz_3'  , np.log(200.0    ), 1.0),
#    'sunset__nb__cyl_first_dz_4'  : hp.lognormal('sunset__nb__cyl_first_dz_4'  , np.log(400.0    ), 0.1),
#    'sunset__nb__cyl_first_dz_5'  : hp.lognormal('sunset__nb__cyl_first_dz_5'  , np.log(400.0    ), 1.0),
#    'sunset__nb__cap_first_dr2_0' : hp.lognormal('sunset__nb__cap_first_dr2_0' , np.log(70.0     ), 0.1),
#    'sunset__nb__cap_first_dr2_1' : hp.lognormal('sunset__nb__cap_first_dr2_1' , np.log(200.0    ), 0.1),
#    'sunset__nb__cap_first_dr2_2' : hp.lognormal('sunset__nb__cap_first_dr2_2' , np.log(200.0    ), 1.0),
#    'nb__min_radius'      : hp.lognormal('nb__min_radius'      , np.log(1.0      ), 1.0),
#    'nb__nbins_radius'    : hp.lognormal('nb__nbins_radius'    , np.log(30       ), 0.5),   # was 100
#    'nb__cap_origin_radius': hp.lognormal('nb__cap_origin_radius', np.log(41.0     ), 0.1),  # was 25.0
#    'nb__cyl_scale'       : hp.lognormal('nb__cyl_scale'       , np.log(5.0      ), 0.05),   # was 5.0
#    'nb__cyl_origin_area'  : hp.lognormal('nb__cyl_origin_area' , np.log(371.5    ), 0.1),   # was 392.7
#    'nb__dist_threshold'  : hp.lognormal('nb__dist_threshold'  , np.log(0.03356  ), 0.02), # was 0.0250, 0.03356
#    'nb__cut_factor'      : hp.lognormal('nb__cut_factor', np.log(2.35), 0.1),
#    'nb__cut_factor'      : hp.lognormal('nb__cut_factor', np.log(2.32), 0.001),
#    'nb__dist_trust'      : hp.uniform('nb__dist_trust', 0.0, 1.0),
#    'nb__cells_cut'      : hp.uniform('nb__cells_cut', 0.08, 0.10),
#    'pair__dist_threshold': hp.lognormal('pair__dist_threshold', np.log(0.01669  ), 0.02),  # was 0.0050, 0.02145
#    'pair__diff_threshold': hp.lognormal('pair__diff_threshold', np.log(0.002125 ), 0.02),  # was 0.0025, 0.002237
#    'pair__cut': hp.lognormal('pair__cut', np.log(1.46), 0.001),
#    'nb__origin_dz'        : hp.lognormal('nb__origin_dz', np.log(20.0), 0.2),
#    'nb__nlayers'          : hp.quniform('nb__nlayers', 1, 3, 1),
#    'nb__radius_exp'       : hp.lognormal('nb__radius_exp', np.log(1.0), 0.2) ,
#    'follow__nskip_max'    : hp.quniform('follow__nskip_max', 1, 3, 1),
#    'sunset__follow__weird_triples': True,
#    'sunset__follow__weird_k'      : hp.quniform('sunset__follow__weird_k', 4, 6, 1),
#    'sunset__nb__cyl_origin_area'  : hp.lognormal('sunset__nb__cyl_origin_area', np.log(14040.0), 0.2),
#    'sunset__nb__cap_origin_radius': hp.lognormal('sunset__nb__cap_origin_radius', np.log(500.0), 0.2),
#    'sunset__rank__ntop'           : 400000,
    'sunset__rank__ntop_qu'        : hp.lognormal('sunset__rank__ntop_qu', np.log(150e-6), 0.2),
#     'sunset__value__hit_bonus'      : hp.lognormal('sunset__value__hit_bonus'     ,  np.log(5.5e-05  ), 0.2),
#     'sunset__value__cross_bonus'    : hp.lognormal('sunset__value__cross_bonus'   ,  np.log(5.357e-05), 0.2),
#     'sunset__value__bayes_weight'   : hp.lognormal('sunset__value__bayes_weight'  ,  np.log(1.603e-05), 0.2),
#     'sunset__value__fit_weight_hcs' : hp.lognormal('sunset__value__fit_weight_hcs',  np.log(0.0006197), 0.2),
#     'sunset__value__ploss_weight'   : hp.lognormal('sunset__value__ploss_weight'  ,  np.log(1.043e-5 ), 0.2),
#     'sunset__value__ploss_bias'     : hp.lognormal('sunset__value__ploss_bias'    ,  np.log(15.08    ), 0.2),
#     'sunset__value__cells_weight'   : hp.lognormal('sunset__value__cells_weight'  ,  np.log(0.008654 ), 0.2),
#     'sunset__value__cells_bias'     : hp.uniform  ('sunset__value__cells_bias'    ,  0.80, 1.0),
#     'sunset__fit__hel_r_min'        : hp.uniform  ('sunset__fit__hel_r_min'       ,  0.0, 120.0),
}

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=1000)

print best
print hyperopt.space_eval(space, best)
