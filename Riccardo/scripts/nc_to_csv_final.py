import pandas as pd
import xarray as xr
import argparse
import numpy as np
from joblib import Parallel, delayed


def extract_years_parallelized(year, min_lat, max_lat, min_lon, max_lon, origin, path_features, path_target):
    df_full = pd.DataFrame()
    for var in variables:
        if origin == 'features':
            ds = xr.open_dataset(path_features + '/2d_clint_' + str(year) + '0101_' + str(year) + '1231.grb',
                    engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': var}})
        else:
            ds = xr.open_dataset(path_target + '/2.5col_obs_' + str(year) + '_48_17.grb', engine='cfgrib')
        df = ds.to_dataframe().reset_index()
        df = df.loc[(df.latitude >= min_lat) & (df.longitude >= min_lon) & (df.latitude <= max_lat) & (
                        df.longitude <= max_lon)]
        df_full['latitude'] = df.reset_index().latitude
        df_full['longitude'] = df.reset_index().longitude
        df_full['time'] = df.reset_index().time
        if var in ['u', 'v']:
            df_red = df.loc[df.isobaricInhPa == 200, [var]].reset_index(drop=True).rename(
                columns={'u': 'u_200', 'v': 'v_200'})
            df_full = pd.concat([df_full, df_red], axis=1)
            df_red = df.loc[df.isobaricInhPa == 850, [var]].reset_index(drop=True).rename(
                columns={'u': 'u_850', 'v': 'v_850'})
            df_full = pd.concat([df_full, df_red], axis=1)
        else:
            df_red = df.reset_index(drop=True).loc[:, [var]]
            df_full = pd.concat([df_full, df_red], axis=1)
    return df_full
                

def extract_from_nc(min_year, max_year, min_lat, max_lat, min_lon, max_lon, origin, path_features, path_target):
    df_all = pd.DataFrame()

    df_full_list = Parallel(n_jobs=3)(delayed(extract_years_parallelized)(
        year, min_lat, max_lat, min_lon, max_lon, origin, path_features, path_target) for year in range(min_year, max_year + 1))
    
    # df_full_list = [extract_years_parallelized(
    #     year, min_lat, max_lat, min_lon, max_lon, origin, path_features, path_target) for year in range(min_year, max_year + 1)]
    for df_full in df_full_list:
        df_all = pd.concat([df_all, df_full])    
    return df_all


def extract_from_nc_cross(min_year, max_year, min_lat, max_lat, min_lon, max_lon, min_lon2, max_lon2, origin,
                          path_features, path_target):
    df_all = pd.DataFrame()
    for year in range(min_year, max_year + 1):
        df_full = pd.DataFrame()

        for var in variables:
            if origin == 'features':
                ds = xr.open_dataset(
                    path_features + '/2d_clint_' + str(year) + '0101_' + str(
                        year) + '1231.grb',
                    engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': var}})
            else:
                ds = xr.open_dataset(
                    path_target + '/2.5col_obs_' + str(year) + '_48_17.grb', engine='cfgrib')
            df = ds.to_dataframe().reset_index()
            df = df.loc[((df.latitude >= min_lat) & (df.longitude >= min_lon) & (df.latitude <= max_lat) & (
                    df.longitude <= max_lon)) |
                        ((df.latitude >= min_lat) & (df.longitude >= min_lon2) & (df.latitude <= max_lat) & (
                                df.longitude <= max_lon2))]
            df_full['latitude'] = df.reset_index().latitude
            df_full['longitude'] = df.reset_index().longitude
            df_full['time'] = df.reset_index().time
            if var in ['u', 'v']:
                df_red = df.loc[df.isobaricInhPa == 200, [var]].reset_index(drop=True).rename(
                    columns={'u': 'u_200', 'v': 'v_200'})
                df_full = pd.concat([df_full, df_red], axis=1)
                df_red = df.loc[df.isobaricInhPa == 850, [var]].reset_index(drop=True).rename(
                    columns={'u': 'u_850', 'v': 'v_850'})
                df_full = pd.concat([df_full, df_red], axis=1)
            else:
                df_red = df.reset_index(drop=True).loc[:, [var]]
                df_full = pd.concat([df_full, df_red], axis=1)
        df_all = pd.concat([df_all, df_full])
        print(year)
    return df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_lat", default=-30, type=float)
    parser.add_argument("--max_lat", default=0, type=float)
    parser.add_argument("--min_lon", default=20, type=float)
    parser.add_argument("--max_lon", default=90, type=float)
    parser.add_argument("--min_lon2", default=20, type=float)
    parser.add_argument("--max_lon2", default=90, type=float)
    parser.add_argument("--min_year", default=1980, type=int)
    parser.add_argument("--max_year", default=2022, type=int)
    parser.add_argument("--savepath_features",
                        default = '/work/bk1318/b382633/data_csv/predictors_v2_dailymean/features_dailymeans_Sindian.csv')
    parser.add_argument("--savepath_features_train",
                        default = '/work/bk1318/b382633/data_csv/predictors_v2_dailymean/train_dailymeans_Sindian.csv')
    parser.add_argument("--savepath_features_val",
                        default = '/work/bk1318/b382633/data_csv/predictors_v2_dailymean/val_dailymeans_Sindian.csv')
    parser.add_argument("--savepath_features_test",
                        default = '/work/bk1318/b382633/data_csv/predictors_v2_dailymean/test_dailymeans_Sindian.csv')
    parser.add_argument("--savepath_target",
                        default = '/work/bk1318/b382633/data_csv/predictors_v2_dailymean/target_dailymeans_Sindian.csv')
    parser.add_argument("--savepath_target_train",
                        default = '/work/bk1318/b382633/data_csv/predictors_v2_dailymean/target_train_dailymeans_Sindian.csv')
    parser.add_argument("--savepath_target_val",
                        default = '/work/bk1318/b382633/data_csv/predictors_v2_dailymean/target_val_dailymeans_Sindian.csv')
    parser.add_argument("--savepath_target_test",
                        default = '/work/bk1318/b382633/data_csv/predictors_v2_dailymean/target_test_dailymeans_Sindian.csv')
    parser.add_argument("--sourcepath_features", default= '/work/bk1318/b382633/data_grib/predictors_v2_dailymean/')
    parser.add_argument("--sourcepath_target", default= '/work/bk1318/b382633/data_grib/target')
    parser.add_argument("--crossing", default='False')

    args = parser.parse_args()
    print(args)

    ############ features ############

    variables = ['vo', 'r', 'u', 'v', 'sst', 'tcwv', 'tclw', 'tciw']

    if args.crossing == 'True':
        df_all = extract_from_nc_cross(args.min_year, args.max_year, args.min_lat, args.max_lat, args.min_lon,
                                       args.max_lon, args.min_lon2, args.max_lon2, 'features', args.sourcepath_features,
                                       args.sourcepath_target)
    else:
        df_all = extract_from_nc(args.min_year, args.max_year, args.min_lat, args.max_lat, args.min_lon,
                                 args.max_lon, 'features', args.sourcepath_features, args.sourcepath_target)

    df_all = df_all.reset_index(drop=True)
    print(df_all)
    df_all.to_csv(args.savepath_features)

    train = df_all.loc[df_all.time <= '2010-12-31'].reset_index(drop=True).fillna(0)
    val = df_all.loc[(df_all.time >= '2011-01-01') & (df_all.time <= '2015-12-31')].reset_index(drop=True).fillna(0)
    test = df_all.loc[(df_all.time >= '2016-04-01') & (df_all.time <= '2022-12-31')].reset_index(drop=True).fillna(0)

    train['shear'] = np.sqrt((train.u_200 - train.u_850) ** 2 + (train.v_200 - train.v_850) ** 2)
    val['shear'] = np.sqrt((val.u_200 - val.u_850) ** 2 + (val.v_200 - val.v_850) ** 2)
    test['shear'] = np.sqrt((test.u_200 - test.u_850) ** 2 + (test.v_200 - test.v_850) ** 2)

    train.to_csv(args.savepath_features_train)
    val.to_csv(args.savepath_features_val)
    test.to_csv(args.savepath_features_test)

    ############ target ############

    variables = ['p131089']

    if args.crossing == 'True':
        df_all = extract_from_nc_cross(args.min_year, args.max_year, args.min_lat, args.max_lat, args.min_lon,
                                       args.max_lon, args.min_lon2, args.max_lon2, 'target', args.sourcepath_features,
                                       args.sourcepath_target)
    else:
        df_all = extract_from_nc(args.min_year, args.max_year, args.min_lat, args.max_lat, args.min_lon, args.max_lon,
                                 'target', args.sourcepath_features, args.sourcepath_target)

    df_all = df_all.reset_index(drop=True)
    print(df_all)
    df_all.to_csv(args.savepath_target)

    df_all['new_target'] = df_all['p131089']
    train = df_all.loc[df_all.time <= '2010-12-31'].reset_index(drop=True).fillna(0)
    val = df_all.loc[(df_all.time >= '2011-01-01') & (df_all.time <= '2015-12-31')].reset_index(drop=True).fillna(0)
    test = df_all.loc[(df_all.time >= '2016-04-01') & (df_all.time <= '2022-12-31')].reset_index(drop=True).fillna(0)

    train.to_csv(args.savepath_target_train)
    val.to_csv(args.savepath_target_val)
    test.to_csv(args.savepath_target_test)
