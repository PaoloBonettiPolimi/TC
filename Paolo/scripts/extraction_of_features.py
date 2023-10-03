import pandas as pd
import xarray as xr


if __name__ == "__main__":

    # can be improved setting these parameters as args, together with the name of the output file and the path
    min_lat = -30
    max_lat = 0
    min_lon = 20
    max_lon = 90
    min_year = 1980
    max_year = 2022

    variables = ['vo','r','u','v','sst','tcwv','tclw','tciw']

    df_all = pd.DataFrame()

    for year in range(min_year,max_year+1):
        df_full = pd.DataFrame()
        
        for var in variables:
            ds = xr.open_dataset('/Users/paolo/Desktop/tc_fullDataset/2d_clint_'+str(year)+'0101_'+str(year)+'1231.grb',
                            engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': var}})
        
            df = ds.to_dataframe().reset_index()
            df = df.loc[(df.latitude>=min_lat) & (df.longitude>=min_lon) & (df.latitude<=max_lat) & (df.longitude<=max_lon)]
            df_full['latitude'] = df.reset_index().latitude
            df_full['longitude'] = df.reset_index().longitude
            df_full['time'] = df.reset_index().time
            if var in ['u','v']:
                df_red = df.loc[df.isobaricInhPa==200,[var]].reset_index(drop=True).rename(columns={'u':'u_200','v':'v_200'})
                df_full = pd.concat([df_full,df_red],axis=1)
                df_red = df.loc[df.isobaricInhPa==850,[var]].reset_index(drop=True).rename(columns={'u':'u_850','v':'v_850'})
                df_full = pd.concat([df_full,df_red],axis=1)
            else:
                df_red = df.reset_index(drop=True).loc[:,[var]]
                df_full = pd.concat([df_full,df_red],axis=1)
        df_all = pd.concat([df_all,df_full])
        print(year)

    df_all = df_all.reset_index(drop=True)
    print(df_all)
    df_all.to_csv('features_all_new.csv')