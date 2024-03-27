import cdsapi

for year in range(1982, 2023):
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature', 'soil_temperature_level_1', 'surface_net_solar_radiation',
                'volumetric_soil_water_layer_1', 'mean_sea_level_pressure', 'surface_pressure',
                'total_column_water_vapour', '10m_v_component_of_wind', '10m_u_component_of_wind',
                'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation'
            ],
            'grid': ['0.25', '0.25'],
            'year': str(year),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': '00:00',
        },
        f'path_to_data/raw/era5land_025_025_grid_surface_{year}.nc')

# Note: '10m_v_component_of_wind', 'mean_sea_level_pressure', 'surface_pressure', 'total_column_water_vapour',
# are variables used in the original FourCastNet paper;
# we added the ground variables soil_temperature_level_1,
# and volumetric_soil_water_layer_1, and radiation
