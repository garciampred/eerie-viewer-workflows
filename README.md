# eerie-data-viewer-workflows

Functions and workflows for preparing data for visualization for the EERIE project (https://eerie-project.eu/) data viewer. The input
data is read from the EERIE intake catalogues and, in the case of observations,
from netCDF files. 

IMPORTANT NOTE: The EERIE data is published in almost real time and is continously updated.
It can cointain issues like inconsistent units or other encoding errors. The scripts do correct some of 
the issues and homogeneize the variable names, but some issues may persist or arise as the data is updated.
If you aim to use this script and EERIE cloud data please review the results critically. Do not consider
this "production ready".

The present code is also changing and evolving. Please report any bugs you may find in the issue tracker.

## Instalation

It is recommended to clone the project and create a conda environment with the environment.yml file. Then 
the project root can be simply added to the PYTHONPATH.

```commandline
git clone git@github.com:eerie-project/eerie-viewer-workflows.git
cd eerie-viewer-workflows
conda env create -f environment.yml -n eerieview
conda activate eerieview
export PYTHONPATH="$PWD"
cd scripts
python get_climatologies.py
```

## Main scripts available

The scripts/entrypoints are in the scripts folder. For them to run the root directory needs to be in the
PYTHONPATH. The following are the main scripts:

* get_climatologies: It computes the decadal products from the EERIE data, which are climatologies but also trends.
* get_obs_climatologies: Computes climatologies and trends for observations. Currently ERA5 and AVISO data are  supported. These are not read from intake catalogues but need to be present as files in the disk.
* download_era5.py: Script to download ERA5 filse
* get_monthly.eke: Computes the monthly Eddy Kinetic Energy from daily sea level data from EERIE models.
* get_aviso_monthly_variables.py: Computes monthly data from the AVISO observations.
* get_time_series: Computes regionally averaged time series from EERIE data
* get_obs_time_series: Computes regionally averaged time series from the observations.
* upload_to_zarr.py: get_climatologies.py et al. generate netCDF files. This scripts merges them and uploads them to an object storage as zarr datasets.
* plot_stripes.py: This script is used for Quality Control. It generates figures by systematically reading all the fields from the zarr files, in order to inspect them looking for gaps or suspicious patterns.


## Workflow for developers/contributors

For best experience create a new conda environment (e.g. DEVELOP) with Python 3.11:

```
conda create -n DEVELOP -c conda-forge python=3.11
conda activate DEVELOP
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Run quality assurance checks: `make qa`
1. Run tests: `make unit-tests`
1. Run the static type checker: `make type-check`

## License

```
Copyright 2025, European Union.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
