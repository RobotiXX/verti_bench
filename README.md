# gym-chrono

This repository consists of a set of gymnasium "environments" which are essentially wrappers around pychrono. In order to install gym-chrono, we must first install its dependecies
1) [pychrono](https://api.projectchrono.org/pychrono_installation.html)
2) [gymnasium](https://pypi.org/project/gymnasium/)
3) [stable-baselines3[extra]](https://pypi.org/project/stable-baselines3/)

## Downloading data files
Before you begin the installation process, you will need to download the `data` folder containing the simulation assets and place it in the right place:
1) Download the data files [here](https://gmuedu-my.sharepoint.com/:f:/r/personal/txu25_gmu_edu/Documents/data?csf=1&web=1&e=BdSEtA), unzip if necessary, you should obtain a folder named `data`.
2) Copy the data to `DIR_OF_REPO/gym-chrono/envs`.

#### Adding Chrono data directory to path
Once the data folder has been downloaded and placed in the right folder, it needs to be added to path:  
For Linux or Mac users:  
  Replace bashrc with the shell your using. Could be `.zshrc`.  
  1. echo `export CHRONO_DATA_DIR=<Downloaded data directory path>' >> ~/.bashrc`  
      Ex. If you have cloned the repository in `home` , then, echo `export CHRONO_DATA_DIR=/home/user/gym-chrono/gym-chrono/envs/data/' >> ~/.bashrc`  
  2. `source ~/.bashrc`

For Windows users:  
  Link as reference: https://helpdeskgeek.com/how-to/create-custom-environment-variables-in-windows/  
  1. Open the System Properties dialog, click on Advanced and then Environment Variables  
  2. Under User variables, click New... and create a variable as described below  
      Variable name: CHRONO_DATA_DIR  
      Variable value: <chrono's data directory>  
          Ex. Variable value: C:\ Users\ user\ chrono\ data\

## Installing dependencies
### Installing pychrono
1) First you need to install pychrono from source. The Chrono source that needs to be cloned is linked [here](https://github.com/xutong05/chrono/tree/feature/robot_model). Please use the feature/robot_model branch. We use this fork with this branch because it contains all the latest robot models that are not currently available in Chrono main.
2) Once you have the source cloned, build pychrono from source using instructions found (1)[Install PyChrono B](https://api.projectchrono.org/pychrono_installation.html). (2)[Install Modules](https://api.projectchrono.org/install_guides.html). Enable modules Chrono::Sensor, Chrono::Irrlicht, Chrono::SynChrono, Chrono::Vehicle, Chrono::Python, Chrono::OPENMP and Chrono::Parsers. For each of these modules, please look at the official Chrono documentation.
3) Make sure you add the appropriate numpy include directory (see linked instructions above)
4) If you are not doing a system wide install of pychrono, make sure you add to PYTHONPATH the path to the installed python libraries (see linked instructions above)
### Installing gymnasium
```bash
pip install gymnasium
```
> [!NOTE]
> If you are using a conda environment, activate the conda environment and then use the same command above.  

### Installing stable-baselines3
```bash
pip install stable-baselines3[extra] 
```

> [!NOTE]
> `stable-baselines3` installs nupmy as a dependency, so it is recomended to remove this installation and install your own version of numpy. Additionally, `pychrono` requires `numpy=1.24.0`, and it must be installed with conda, so it is necessary to run `pip uninstall numpy` and `conda install -c conda-forge numpy=1.24.0` to not get a `pychrono.sensor` error.
### Rough Edges
#### Adding gym-chrono to path
Due to the lack of a pip installer for this package currently, you must add gym-chrono to `PYTHONPATH`:
```
 echo 'export PYTHONPATH=$PYTHONPATH:<path to gym-chrono>' >> ~/.bashrc
```
Replace `~/.bashrc` with `~/.zshrc` in case you are using `zsh`.<br>
For Windows users, follow instructions from [here](https://helpdeskgeek.com/how-to/create-custom-environment-variables-in-windows/).

     
## Repository Structure

This repository is structured as follows:
1. Within the `gym-chrono` folder is all that you need:
   - **baselines**: Two methods to compare with RL 
   - **env**: gymnasium environment wrapper to enable RL training using PyChrono simulation
   - **test**: testing scripts to visualize the training environment and debug it
   - **train**: python scripts to train the models for each example env with stable-baselines3
   - **evaluate**: python scripts to evaluate a trained model
2. The `demo` folder consists of a video of the environment on rock terrain!

