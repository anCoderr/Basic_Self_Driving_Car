# Basic Self Driving Car on 2D Map :

## Getting the project up and running :
The `requirement.txt` file contains the conda packages to be installed.
Apart from those we also need to manually install numpy, kivy, matplotlib, pillow using pip as installation from conda frequently throws `.dll not found errors` with the current version of numpy **(is 1.9.0 for conda and 1.21 for pip)**.
***
### Setting up the virtual environment :
- if you want to make virtual env in defualt ~anaconda3/envs/ folder :
>`conda create --name $env_name --file $dir_path_where_requirement_file_is_stored\requirement.txt`
>
>`conda activate env_name` // Activating the virtual environment

- if you want to make virtual env in directory of your choosing (i.e dir_path):
>`conda create --prefix $dir_path\$virtual_env_name --file $dir_path_where_requirement_file_is_stored\requirement.txt`
>
>`conda activate $dir_path\$virtual_env_name` // Activating the virtual environment
***
### Installing some additional packages :
>`pip install numpy==1.21.2 pillow==8.3.2 matplotlib==3.4.2 kivy==2.0.0`

Use `conda list` to check if all packages were installed, mainly :
- Cudatoolkit
- PyTorch
- Numpy
- Matplotlib
- Pillow
- Kivy
***
If pytorch(1.9.0) and cudatoolkit(10.2) are not installed for some reason then use the following command to install them :
- Use this if you have NVIDIA GPU. May take a while as PyTorch + Cudatoolkit is quite big :
>`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` 
                         
- Use this if you dont have a NVIDIA GPU or just want to use PyTorch with CPU (cudatoolkit is hightly recommended) :
>`conda install pytorch torchvision torchaudio cpuonly -c pytorch` 

After the initial setup set the Python Interpreter to the newly created environment in your IDE ***OR*** python kernel for Jupyter Notebook users)
***
### -> Now run the the map.py file.
