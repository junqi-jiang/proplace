# Provably Robust and Plausible Counterfactual Explanations

## Requirements
- conda create --name proplace python==3.7.13
- conda install pytorch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
- pip install git+https://github.com/carla-recourse/carla.git
- pip install notebook
- pip install gurobipy==10.0.2
- pip install dice-ml==0.9
- pip install alibi==0.8.0
- pip install tensorflow==1.15.0 protobuf==3.20.3 choix
- pip install tabulate
- Change alibi/explainers/cfproto.py::123, tf.keras.backend.get_session(). This is to resolve the package compatibility issue

To reproduce the results in Table 1, simply run the notebooks in /experiments.

Gurobi requires an active license to run. Free academic licenses for students, staff, and academics: https://www.gurobi.com/academia/academic-program-and-licenses/

