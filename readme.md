## GAM: General Affordance-based Manipulation for Robotic Object Disentangling Tasks
- [Paper](https://www.sciencedirect.com/science/article/pii/S0925231224001577)

### Installation guide
- Create conda env with the provided environment.yml file.
- Install [graspnet-baseline](https://github.com/graspnet/graspnet-baseline).
- Install [mujoco and mujoco-py](https://github.com/openai/mujoco-py) (the experiments use 2.1.0 version, others may work).
- Make sure gym version <=0.20.0.
- From repository root, run `python -m pip install .`.

## Data
- Follow https://github.com/IanYangChina/GAM-paper-data to download the data.
- Place the SG_data folder in the root directory of this repo.

## Pretrained agents
- There are some pre-trained checkpoints that can be downloaded [onedrive](https://cf-my.sharepoint.com/:f:/g/personal/yangx66_cardiff_ac_uk/EoeMc7qgg_VIuK4u-Lmh1cABuFj4SmNPqd3Ds3NVP5mmKw?e=pUyhH6).
- Place the result folders in the root directory (e.g., GAM-paper-codes/results_3).
- Namings:
	- C/C+/S: hook shapes
	- gf: grasp filter checkpoints
	- hm: hemisphere action checkpoints
	- 3/30: num of episode timesteps

## Scripts
- With the data and checkpoints, you may play around with evaluations.
	- E.g., `python script/evaluate_dqn --render --nenv 1 --hs C --nh 3`
- Other scripts for training, generating dataset are also available.

## Citation
```
@article{yang2024gam,
  title={GAM: General affordance-based manipulation for contact-rich object disentangling tasks},
  author={Yang, Xintong and Wu, Jing and Lai, Yu-Kun and Ji, Ze},
  journal={Neurocomputing},
  pages={127386},
  year={2024},
  publisher={Elsevier}
}
```
