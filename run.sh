#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco210/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
#python run_experiment.py -e exp_specs/mpe/tag/mad_mpe_tag_ctde_exp.yaml
#python run_experiment.py -e exp_specs/mpe/spread/mad_mpe_spread_attn_exp.yaml
python run_experiment.py -e exp_specs/mpe/tag/mad_mpe_tag_attn_exp.yaml