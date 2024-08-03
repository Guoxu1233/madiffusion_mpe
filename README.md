# MADiff only for mpe


### Setup MPE

We use the MPE dataset from [OMAR](https://github.com/ling-pan/OMAR). The dataset download link and instructions can be found in OMAR's [repo](https://github.com/ling-pan/OMAR). Since their BaiduPan download links might be inconvenient for non-Chinese users, we maintain a anonymous mirror [repo](https://osf.io/jxawh/?view_only=dd3264a695af4c03bffde0350b8e8c4a) in OSF for acquiring the dataset.

The downloaded dataset should be placed under `diffuser/datasets/data/mpe`.

Install MPE environment:

```bash
pip install -e third_party/multiagent-particle-envs
pip install -e third_party/ddpg-agent
```


## Training and Evaluation
To start training, run the following commands

```bash
# multi-agent particle environment
python run_experiment.py -e exp_specs/mpe/<task>/mad_mpe_<task>_attn_<dataset>.yaml  # CTCE
python run_experiment.py -e exp_specs/mpe/<task>/mad_mpe_<task>_ctde_<dataset>.yaml  # CTDE
```

To evaluate the trained model, first replace the `log_dir` with those need to be evaluated in `exp_specs/eval_inv.yaml` and run
```bash
python run_experiment.py -e exp_specs/eval_inv.yaml
```

## Acknowledgements

The codebase is built upon [madiff repo](https://github.com/zbzhu99/madiff). Thanks!
