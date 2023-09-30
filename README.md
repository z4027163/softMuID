# Training MVA with XGBoost

## Code organization

The code works for a wide variaty of tasks. The core is in
ModelHandler.py, which is a wrapper around XGBoost. All that you need
to do is to create a new class that inherits from ModelHandler,
provide feature names, input files and tune the parameters.

`plot_results_*` are plain Python scripts to make comparison plots

## Prepare Input Datasets

We use flat ntuples produced with
https://github.com/drkovalskyi/Bmm5/blob/master/NanoAOD/postprocess/FlatNtupleForBmmMva.py
code

##Frame set up
same as: https://github.com/drkovalskyi/Bmm5

The content is under MVA/ folder
## Examples

`python3 train_muon_mva.py`

# compare ROC from different trainings
`python3 study_muon_mva.py`
