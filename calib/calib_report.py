import optuna.visualization as vis
import optuna

def plot_stuff( study_name, storage_url ):
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()
    vis.plot_slice(study).show()
    vis.plot_contour(study, params=["r0", "gravity_k"]).show() # pick any 2 params

