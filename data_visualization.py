import optuna
import plotly
from optuna import visualization

storage_info = "mysql://root:1115@192.168.50.11:3306/co_mixing"

study = optuna.load_study(study_name="animal10n_unfreeze_weight_latest_additional_models",
                          storage=storage_info)
fg = visualization.plot_slice(study)
fg.write_image('test.jpg')

list_studies = optuna.study.get_all_study_summaries(storage_info)
plotly.io.orca.config.executable = r'/path/to/orca'
plotly.io.orca.config.save()

for study_summary in list_studies:
    plot_dict = {}
    study = optuna.load_study(study_name=study_summary.study_name,
                              storage=storage_info)

    plot_dict['optimization_histogram'] = visualization.plot_optimization_history(study)
    plot_dict['intermediate_values'] = visualization.plot_intermediate_values(study)
    plot_dict['plot_parallel_coordinate'] = visualization.plot_parallel_coordinate(study)
    plot_dict['plot_contour'] = visualization.plot_contour(study)
    plot_dict['plot_slice'] = visualization.plot_slice(study)
    plot_dict['plot_param_importance'] = visualization.plot_param_importances(study)
    plot_dict['plot_edf'] = visualization.plot_edf(study)

    for name, fig in plot_dict.items():
        dir_plot = f"charts/{study_summary.study_name}/{name}.png"
        fig.update_layout(width=2000, height=2000)
        fig.write_image(dir_plot)
        # plotly.io.write_image(fig, dir_plot, engine='orca')