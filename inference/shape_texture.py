from modelvshuman import Evaluate, Plot
from modelvshuman.plotting.decision_makers import DecisionMaker

orange = '#fc8e62'
purple = '#412F8A'


def plotting_definition_template(df, models):
    decision_makers = []

    colors = [orange, purple]
    markers = ['o', '^']

    for i, model in enumerate(models):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        decision_makers.append(
            DecisionMaker(name_pattern=model,
                          color=color,
                          marker=marker,
                          df=df,
                          plotting_name=model))

    return decision_makers


def run_evaluation(models):
    datasets = ["cue-conflict"]
    params = {
        "batch_size": 64,
        "pretrained_dir": "../pretrained",
        "print_predictions": True,
        "num_workers": 20
    }
    Evaluate()(models, datasets, **params)


def run_plotting(models):
    plot_types = ["shape-bias"]
    plotting_def = lambda df: plotting_definition_template(df, models)
    figure_dirname = f"./outputs/shape_texture/{'-'.join(models)}/"  # Unique folder for each experiment
    Plot(plot_types=plot_types,
         plotting_definition=plotting_def,
         figure_directory_name=figure_dirname,
         crop_PDFs=False)


if __name__ == "__main__":
    experiments = [["deit3_21k", "convnext_base_21k"],
                   ["vit_clip", "convnext_clip"]]

    for models in experiments:
        run_evaluation(models)
        run_plotting(models)
