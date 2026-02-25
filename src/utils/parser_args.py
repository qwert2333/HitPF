import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--freeze-clustering",
    action="store_true",
    default=False,
    help="Freeze the clustering part of the model",
)


parser.add_argument("-c", "--data-config", type=str, help="data config YAML file")

parser.add_argument(
    "-i",
    "--data-train",
    nargs="*",
    default=[],
    help="training files; supported syntax:"
    " (a) plain list, `--data-train /path/to/a/* /path/to/b/*`;"
    " (b) (named) groups [Recommended], `--data-train a:/path/to/a/* b:/path/to/b/*`,"
    " the file splitting (for each dataloader worker) will be performed per group,"
    " and then mixed together, to ensure a uniform mixing from all groups for each worker.",
)
parser.add_argument(
    "-l",
    "--data-val",
    nargs="*",
    default=[],
    help="validation files; when not set, will use training files and split by `--train-val-split`",
)
parser.add_argument(
    "-t",
    "--data-test",
    nargs="*",
    default=[],
    help="testing files; supported syntax:"
    " (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;"
    " (b) keyword-based, `--data-test a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;"
    " (c) split output per N input files, `--data-test a%10:/path/to/a/*`, will split per 10 input files",
)

parser.add_argument(
    "--data-fraction",
    type=float,
    default=1,
    help="fraction of events to load from each file; for training, the events are randomly selected for each epoch",
)
parser.add_argument(
    "--file-fraction",
    type=float,
    default=1,
    help="fraction of files to load; for training, the files are randomly selected for each epoch",
)
parser.add_argument(
    "--fetch-by-files",
    action="store_true",
    default=False,
    help="When enabled, will load all events from a small number (set by ``--fetch-step``) of files for each data fetching. "
    "Otherwise (default), load a small fraction of events from all files each time, which helps reduce variations in the sample composition.",
)
parser.add_argument(
    "--fetch-step",
    type=float,
    default=0.01,
    help="fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); "
    "Or: number of files to load each time (when ``--fetch-by-files`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.",
)

parser.add_argument(
    "--train-val-split",
    type=float,
    default=0.8,
    help="training/validation split fraction",
)


parser.add_argument(
    "-n",
    "--network-config",
    type=str,
    help="network architecture configuration file; the path must be relative to the current dir",
)
parser.add_argument(
    "-m",
    "--model-prefix",
    type=str,
    default="models/{auto}/networkss",
    help="path to save or load the model; for training, this will be used as a prefix, so model snapshots "
    "will saved to `{model_prefix}_epoch-%d_state.pt` after each epoch, and the one with the best "
    "validation metric to `{model_prefix}_best_epoch_state.pt`; for testing, this should be the full path "
    "including the suffix, otherwise the one with the best validation metric will be used; "
    "for training, `{auto}` can be used as part of the path to auto-generate a name, "
    "based on the timestamp and network configuration",
)

parser.add_argument(
    "--load-model-weights",
    type=str,
    default=None,
    help="initialize model with pre-trained weights",
)
parser.add_argument(
    "--load-model-weights-clustering",
    type=str,
    default=None,
    help="initialize model with pre-trained weights for clustering part of the model",
)
parser.add_argument("--start-lr", type=float, default=5e-3, help="start learning rate")

parser.add_argument("--num-epochs", type=int, default=20, help="number of epochs")
parser.add_argument(
    "--steps-per-epoch",
    type=int,
    default=None,
    help="number of steps (iterations) per epochs; "
    "if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples",
)
parser.add_argument(
    "--steps-per-epoch-val",
    type=int,
    default=None,
    help="number of steps (iterations) per epochs for validation; "
    "if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples",
)
parser.add_argument(
    "--samples-per-epoch",
    type=int,
    default=None,
    help="number of samples per epochs; "
    "if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples",
)
parser.add_argument(
    "--samples-per-epoch-val",
    type=int,
    default=None,
    help="number of samples per epochs for validation; "
    "if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples",
)
parser.add_argument("--batch-size", type=int, default=128, help="batch size")

parser.add_argument(
    "--gpus",
    type=str,
    default="0",
    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`',
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help="number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers",
)
parser.add_argument(
    "--prefetch-factor",
    type=int,
    default=1,
    help="How many items to prefetch in the dataloaders. Should be about the same order of magnitude as batch size for optimal performance.",
)
parser.add_argument(
    "--predict",
    action="store_true",
    default=False,
    help="run prediction instead of training",
)




parser.add_argument(
    "--log-wandb", action="store_true", default=False, help="use wandb for loging"
)
parser.add_argument(
    "--wandb-displayname",
    type=str,
    help="give display name to wandb run, if not entered a random one is generated",
)
parser.add_argument(
    "--wandb-projectname", type=str, help="project where the run is stored inside wandb"
)
parser.add_argument(
    "--wandb-entity", type=str, help="username or team name where you are sending runs"
)


parser.add_argument(
    "--qmin", type=float, default=0.1, help="define qmin for condensation"
)


parser.add_argument(
    "--frac_cluster_loss",
    type=float,
    default=0,
    help="Fraction of total pairs to use for the clustering loss",
)





parser.add_argument(
    "--use-average-cc-pos",
    default=0.0,
    type=float,
    help="push the alpha to the mean of the coordinates in the object by this value",
)


parser.add_argument(
    "--correction",
    action="store_true",
    default=False,
    help="Train correction only",
)




parser.add_argument(
    "--use-gt-clusters",
    default=False,
    action="store_true",
    help="If toggled, uses ground-truth clusters instead of the predicted ones by the model. We can use this to simulate 'ideal' clustering.",
)


parser.add_argument(
    "--name-output",
    type=str,
    help="name of the dataframe stored during eval",
)
parser.add_argument(
    "--train-batches",
    default=100,
    type=int,
    help="number of train batches",
)
parser.add_argument(
    "--pandora",
    default=False,
    action="store_true",
    help="using pandora information",
)
