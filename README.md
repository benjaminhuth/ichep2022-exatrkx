# Training, inferance and performance evaluation scripts for Exa.TrkX+ACTS+ODD

This repository collects some scripts that have been used for the ICHEP contribution [Applying and optimizing the Exa.TrkX Pipeline on the OpenDataDetector with ACTS](https://agenda.infn.it/event/28874/contributions/169199/). For help, contact *benjamin.huth [at] ur.de*.

**Note**: This repository uses git lfs to store the root files.
**Warning**: These files may contain absolute paths, you may need to edit them in order to use all of them in your environment.

## Training

The training scripts are in the subfolder `traintrack_configs` and consists of scripts for two training runs: one with smeared data, one with true hit data (each steered by the respective `run.sh` script). The training data can be produced with `datagen.py generate <output dir> <number events>`.

## Inference

The inference can be run with `datagen.py reconstruct <output dir> <number events>`. It either takes pytorch lighnting checkpoint paths from a `model_config.txt` file and converts them directly to `*.pt` files, or `*.pt` files from the `<output dir>/torchscript` path. In the first way, the correct hyperparameters are directly taken from the model, in the latter case they may need to be adjusted either with the `--overwrite_config` option or by editing the file directly.

## Performance evaluation

The subfolders `inference_results_smeared` and `inference_results_truth` contain `make_data.sh` scripts, that run the `datagen.py` script. They also contain `*.pt` files with the used torchscript models and `*.root` files containing performance results. The top-level `make_plots.py` makes performance plots out of these root files.s
