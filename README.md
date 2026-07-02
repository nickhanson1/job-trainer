# job-trainer
Claude skill for creating job configurations for av-training, sending jobs to the scheduling server, pausing/resuming jobs, and monitoring jobs.

# How to Install

To install, run the following three commands within the Claude Code terminal. They will add this repository as marketplace, install the plugin, then refresh the context:
'/plugin marketplace add nickhanson1/job-trainer'
'/plugin install job-trainer@job-trainer'
'/plugin reload-plugins'

# How to Use

This skill allows Claude to setup and send training jobs to the scheduler server. Claude will first need to know your username in order to send jobs and properly access your datasets. You need to use the same name you used to upload your datasets from the iPhone app. 

To set up a job, there exist three tiers of options.

## Tier 0

Tier 0 configuration is automated. Claude will choose configuration values entirely for you. You may also tell Claude a little bit about your dataset and intended outcome before running a tier 0 configuration setup, so that Claude has a better idea of what hyperparameters to choose.

## Tier 1

The tier 1 configuration setup provides the basic options for the user that they would find in the Jupyter notebook. They may choose a preset dataset or tell Claude which dataset they would like to use, they may choose the type of model used, the learning rate, etc. Claude may also provide suggestions for which values/models to use, but the choice is up to the user.

## Tier 2

The tier 2 configuration is completely comprehensive. Claude wil run through a checklist of various knobs and parameters that the user can change, including image cropping, neural network layer sizes, and much more. Claude will also sanity check options to ensure the options chosen make sense.

# Fine Tuning

This skill also has a fine-tuning setting, where a model can be fine tuned using extra data. This can mean extra training to lower error, or it can mean training your car to go counter-clockwise if your model currently only can drive clockwise. Ask Claude which options are available.

