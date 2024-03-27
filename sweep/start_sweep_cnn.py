import wandb


def get_sweep_configuration():
    sweep_configuration = {
        "method": "bayes",
        "metric": {
            "name": "optim_target",
            "goal": "minimize"
        },
        "parameters": {
            "n_filters_l1": {
                "values": [16, 32, 64, 128, 256, 512]
            },
            "n_filters_l2": {
                "values": [16, 32, 64, 128, 256, 512]
            },
            "n_filters_l3": {
                "values": [16, 32, 64, 128, 256, 512]
            },
            "n_filters_l4": {
                "values": [16, 32, 64, 128, 256, 512]
            },
            "n_filters_l5": {
                "values": [16, 32, 64, 128, 256, 512]
            },
            "n_filters_l6": {
                "values": [16, 32, 64, 128, 256, 512]
            },
            "n_filters_l7": {
                "values": [16, 32, 64, 128, 256, 512]
            },
            "n_filters_l8": {
                "values": [16, 32, 64, 128, 256, 512]
            },
            "kernel_size_l1": {
                "values": [3, 5, 7]
            },
            "kernel_size_l2": {
                "values": [3, 5, 7]
            },
            "kernel_size_l3": {
                "values": [3, 5, 7]
            },
            "kernel_size_l4": {
                "values": [3, 5, 7]
            },
            "kernel_size_l5": {
                "values": [3, 5, 7]
            },
            "kernel_size_l6": {
                "values": [3, 5, 7]
            },
            "kernel_size_l7": {
                "values": [3, 5, 7]
            },
            "kernel_size_l8": {
                "values": [3, 5, 7]
            },

            "n_layers": {
                "values": [3, 4, 5, 6, 7, 8]
            },
            "lr":  # learning rate
                {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 1e-1
                },
            "epochs":
                {
                    "values": [50, 60, 70, 80, 90, 100]
                },
        }
    }

    return sweep_configuration


sweep_id = wandb.sweep(sweep=get_sweep_configuration(), project="", entity="")

print(f"sweep id: {sweep_id}")
