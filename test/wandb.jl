using Wandb, Logging

# Initialize the project
lg = WandbLogger(; project = "Wandb.jl", name = nothing)

# Set logger globally / in scope / in combination with other loggers
global_logger(lg)

# Logging Values
Wandb.log(lg, Dict("accuracy" => 0.9, "loss" => 0.3))

# Even more conveniently
@info "metrics" accuracy=0.9 loss=0.3
@debug "metrics" not_print=-1  # Will have to change debug level for this to be logged

# Tracking Hyperparameters
update_config!(lg, Dict("dropout" => 0.2))

# Close the logger
close(lg)