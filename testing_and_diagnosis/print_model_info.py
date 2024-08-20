def printModelInfo(model):
    # Print network architecture
    print(model.policy)

    # Print the current learning rate
    current_learning_rate = model.lr_schedule(1)  # Get the current learning rate (the argument '1' gets the last learning rate in the schedule)
    print(f"Current learning rate: {current_learning_rate}")

    # Print the current PPO clip range
    current_clip_range = model.clip_range(1)  # The argument '1' ensures the latest clip range is retrieved
    print(f"Current PPO clip range: {current_clip_range}")

    # Print the gamma value
    gamma_value = model.gamma
    print(f"Gamma: {gamma_value}")

    # Print the optimizer being used
    optimizer = model.policy.optimizer
    print(f"Optimizer: {optimizer}")


    # Print weight decay (L2 regularization) if applied
    if hasattr(optimizer, 'param_groups'):
        weight_decay = optimizer.param_groups[0].get('weight_decay', 0)
        print(f"Weight Decay (L2 Regularization): {weight_decay}")
    else:
        print("No weight decay (L2 regularization) is applied.")

    # Print the entropy coefficient (used for regularization in PPO)
    entropy_coef = model.ent_coef
    print(f"Entropy Coefficient: {entropy_coef}")

    # Check and print the replay buffer size if applicable
    if hasattr(model, 'replay_buffer'):
        replay_buffer_size = model.replay_buffer.buffer_size
        print(f"Replay buffer size: {replay_buffer_size}")
    else:
        print("Replay buffer is not applicable for the PPO algorithm.")