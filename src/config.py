common_config = {
    "data_dir": r"C:\Users\KMA62139\OneDrive - Kia\Documents - Big Data, Data Science\Projects\crnn-pytorch\data\WBSIN",
    "img_width": 1440,
    "img_height": 160,
    "map_to_seq_hidden": 64,
    "rnn_hidden": 256,
    "leaky_relu": False,
}

train_config = {
    "epochs": 10000,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "lr": 0.00001,
    "show_interval": 25,
    "valid_interval": 500,
    "save_interval": 2000,
    "cpu_workers": 8,
    "reload_checkpoint": None,
    "valid_max_iter": 100,
    "decode_method": "beam_search",
    "beam_size": 10,
    "checkpoints_dir": "proc_checkpoints/",
}
train_config.update(common_config)

evaluate_config = {
    "eval_batch_size": 512,
    "cpu_workers": 8,
    "reload_checkpoint": "checkpoints/crnn_synth90k.pt",
    "decode_method": "beam_search",
    "beam_size": 10,
}
evaluate_config.update(common_config)