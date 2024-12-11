


def get_device_map(model_id, n_gpus):
    if model_id == "meta-llama/Meta-Llama-3-8B-instruct" and n_gpus == 4:
        return {'model.embed_tokens': 0
                ,'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0
                ,'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 1
                ,'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1
                ,'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1
                ,'model.layers.20': 1, 'model.layers.21': 2, 'model.layers.22': 2, 'model.layers.23': 2, 'model.layers.24': 2
                ,'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 2, 'model.layers.28': 2, 'model.layers.29': 2,
                'model.layers.30': 2, 'model.layers.31': 3, 'model.norm': 3, 'lm_head': 3}
    elif model_id == "Aleph-Alpha/Pharia-1-LLM-7B-control-hf" and n_gpus == 4:
        return {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0
                , 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 1, 'model.layers.8': 1
                , 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1
                , 'model.layers.14': 1, 'model.layers.15': 2, 'model.layers.16': 2, 'model.layers.17': 2, 'model.layers.18': 2
                , 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2, 'model.layers.23': 3
                , 'model.layers.24': 3, 'model.layers.25': 3, 'model.layers.26': 3, 'model.norm': 3, 'lm_head': 3}
    else:
        return "auto"