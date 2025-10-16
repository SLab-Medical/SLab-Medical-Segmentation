import numpy as np

def build_dataset(args, mode):
    
    if args.dataset.dataset_name == 'toothfairy':
        from .toothfairy import ToothFairy
        # TODO
        return ToothFairy(args.dataset)

    elif args.dataset.dataset_name == 'liuxiangyue':
        from .liuxiangyue import liuxiangyue

        return liuxiangyue(args.dataset)