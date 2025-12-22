import numpy as np

def build_dataset(args, mode):
    
    if args.dataset.dataset_name == 'toothfairy':
        from .toothfairy import ToothFairy
        # TODO
        return ToothFairy(args.dataset)

    elif args.dataset.dataset_name == 'liuxiangyue':
        from .liuxiangyue import liuxiangyue

        return liuxiangyue(args.dataset)
    
    elif args.dataset.dataset_name == 'brat':
        from .brat import Brat

        return Brat(args.dataset)

    elif args.dataset.dataset_name == 'dsa':
        from .dsa import DSAWithTorchio

        return DSAWithTorchio(args.dataset, mode=mode)