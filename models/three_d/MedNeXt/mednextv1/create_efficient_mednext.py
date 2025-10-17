#from networks.MedNeXt.mednextv1.MD_MedNextOV1 import MD_MedNeXtO
from models.three_d.MedNeXt.mednextv1.EfficientMedNext import EfficientMedNeXt

from models.three_d.MedNeXt.mednextv1.EfficientMedNext_Full import EfficientMedNeXt_L
#from networks.MedNeXt.mednextv1.MD_MedNextOVF import MD_MedNeXtO

def create_efficient_mednext_tiny(num_input_channels, num_classes, n_channels=32, kernel_sizes=[1,3,5], strides=[1,1,1], uniform_dec_channels=None, ds=False, mode='train'):

    return EfficientMedNeXt(
        in_channels = num_input_channels, 
        n_channels = n_channels,
        n_classes = num_classes, 
        kernel_sizes=kernel_sizes,
        strides=strides,
        uniform_dec_channels=uniform_dec_channels,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2],
        checkpoint_style = 'outside_block',
        mode = mode
    )

def create_efficient_mednext_small(num_input_channels, num_classes, n_channels=32, kernel_sizes=[1,3,5], strides=[1,1,1], uniform_dec_channels=None, ds=False, mode='train'):

    return EfficientMedNeXt(
        in_channels = num_input_channels, 
        n_channels = n_channels,
        n_classes = num_classes, 
        kernel_sizes=kernel_sizes,
        strides=strides,
        uniform_dec_channels=uniform_dec_channels,                     
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [3,4,8,8,8,8,8,4,3],
        checkpoint_style = 'outside_block',
        mode = mode
    )

def create_efficient_mednext_medium(num_input_channels, num_classes, n_channels=32, kernel_sizes=[1,3,5], strides=[1,1,1], uniform_dec_channels=None, ds=False, mode='train'):

    return EfficientMedNeXt(
        in_channels = num_input_channels, 
        n_channels = n_channels,
        n_classes = num_classes, 
        kernel_sizes=kernel_sizes,
        strides=strides,
        uniform_dec_channels=uniform_dec_channels,        
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [3,4,4,4,4,4,4,4,3],
        checkpoint_style = 'outside_block',
        mode = mode
    )

def create_efficient_mednext_large(num_input_channels, num_classes, n_channels=32, kernel_sizes=[1,3,5], strides=[1,1,1], uniform_dec_channels=None, ds=False, mode='train'):

    return EfficientMedNeXt_L(
        in_channels = num_input_channels, 
        n_channels = n_channels,
        n_classes = num_classes, 
        kernel_sizes=kernel_sizes,
        strides=strides,
        uniform_dec_channels=uniform_dec_channels,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [3,4,4,4,4,4,4,4,3],
        checkpoint_style = 'outside_block',
        mode = mode
    )

def create_efficient_mednext(num_input_channels, num_classes, model_id, n_channels=32, kernel_sizes=[1,3,5], strides=[1,1,1],
                      uniform_dec_channels=None, deep_supervision=False, mode='train'):

    model_dict = {
        'T': create_efficient_mednext_tiny,
        'S': create_efficient_mednext_small,
        'M': create_efficient_mednext_medium,
        'L': create_efficient_mednext_large,
        }
    
    return model_dict[model_id](
        num_input_channels, num_classes, n_channels, kernel_sizes, strides, uniform_dec_channels, deep_supervision, mode=mode
        )

if __name__ == "__main__":

    model = create_efficient_mednext_large(1, 3, 32, [1,3,5], False)
    print(model)