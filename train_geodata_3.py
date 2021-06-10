#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on S3DIS dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os

# Dataset
from datasets.Geo_data_1 import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN
from utils.tester import ModelTester


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class S3DISConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'S3DIS'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = 'cloud_segmentation'

    # Number of CPU threads for the input pipeline
    input_threads = 8

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ###################
    # KPConv parameters
    ###################

    # # Radius of the input sphere
    # # in_radius = 1.5
    #
    # # Number of kernel points
    # num_kernel_points = 15
    #
    # # Size of the first subsampling grid in meter
    # first_subsampling_dl = 0.03
    #
    # # Radius of convolution in "number grid cell". (2.5 is the standard value)
    # conv_radius = 2.5
    #
    # # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    # deform_radius = 6.0
    #
    # # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    # KP_extent = 1.2

    # Radius of the input sphere
    # in_radius = 1.5
    in_radius = 20.0

    # Number of kernel points
    num_kernel_points = 30

    density_parameter = 1.0

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.1

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    # conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 0.006

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.0

######################### inner ##
    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128
    in_features_dim = 5

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 500

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 6

    # Number of steps per epochs
    epoch_steps = 50

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 0.8

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '3'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    # previous_training_path = 'Log_2020-03-19_19-53-27'
    previous_training_path = ''

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = S3DISConfig()
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]
    #data_set = "/data/REASEARCH/DEEPCROP/PointCloudData/clear_data/202102_train_test_val_set_whole_class_sub"
    #data_set = "/home/dmn774/Deepcrop/data/geodata/202102_train_test_val_set_whole_class"
    #data_set = "/home/dmn774/Deepcrop/data/geodata/20210430_train_test_val_set_whole_class"
    data_set = "/home/dmn774/Deepcrop/data/geodata/202105_train_test_val_set_whole_class"
    #data_set = "/data/REASEARCH/DEEPCROP/PointCloudData/clear_data/test_whole_1"

    # Initialize datasets
    training_dataset = S3DISDataset(config,data_set=data_set, set='training', use_potentials=True)
    validate_dataset = S3DISDataset(config,data_set=data_set, set='validating', use_potentials=True)

    # Initialize samplers
    training_sampler = S3DISSampler(training_dataset)
    validate_sampler = S3DISSampler(validate_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=4,
                                 sampler=training_sampler,
                                 collate_fn=S3DISCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True,
                                 drop_last=True)

    validate_loader = DataLoader(validate_dataset,
                             batch_size=1,
                             sampler=validate_sampler,
                             collate_fn=S3DISCollate,
                             num_workers=config.input_threads,
                             pin_memory=True,
                                 drop_last=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    validate_sampler.calibration(validate_loader, verbose=True)

    # Optional debug functions
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_upsampling(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)

    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')
    #trainer.train(net, training_loader, validate_loader, config)
    trainer.train(net, training_loader, validate_loader, config)


    # Testing
    test_dataset = S3DISDataset(config,data_set=data_set, set='testing', use_potentials=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=S3DISCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)
    test_sampler.calibration(test_loader, verbose=True)

    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    if config.dataset_task == 'classification':
        tester.classification_test(net, test_loader, config)
    elif config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config)
    elif config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)