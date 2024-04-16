import os
import torch
import random
import numpy as np
import inspect

from datetime import datetime

def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dino_head_weights(pretrain_path):

    """
    :param pretrain_path: Path to full DINO pretrained checkpoint as in https://github.com/facebookresearch/dino
     'full_ckpt'
    :return: weights only for the projection head
    """

    all_weights = torch.load(pretrain_path)

    head_state_dict = {}
    for k, v in all_weights['teacher'].items():
        if 'head' in k and 'last_layer' not in k:
            head_state_dict[k] = v

    head_state_dict = strip_state_dict(head_state_dict, strip_key='head.')

    # Deal with weight norm
    weight_norm_state_dict = {}
    for k, v in all_weights['teacher'].items():
        if 'last_layer' in k:
            weight_norm_state_dict[k.split('.')[2]] = v

    linear_shape = weight_norm_state_dict['weight'].shape
    dummy_linear = torch.nn.Linear(in_features=linear_shape[1], out_features=linear_shape[0], bias=False)
    dummy_linear.load_state_dict(weight_norm_state_dict)
    dummy_linear = torch.nn.utils.weight_norm(dummy_linear)

    for k, v in dummy_linear.state_dict().items():

        head_state_dict['last_layer.' + k] = v

    return head_state_dict


def init_experiment(args, runner_name=None, exp_id=None):

    args.cuda = torch.cuda.is_available()

    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:

        # Unique identifier for experiment
        now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
              datetime.now().strftime("%S.%f")[:-3] + ')'

        log_dir = os.path.join(root_dir, 'log', now)
        while os.path.exists(log_dir):
            now = '({:02d}.{:02d}.{}_|_'.format(datetime.now().day, datetime.now().month, datetime.now().year) + \
                  datetime.now().strftime("%S.%f")[:-3] + ')'

            log_dir = os.path.join(root_dir, 'log', now)

    else:

        log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')

    print(f'Experiment saved to: {args.log_dir}')

    print(runner_name)
    print(args)

    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

