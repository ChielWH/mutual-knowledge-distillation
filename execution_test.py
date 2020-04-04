import torch
from trainer import Trainer
from config import get_config
from resnet import resnet32
from utils import prepare_dirs, save_config
from data_loader import get_test_loader, get_train_loader

# get the config
config, unparsed = get_config()

# ensure directories are setup
prepare_dirs(config)

# ensure reproducibility
# torch.manual_seed(config.random_seed)
kwargs = {}
if config.use_gpu:
    # torch.cuda.manual_seed_all(config.random_seed)
    kwargs = {'num_workers': config.num_workers,
              'pin_memory': config.pin_memory}
    #torch.backends.cudnn.deterministic = True

# instantiate data loaders
teachers = [resnet32() for _ in range(config.model_num)]
test_data_loader = get_test_loader(
    config.data_dir, config.batch_size, **kwargs
)

if config.is_train:
    train_data_loader = get_train_loader(data_dir=config.data_dir,
                                         batch_size=config.batch_size,
                                         random_seed=config.random_seed,
                                         shuffle=config.shuffle,
                                         teachers=teachers,
                                         cuda=config.use_gpu,
                                         **kwargs)

    data_loader = (train_data_loader, test_data_loader)
else:
    data_loader = test_data_loader

# instantiate trainer
trainer = Trainer(config, data_loader)

# either train
if config.is_train:
    save_config(config)
    trainer.train()

# or load a pretrained model and test
else:
    trainer.test()
