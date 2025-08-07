import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

#Data saving directories
CONFIGS_DIR = os.path.join(ROOT_DIR, 'configs')
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, 'checkpoints')
METRICS_DIR = os.path.join(ROOT_DIR, 'metrics')

#Data directories
IDENTIFIERS_DIR = os.path.join(ROOT_DIR, 'src', 'data', 'identifiers')
TRAIN_IDS_PATH = os.path.join(IDENTIFIERS_DIR, 'train_ids.npy')
VAL_IDS_PATH = os.path.join(IDENTIFIERS_DIR, 'val_ids.npy')
TEST_IDS_PATH = os.path.join(IDENTIFIERS_DIR, 'test_ids.npy')

