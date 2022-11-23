IMG_PATH = './wine_data/train_and_valid/X_train/'
IMG_HEIGHT = 512
IMG_WIDTH =  512

SEED  = 42

TRAIN_RATIO = 0.95

VAL_RATIO = 1 - TRAIN_RATIO
SHUFFLE_BUFFER_SIZE = 100

LEARNING_RATE =  1e-3

EPOCHS = 30

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
FULL_BATCH_SIZE = 32

# Train and test Time


DATA_PATH ='./wine_data/train_and_valid/X_train/' #'./quick_dataset/'
# DATA_VALID='./wine_data/train_and_valid/X_valid/'
# DATA_NEW ='./wine_data/train_and_valid/X/'

AUTOENCODER_MODEL_PATH = "./AutoEncoder_Model/baseline_autoencoder.pt"
ENCODER_MODEL_PATH = "./Encoder_Model/deep_encoder.pt"
DECODER_MODEL_PATH = "./Decoder_Model/deep_decoder.pt"
EMBEDDING_PATH = "./EMBEDDING_FINAL/data_embedding_f.npy"
EMBEDDING_SHAPE = (1,256,16,16)

NUM_IMAGES = 3
TEST_IMAGE_PATH = './wine_data/test_data/0.jpg'