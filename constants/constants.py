from constants.secrets import USERPATH

# Repo path
path_repo = f'{USERPATH}/kaggle-predicting-arguments'

# Path for training data
path_train = f'{path_repo}/data/feedback-prize-effectiveness'

# Training data
train_data = 'train.csv'

# Full path - training data
train_data_fpath = f"{path_train}/{train_data}"

if __name__ == "__main__":

    print(f"Training path (train_data_fpath) = {train_data_fpath}")