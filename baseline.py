import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

from utils.denmark_dataset import CLASSES
from utils.logger import init_logging, log_eval


def parse_args():
    parser = argparse.ArgumentParser('Baseline for training')
    parser.add_argument('--data-path', type=str, default='./data/train_test_whole_class_1km',
                        help='path to data [default: "./data/train_test_whole_class_1km"]')
    parser.add_argument('--model', type=str, default='RF', help='model name [default: RF]')
    parser.add_argument('--log-dir', type=str, default="./log", help='Log path [default: "./log"]')
    parser.add_argument('--no-console-logging', default=False, action='store_true',
                        help='deactivates most console output')
    parser.add_argument('--no-file-logging', default=False, action='store_true',
                        help='deactivates logging into a file')
    parser.add_argument('--max-training-samples', type=int, default=-1,
                        help='max number of training samples used (default: -1 [meaning all are uses])')

    return parser.parse_args()


def get_dataset(data_path, split):
    path = os.path.join(data_path, split)
    dataset = []

    # load all room data
    for room_name in os.listdir(path):
        room_path = os.path.join(path, room_name)

        print(room_path)

        # load data (pandas is way faster than numpy in this regard)
        room_data = pd.read_csv(room_path, sep=" ", header=None).values  # xyzrgbl, N*7

        # split into points and labels
        heights, labels = room_data[:, [2]], room_data[:, 6]  # xyzrgb, N*6; l, N

        dataset.append((heights, labels))

    X = np.concatenate([d[0] for d in dataset], 0)
    y = np.concatenate([d[1] for d in dataset], 0)

    return X, y


def main(args):
    args = parse_args()

    # init logger
    logger, experiment_dir, checkpoint_dir = init_logging(
        args.log_dir, args.model, not args.no_console_logging, not args.no_file_logging
    )

    logger.info("start loading train data ...")
    X_train, y_train = get_dataset(data_path=args.data_path, split="train")
    X_val, y_val = get_dataset(data_path=args.data_path, split="validate")
    X_train = np.concatenate([X_train, X_val], 0)
    y_train = np.concatenate([y_train, y_val], 0)

    # init model and optimizer
    model_name = args.model.lower().replace(" ", "_").replace("_", "")
    if model_name in ["rf", "randomforest"]:
        model = RandomForestClassifier(1000, n_jobs=-1)
        param_grid = {
            'max_features': [None, "auto"],
            'max_depth': [10, 50, None],
            'bootstrap': [True, False]
        }
    elif model_name in ["et", "extratrees"]:
        model = ExtraTreesClassifier(1000, n_jobs=-1)
        param_grid = {
            'max_features': [None, "auto"],
            'max_depth': [10, 50, None],
            'bootstrap': [True, False]
        }
    else:
        raise Exception(f"Unknown baseline model: {args.model}")

    rs = np.random.RandomState(42)
    # get a stratified subset for faster training
    if args.max_training_samples != -1:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=args.max_training_samples, random_state=rs,
            stratify=y_train
        )

    # do parameter selection on stratified cv sets
    logger.info("Start training")
    skf = StratifiedKFold(random_state=rs)
    clf = GridSearchCV(model, param_grid, scoring="accuracy", cv=skf)
    clf.fit(X_train, y_train)

    logger.info("Best parameters set found on development set:")
    logger.info(clf.best_params_)

    del X_train, y_train

    logger.info("start loading test data ...")
    X_test, y_test = get_dataset(data_path=args.data_path, split="test")

    logger.info("Predicting the test set")
    y_pred = clf.predict(X_test)

    n_classes = len(np.unique(y_test))
    total_correct = np.sum((y_test == y_pred))
    class_distribution, _ = np.histogram(y_test, range(n_classes + 1))

    total_seen_class = np.zeros(n_classes)
    total_correct_class = np.zeros(n_classes)
    total_iou_deno_class = np.zeros(n_classes)
    for j in range(n_classes):
        total_seen_class[j] += np.sum((y_test == j))
        total_correct_class[j] += np.sum((y_pred == j) & (y_test == j))
        total_iou_deno_class[j] += np.sum(((y_pred == j) | (y_test == j)))

    class_distribution = class_distribution.astype(np.float32) / np.sum(class_distribution.astype(np.float32))
    mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
    acc = total_correct / len(y_test)
    class_acc = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))

    log_eval(
        logger, np.nan, mIoU, acc, class_acc, total_correct_class, total_iou_deno_class, class_distribution, CLASSES
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
