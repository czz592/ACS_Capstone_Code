# preprocessing.py
# Version: 1.0


import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce


def read_data(data_path: str) -> pd.DataFrame:
    """
    Read data from a csv file and cleans the data by dropping duplicates and null values.

    Args:
        data_path (str): The path to the csv file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    cols_to_use = ['icmp.checksum', 'icmp.seq_le', 'tcp.ack', 'tcp.ack_raw',
                   'tcp.checksum', 'tcp.connection.fin', 'tcp.connection.rst',
                   'tcp.connection.syn', 'tcp.dstport', 'tcp.flags', 'tcp.flags.ack',
                   'tcp.len', 'tcp.seq', 'tcp.srcport', 'udp.stream', 'mqtt.hdrflags',
                   'mqtt.len', 'mqtt.msgtype', 'Attack_label']

    df = pd.read_csv(data_path, encoding='utf-8',
                     low_memory=False, usecols=cols_to_use)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns determined to be irrelevant and sparse.

    The columns to drop are based on the results of the data exploration notebook. The irrelevant columns are dropped first, then the sparse columns are dropped.

    Returns:
        pd.DataFrame: The cleaned data with the irrelevant and sparse columns dropped.
    """
    irrelevant_cols = [
        "frame.time",
        "ip.src_host",
        "ip.dst_host",
        "arp.src.proto_ipv4",
        "arp.dst.proto_ipv4",
        "http.file_data",
        "http.request.full_uri",
        "icmp.transmit_timestamp",
        "tcp.options",
        "tcp.payload",
        "mqtt.msg",
        "icmp.unused",
        "mqtt.msg_decoded_as",
        "Attack_type"
    ]
    df.drop(columns=irrelevant_cols, inplace=True)

    sparse_cols = [
        'arp.opcode',
        'arp.hw.size',
        'http.content_length',
        'http.request.uri.query',
        'http.request.method',
        'http.referer',
        'http.request.version',
        'http.response',
        'http.tls_port',
        'tcp.connection.synack',
        'udp.port',
        'udp.time_delta',
        'dns.qry.name',
        'dns.qry.name.len',
        'dns.qry.qu',
        'dns.qry.type',
        'dns.retransmission',
        'dns.retransmit_request',
        'dns.retransmit_request_in',
        'mqtt.conack.flags',
        'mqtt.conflag.cleansess',
        'mqtt.conflags',
        'mqtt.proto_len',
        'mqtt.protoname',
        'mqtt.topic',
        'mqtt.topic_len',
        'mqtt.ver',
        'mbtcp.len',
        'mbtcp.trans_id',
        'mbtcp.unit_id'
    ]
    df.drop(columns=sparse_cols, inplace=True)
    return df


def tensorise_and_wrap(X: pd.DataFrame, y: pd.Series, batch_size=64):
    """
    Converts a pandas DataFrame and Series into a PyTorch DataLoader.

    Args:
        X (pd.DataFrame): The features to tensorise.
        y (pd.Series): The labels to tensorise.
        batch_size (int, optional): The batch size of the DataLoader. Defaults to 64.

    Returns:
        torch.utils.data.DataLoader: A DataLoader containing the tensorised data.
    """
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def split_encode_and_scale(df):
    """
    Splits data into training and testing sets, encodes categorical features, and scales numeric features.

    Note: Centralised pipeline function, keeping for archival purposes.

    Args:
        X (pd.DataFrame): The features to split, encode, and scale.
        y (pd.Series): The labels to split.

    Returns:
        X_train (pd.DataFrame): The training features after encoding and scaling.
        X_test (pd.DataFrame): The testing features after encoding and scaling.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The testing labels.
    """
    assert "Attack_label" in df.columns
    print(f"{'-'*5}Splitting dataset into train and test...")
    X = df.drop(columns=["Attack_label"])
    y = df["Attack_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Identify categorical features
    categorical_features = [
        col for col in X_train.columns
        if X_train[col].dtype == object or X_train[col].nunique() <= 10
    ]

    # Encoding
    print(f"{'-'*5}Encoding features...")
    encoder_onehot = ce.OneHotEncoder()
    encoder_count = ce.CountEncoder()

    # One-hot for low cardinality
    X_train[categorical_features] = X_train[categorical_features].astype(str)
    X_test[categorical_features] = X_test[categorical_features].astype(str)

    low_card = [
        col for col in categorical_features if X_train[col].nunique() <= 10]
    X_train_oh = encoder_onehot.fit_transform(X_train[low_card])
    X_test_oh = encoder_onehot.transform(X_test[low_card])

    X_train = X_train.drop(columns=low_card).join(X_train_oh)
    X_test = X_test.drop(columns=low_card).join(X_test_oh)

    # Count encode 'tcp.srcport' if it exists
    if 'tcp.srcport' in X_train.columns:
        X_train['tcp.srcport'] = encoder_count.fit_transform(
            X_train['tcp.srcport'].astype(str))
        X_test['tcp.srcport'] = encoder_count.transform(
            X_test['tcp.srcport'].astype(str))

    # Scale
    print(f"{'-'*5}Scaling features...")
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    scaler = MinMaxScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, y_train, y_test


def fl_encode_and_scale(df) -> pd.DataFrame:
    """
    Splits data into features and labels, encodes categorical features and scales numeric features.

    *Note*: The function violates the DRY (Don't Repeat Yourself) principle as it is very similar to split_encode_and_scale, but it's for the federated pipeline, just different enough to necessitate a separate function.

    Args:
        df (pd.DataFrame): The DataFrame to encode and scale.

    Returns:
        pd.DataFrame: The encoded and scaled DataFrame.
    """
    assert "Attack_label" in df.columns

    X = df.drop(columns=["Attack_label"])
    y = df["Attack_label"]

    # Identify categorical features
    categorical_features = [
        col for col in X.columns
        if X[col].dtype == object or X[col].nunique() <= 10
    ]

    # Encoding
    encoder_onehot = ce.OneHotEncoder()
    encoder_count = ce.CountEncoder()

    # One-hot for low cardinality
    X[categorical_features] = X[categorical_features].astype(str)

    low_card = [
        col for col in categorical_features if X[col].nunique() <= 10]
    X_oh = encoder_onehot.fit_transform(X[low_card])

    X = X.drop(columns=low_card).join(X_oh)

    # Count encode 'tcp.srcport' if it exists
    if 'tcp.srcport' in X.columns:
        X['tcp.srcport'] = encoder_count.fit_transform(
            X['tcp.srcport'].astype(str))

    # Scale
    numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
    scaler = MinMaxScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X["Attack_label"] = y

    return X


def partition_and_save(
        df: pd.DataFrame,
        iid: bool = True,
        seed: int = 42,
        base_dir: str = 'dataset/edge-iiotset/partitions_'
):
    base_output_dir = 'iid' if iid else 'non-iid'
    output_dir = base_dir + base_output_dir

    for num_clients in [3, 5, 10]:
        partition_dir = os.path.join(output_dir, f'{num_clients}_clients')

        print(f"Saving {num_clients} partitions to {partition_dir}...")

        os.makedirs(partition_dir, exist_ok=True)
        df_copy = df.copy()

        if iid:
            df_copy = df.sample(
                frac=1, random_state=seed).reset_index(drop=True)

        partitions = np.array_split(df_copy, num_clients)

        for i, partition in enumerate(partitions):
            partition_path = os.path.join(partition_dir, f'client_{i+1}.csv')
            partition.to_csv(partition_path, index=False)
            print(f"{'-'*5}Saved partition {i+1} to {partition_path}")

    print("Partitioning complete.")
    return


def read_partition_data(cid: int, iid: bool = True):
    data_path = f'dataset/edge-iiotset/partitions_{"iid" if iid else "non-iid"}/client_{cid}.csv'
    return read_data(data_path)


def preprocess_centralised(data_path: str = 'dataset/edge-iiotset/eval/DNN-EdgeIIoT-dataset.csv', batch_size=64):
    """
    Preprocesses the dataset at the given path. 

    Args:
        data_path (str): The path to the dataset. Defaults to DNN eval file if none provided.
        batch_size (int, optional): The batch size to use when creating DataLoaders. Defaults to 64.

    Returns:
        tuple: A tuple containing the training and testing DataLoader, and a list of column names in the order they appear in the DataLoaders.
    """

    print("Begin centralised preprocessing pipeline...")
    df = read_data(data_path)
    # df = drop_irrelevant_columns(df)

    X_train, X_test, y_train, y_test = split_encode_and_scale(df)

    # Tensorise
    train_loader = tensorise_and_wrap(X_train, y_train, batch_size)
    test_loader = tensorise_and_wrap(X_test, y_test, batch_size)

    return train_loader, test_loader, X_train.columns.tolist()


def preprocess_federated(
        data_path: str = 'dataset/edge-iiotset/eval/DNN-EdgeIIoT-dataset.csv',
        iid: bool = True,
        seed: int = 42
):
    print("Begin federated preprocessing pipeline...")
    df = read_data(data_path)

    df = fl_encode_and_scale(df)

    partition_and_save(df, iid, seed)


if __name__ == "__main__":
    # train_loader, test_loader, column_names = preprocess_centralised()
    # print(column_names)

    # torch.save(train_loader, "train_loader.pth")
    # torch.save(test_loader, "test_loader.pth")

    # load from saved files
    # train_loader = torch.load("train_loader.pth", weights_only=False)
    # test_loader = torch.load("test_loader.pth", weights_only=False)
    # print(train_loader.dataset[0][0].shape)
    # print(train_loader.dataset[0][1].shape)

    # print(test_loader.dataset[0][0].shape)
    # print(test_loader.dataset[0][1].shape)

    preprocess_federated(iid=True, seed=42)
    # preprocess_federated(iid=False, seed=42)
