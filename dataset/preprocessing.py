# preprocessing.py
# Version: 1.0


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

    df = pd.read_csv(data_path, encoding='utf-8', low_memory=False)
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


def split_encode_and_scale(X, y):
    """
    Splits data into training and testing sets, encodes categorical features, and scales numeric features.

    Args:
        X (pd.DataFrame): The features to split, encode, and scale.
        y (pd.Series): The labels to split.

    Returns:
        X_train (pd.DataFrame): The training features after encoding and scaling.
        X_test (pd.DataFrame): The testing features after encoding and scaling.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The testing labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Identify categorical features
    categorical_features = [
        col for col in X_train.columns
        if X_train[col].dtype == object or X_train[col].nunique() <= 10
    ]

    # Encoding
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
    numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    scaler = MinMaxScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, y_train, y_test


def preprocess(data_path: str = 'dataset/edge-iiotset/eval/DNN-EdgeIIoT-dataset.csv', batch_size=64):
    """
    Preprocesses the dataset at the given path. 

    Args:
        data_path (str): The path to the dataset. Defaults to DNN eval file if none provided.
        batch_size (int, optional): The batch size to use when creating DataLoaders. Defaults to 64.

    Returns:
        tuple: A tuple containing the training and testing DataLoader, and a list of column names in the order they appear in the DataLoaders.
    """

    print("Begin preprocessing...")
    df = read_data(data_path)
    df = drop_irrelevant_columns(df)

    assert "Attack_label" in df.columns
    X = df.drop(columns=["Attack_label"])
    y = df["Attack_label"]

    X_train, X_test, y_train, y_test = split_encode_and_scale(X, y)

    # Tensorise
    train_loader = tensorise_and_wrap(X_train, y_train, batch_size)
    test_loader = tensorise_and_wrap(X_test, y_test, batch_size)

    return train_loader, test_loader, X_train.columns.tolist()


if __name__ == "__main__":
    train_loader, test_loader, column_names = preprocess()

    # testing
    for X, y in train_loader:
        print(X.shape)
        print(y.shape)
        break
    print(column_names)
