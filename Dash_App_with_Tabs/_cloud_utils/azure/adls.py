import gc
import os
from datetime import datetime

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.filedatalake import FileSystemClient

from _cloud_utils.azure._tigerml_utilities import reduce_mem_usage, timer


class ADLSUtil:
    def __init__(self, azure_details_dict: dict) -> None:
        """
        Parameters
        ----------
        azure_details_dict : dict
            dictionary of the credentials for azure cloud platforms
        """
        credential = DefaultAzureCredential()
        # get necessary data from config to access adls storage path
        conn_str_name = azure_details_dict["conn_str_name"]
        self.vault_url = azure_details_dict["key_vault_uri"]
        self.container = azure_details_dict["adls_data_container"]
        self.adls_host = azure_details_dict["adls_host"]
        self.protocol = azure_details_dict["protocol"]
        self.directory_ = azure_details_dict["dir_location"]

        secret_client = SecretClient(vault_url=self.vault_url, credential=credential)
        self.__conn_string = secret_client.get_secret(conn_str_name)
        self.adls_client = FileSystemClient.from_connection_string(
            conn_str=self.conn_string.value,
            file_system_name=self.container,
            credential=credential,
        )

        super(ADLSUtil, self).__init__()

    @property
    def conn_string(self):
        # access a resource in this case a secret from azure keyvault
        return self.__conn_string

    @property
    def container_path(self) -> str:
        """Generate the container path using the credential provided for the ADSL

        Returns
        -------
        str
            container path from where files will be read and written.

        Raises
        ------
        ValueError
            Raised when credential for the ADLS is either None or empty string.
        """

        # # check protocol , container, adls_host in path
        if not (self.protocol and self.container and self.adls_host):
            raise ValueError(
                "Not able to create container path. "
                + "Attribute protocol, container, adls_host value cannot be '' or None."
            )
        return f"{self.protocol}://{self.container}@{self.adls_host}"

    def read_azure_file(self, file: str, dir_path: str) -> pd.DataFrame:
        """process the files in an azure storage

        Parameters
        ----------
        file : str
            file name which need to read from adls
        dir_path : str
            directory path where the file which need to read is present.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the file data.
        """

        path = os.path.join(dir_path, file)
        if "\\" in path:
            path = path.replace("\\", "/")
        _, ext = os.path.splitext(path)
        read_func = {
            ".csv": pd.read_csv,
            ".parquet": pd.read_parquet,
        }
        df = read_func.get(ext)(
            path,
            storage_options={"connection_string": self.conn_string.value},
        )

        gc.collect()
        return df

    def write_azure_file(self, df: pd.DataFrame, dir_path: str, ext: str) -> None:
        """write the file in desired extension

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame which need to be stored.
        dir_path : str
            directory path where file will be stored with file name
        ext : str
            extension in which file should be stored.
        """

        write_func = {
            ".csv": df.to_csv,
            ".parquet": df.to_parquet,
        }
        write_func.get(ext)(
            dir_path + ext,
            index=False,
            storage_options={"connection_string": self.conn_string.value},
        )

    @timer
    def read_process_adls_blobs(self) -> pd.DataFrame:
        """Read blobs from a directory in an ADLS Storage Gen2 container,
            reduce memory consumption and retunr a dataframe

        Returns
        -------
        pd.DataFrame
            DataFrame with data of all the files concatenated.
        """
        file_paths = self.adls_client.get_paths(path=self.directory_)
        df = pd.concat(
            [
                self.read_azure_file(file=file.name, dir_path=self.container_path)
                for file in file_paths
                if os.path.splitext(file.name)[1]
                and not (
                    "processed_data_eda" in file.name
                )  # don't read from processed_data_eda
            ],
            ignore_index=True,
            axis=0,
        )
        df = reduce_mem_usage(df)
        gc.collect()
        return df

    @timer
    def write_to_adls(self, df: pd.DataFrame, exe: str) -> None:
        """Write a dataframe to a particular directory in ADLS Gen 2

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame which contained the processed data and need to be stored.
        exe : str
            extension in which file should be stored.
        """
        processed_dir = "processed_data_"

        dir_path = (
            self.container_path
            + f"/{self.directory_}/{processed_dir}/processed_data_dash_testing ("
            + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
            + ")"
        )
        self.write_azure_file(df=df, dir_path=dir_path, ext=exe)
