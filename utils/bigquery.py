"""Utilities for loading data from BigQuery."""

from __future__ import annotations

import pandas as pd
from google.cloud import bigquery


def read_bigquery(query: str, project_id: str) -> pd.DataFrame:
    """Execute ``query`` against ``project_id`` and return a DataFrame.

    Parameters
    ----------
    query: str
        SQL query to execute in BigQuery.
    project_id: str
        Google Cloud project that hosts the BigQuery dataset.

    Returns
    -------
    pd.DataFrame
        Query results as a DataFrame. Uses the BigQuery Storage API when
        available which offers better performance on large result sets.
    """
    client = bigquery.Client(project=project_id)
    job = client.query(query)
    return job.result().to_dataframe(create_bqstorage_client=True)
