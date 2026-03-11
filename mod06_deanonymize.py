import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    keys = ["age", "zip3", "gender"]

    anon_counts = anon_df.groupby(keys).size().reset_index(name="anon_count")
    aux_counts = aux_df.groupby(keys).size().reset_index(name="aux_count")

    anon_with_counts = anon_df.merge(anon_counts, on=keys)
    aux_with_counts = aux_df.merge(aux_counts, on=keys)

    anon_unique = anon_with_counts[anon_with_counts["anon_count"] == 1]
    aux_unique = aux_with_counts[aux_with_counts["aux_count"] == 1]

    matches = anon_unique.merge(aux_unique, on=keys)

    return matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)
