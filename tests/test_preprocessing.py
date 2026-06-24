import pandas as pd

from src.preprocessing import clean_world_bank_wide_to_long, harmonise_country_names


def test_world_bank_wide_to_long_reshapes_indicator_years():
    raw = pd.DataFrame(
        {
            "Country Name": ["Aland", "Aland", "Borland"],
            "Indicator Code": ["NY.GDP.PCAP.KD", "SP.POP.TOTL", "NY.GDP.PCAP.KD"],
            "2000": [1000, 10, 2000],
            "2001": [1100, 11, 2100],
        }
    )
    out = clean_world_bank_wide_to_long(raw)
    assert {"country", "year", "NY.GDP.PCAP.KD", "SP.POP.TOTL"}.issubset(out.columns)
    assert out.loc[(out["country"] == "Aland") & (out["year"] == 2001), "NY.GDP.PCAP.KD"].item() == 1100


def test_harmonise_country_names_applies_aliases():
    out = harmonise_country_names(["Russian Federation", "Viet Nam"])
    assert out.tolist() == ["Russia", "Vietnam"]
