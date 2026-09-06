"""Configuration for Freddie Mac annual sample processing."""

ORIGINATION_COLUMNS = [
    "CLASSIC FICO",
    "FIRST PAYMENT DATE",
    "FIRST TIME HOMEBUYER INDICATOR",
    "MATURITY DATE",
    "METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION",
    "MORTGAGE INSURANCE PERCENTAGE (MI %)",
    "NUMBER OF UNITS",
    "OCCUPANCY STATUS",
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)",
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO",
    "ORIGINAL UPB",
    "ORIGINAL LOAN-TO-VALUE (LTV)",
    "ORIGINAL INTEREST RATE",
    "CHANNEL",
    "PREPAYMENT PENALTY INDICATOR",
    "AMORTIZATION TYPE",
    "PROPERTY STATE",
    "PROPERTY TYPE",
    "POSTAL CODE",
    "LOAN IDENTIFIER",
    "LOAN PURPOSE",
    "ORIGINAL LOAN TERM",
    "NUMBER OF BORROWERS",
    "SELLER NAME",
    "SUPER CONFORMING FLAG",
    "PRE-HARP LOAN SEQUENCE NUMBER",
    "SPECIAL ELIGIBILITY PROGRAM",
    "HARP INDICATOR",
    "PROPERTY VALUATION METHOD",
    "INTEREST ONLY (I/O) INDICATOR",
    "VANTAGESCORE 4.0",
]

# Maximum raw field lengths from the July 2026 Freddie Mac file layout.
ORIGINATION_MAX_LENGTHS = {
    "CLASSIC FICO": 4,
    "FIRST PAYMENT DATE": 6,
    "FIRST TIME HOMEBUYER INDICATOR": 1,
    "MATURITY DATE": 6,
    "METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION": 5,
    "MORTGAGE INSURANCE PERCENTAGE (MI %)": 3,
    "NUMBER OF UNITS": 2,
    "OCCUPANCY STATUS": 1,
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)": 3,
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": 3,
    "ORIGINAL UPB": 12,
    "ORIGINAL LOAN-TO-VALUE (LTV)": 3,
    "ORIGINAL INTEREST RATE": 6,
    "CHANNEL": 1,
    "PREPAYMENT PENALTY INDICATOR": 1,
    "AMORTIZATION TYPE": 5,
    "PROPERTY STATE": 2,
    "PROPERTY TYPE": 2,
    "POSTAL CODE": 3,
    "LOAN IDENTIFIER": 12,
    "LOAN PURPOSE": 1,
    "ORIGINAL LOAN TERM": 3,
    "NUMBER OF BORROWERS": 2,
    "SELLER NAME": 60,
    "SUPER CONFORMING FLAG": 1,
    "PRE-HARP LOAN SEQUENCE NUMBER": 12,
    "SPECIAL ELIGIBILITY PROGRAM": 1,
    "HARP INDICATOR": 1,
    "PROPERTY VALUATION METHOD": 1,
    "INTEREST ONLY (I/O) INDICATOR": 1,
    "VANTAGESCORE 4.0": 4,
}

ORIGINATION_STRING_COLUMNS = {
    "FIRST TIME HOMEBUYER INDICATOR": "string",
    "OCCUPANCY STATUS": "string",
    "CHANNEL": "string",
    "PREPAYMENT PENALTY INDICATOR": "string",
    "AMORTIZATION TYPE": "string",
    "PROPERTY STATE": "string",
    "PROPERTY TYPE": "string",
    "POSTAL CODE": "string",
    "LOAN IDENTIFIER": "string",
    "LOAN PURPOSE": "string",
    "SELLER NAME": "string",
    "SUPER CONFORMING FLAG": "string",
    "PRE-HARP LOAN SEQUENCE NUMBER": "string",
    "SPECIAL ELIGIBILITY PROGRAM": "string",
    "HARP INDICATOR": "string",
    "INTEREST ONLY (I/O) INDICATOR": "string",
}


NUMERIC_SENTINELS = {
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": 999,
    "ORIGINAL LOAN-TO-VALUE (LTV)": 999,
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)": 999,
    "PROPERTY VALUATION METHOD": 7,
}


CATEGORICAL_SENTINELS = {
    "FIRST TIME HOMEBUYER INDICATOR": "9",
}


POSTAL_CODE_WIDTH = 3

# Documented code sets from the Freddie Mac General User Guide.
# Missing/null values are handled separately and are not included here.
DOCUMENTED_CODE_VALUES = {
    "FIRST TIME HOMEBUYER INDICATOR": {
        "Y",
        "N",
        "9",
    },
    "NUMBER OF UNITS": {
        "1",
        "2",
        "3",
        "4",
        "99",
    },
    "OCCUPANCY STATUS": {
        "P",
        "I",
        "S",
        "9",
    },
    "CHANNEL": {
        "R",
        "B",
        "C",
        "T",
        "9",
    },
    "PREPAYMENT PENALTY INDICATOR": {
        "Y",
        "N",
    },
    "AMORTIZATION TYPE": {
        "FRM",
        "ARM",
    },
    "PROPERTY TYPE": {
        "CP",
        "CO",
        "PU",
        "SF",
        "MH",
        "99",
    },
    "LOAN PURPOSE": {
        "P",
        "C",
        "N",
        "R",
        "9",
    },
    "SUPER CONFORMING FLAG": {
        "Y",
        "N",
    },
    "SPECIAL ELIGIBILITY PROGRAM": {
        "H",
        "F",
        "R",
    },
    "HARP INDICATOR": {
        "Y",
        "N",
    },
    "PROPERTY VALUATION METHOD": {
        "1",
        "2",
        "3",
        "4",
        "7",
    },
    "INTEREST ONLY (I/O) INDICATOR": {
        "Y",
        "N",
    },
}


# Documented unavailable or unknown codes.
# These are monitored even when they do not occur in a particular year.
DOCUMENTED_SENTINELS = {
    "CLASSIC FICO": 9999,
    "FIRST TIME HOMEBUYER INDICATOR": "9",
    "MORTGAGE INSURANCE PERCENTAGE (MI %)": 999,
    "NUMBER OF UNITS": 99,
    "OCCUPANCY STATUS": "9",
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)": 999,
    "ORIGINAL DEBT-TO-INCOME (DTI) RATIO": 999,
    "ORIGINAL LOAN-TO-VALUE (LTV)": 999,
    "CHANNEL": "9",
    "PROPERTY TYPE": "99",
    "POSTAL CODE": "000",
    "LOAN PURPOSE": "9",
    "NUMBER OF BORROWERS": 99,
    "PROPERTY VALUATION METHOD": 7,
    "VANTAGESCORE 4.0": 9999,
}