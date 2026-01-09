# load necessary libraries
from pathlib import Path
import pandas as pd
import numpy as np

# set up directories 
ROOT = Path.cwd()
DATA_DIR = ROOT/ "data"
DATA_DIR.mkdir(exist_ok= True)
PROCESSED_DATA = DATA_DIR/ "cleaned.csv" # the csv file that contains processed csv

# load the data
dat_path = Path("./data/SBAnational.csv")
df = pd.read_csv(dat_path)
df_copy = df.copy()
df_copy = df.drop_duplicates()

# drop rows that contains any missing value
df_copy.dropna(subset= ["State", "BankState", "NewExist", "RevLineCr", "LowDoc", 
                        "DisbursementDate", "MIS_Status"],
               inplace= True)

# convert some columns to its correct data type
# they are object prior to the transformation 
df_copy[["DisbursementGross", "BalanceGross", "ChgOffPrinGr", "GrAppv", "SBA_Appv"]] = \
    df_copy[["DisbursementGross", "BalanceGross", "ChgOffPrinGr", "GrAppv", "SBA_Appv"]].map(lambda x:
        x.strip().replace("$", "").replace(",", "")).astype("float64")

# change the dtype of ApprovalFY to int
# it's a mix of str and int 
def str_cleaner(x):
    if isinstance(x, str):
        return x.replace("A", "") # there's one row with "A"
    return x
df_copy["ApprovalFY"] = df_copy["ApprovalFY"].apply(str_cleaner).astype("int64")

# change the dtype of other cols
# NewExist -> int; Zip, UrbanRural -> str(categorical)
df_copy = df_copy.astype({"Zip": "str", "NewExist": "int8", "UrbanRural": "str"})

# extract industry information and convert it to NACE 
# create a dict for the corresponding values 
naics_2_to_nace = {
    '11': 'A', # Agriculture, Forestry and Fishing
    '21': 'B', # Mining and Quarrying
    '22': 'D', # Electricity, Gas, Steam and Air Conditioning
    '23': 'F', # Construction
    '31': 'C', # Manufacturing
    '32': 'C', # Manufacturing
    '33': 'C', # Manufacturing
    '42': 'G', # Wholesale Trade
    '44': 'G', # Retail Trade
    '45': 'G', # Retail Trade
    '48': 'H', # Transportation and Storage
    '49': 'H', # Transportation and Storage
    '51': 'J', # Information and Communication
    '52': 'K', # Financial and Insurance Activities
    '53': 'L', # Real Estate Activities
    '54': 'M', # Professional, Scientific and Technical
    '55': 'M', # Management of Companies
    '56': 'N', # Administrative and Support Service
    '61': 'P', # Education
    '62': 'Q', # Human Health and Social Work
    '71': 'R', # Arts, Entertainment and Recreation
    '72': 'I', # Accommodation and Food Service
    '81': 'S', # Other Service Activities
    '92': 'O'  # Public Administration and Defence
}

df_copy["NAICS"] = df_copy["NAICS"].astype("str").apply(lambda x: x[:2])
df_copy["NACE"] = df_copy['NAICS'].map(naics_2_to_nace)

# remove rows where the industry is NaN after the transformation
df_copy.dropna(subset=['NACE'], inplace= True)

# turn the col IsFranchise into a binary variable (= 0 or 1)
df_copy.loc[(df_copy["FranchiseCode"] <= 1), "IsFranchise"] = 0
df_copy.loc[(df_copy["FranchiseCode"]) > 1, "IsFranchise"] = 1

# similarly, turn the col NewExist to a binary variable 
# in the original classification, 1 = existing, 2 = new business
df_copy = df_copy[(df_copy["NewExist"] == 1) | (df_copy["NewExist"] == 2)]

# turn 1 to 0 (existing business) and 2 to 1 (new business)
df_copy.loc[(df_copy["NewExist"] == 1), "NewBusiness"] = 0
df_copy.loc[(df_copy["NewExist"] == 2), "NewBusiness"] = 1

# clean the two cols, RevLineCr and LowDoc, and keep rows whose values = y or n
df_copy = df_copy[(df_copy["RevLineCr"] == "Y") | (df_copy["RevLineCr"] == "N")]
df_copy = df_copy[(df_copy["LowDoc"] == "Y") | (df_copy["LowDoc"] == "N")]

# dichotomization
df_copy["RevLineCr"] = np.where(df_copy["RevLineCr"] == "N", 0, 1)
df_copy["LowDoc"] = np.where(df_copy["LowDoc"] == "N", 0, 1)

# turn default status (=MIS_status), into binary variable
df_copy["Default"] = np.where(df_copy["MIS_Status"] == "P I F", 0, 1)

# convert date to datetime values
df_copy[["ApprovalDate", "DisbursementDate"]] = \
df_copy[["ApprovalDate", "DisbursementDate"]].apply(pd.to_datetime)

# calculate the days passed between approval date and disbursement
df_copy["DaysToDisbursement"] = df_copy["DisbursementDate"] - df_copy["ApprovalDate"]

# convert the dtype to int64
df_copy["DaysToDisbursement"] = df_copy["DaysToDisbursement"].dt.days
# remove negative values 
df_copy = df_copy[df_copy["DaysToDisbursement"] >= 0]

# create a column for the year of Disbursement
df_copy["DisbursementFY"] = df_copy["DisbursementDate"].map(lambda x: x.year)

# dummy variable for marking if the business and the bank are in the same state
df_copy["StateSame"] = np.where(df_copy["State"] == df_copy["BankState"], 1, 0)

# create a new col quantifying the risks taken by other organization 
df_copy["GuarantyRate"] = df_copy["SBA_Appv"] / df_copy["GrAppv"]

# dummy variable for marking rows where the loan approved by the bank equals to disbursement
df_copy["AppvDisbursed"] = np.where(df_copy["DisbursementGross"] == df_copy["GrAppv"], 1, 0)

# Format dtypes where necessary after feature engineering
df_copy = df_copy.astype({"IsFranchise": "int64", "NewBusiness": "int64"})

# drop columns that are not needed for analysis
df_copy.drop(columns=["City", "Zip", "Bank", "NAICS", "ApprovalDate", 
                      "NewExist", "FranchiseCode","ChgOffDate", "DisbursementDate", "BalanceGross", 
                      "ChgOffPrinGr", "SBA_Appv", "MIS_Status"], inplace=True)

# create a marker for cases that are backed up by real estate (term > 240, aka 2y)
# this is an estimate instead of fixed number 
df_copy["RealEstate"] = np.where(df_copy["Term"] >= 240, 1, 0)

# field for loans active during the Great Recession (2007-2009)
df_copy["GreatRecession"] = \
    np.where(((2007 <= df_copy["DisbursementFY"]) & (df_copy["DisbursementFY"] <= 2009)) | 
    ((df_copy["DisbursementFY"] < 2007) & (df_copy["DisbursementFY"] + (df_copy["Term"]/12) >= 2007)), 1, 0)

#--------------------------------#
# for simulating ESG risk scores
#--------------------------------#
# higher score represent higher risks 
# dictionary defining risk scores for each industry
# nace_esg_logic = {
#     "A": {"carbon": (70, 10), "social": (50, 15), "gov": (40, 10)}, 
#     "B": {"carbon": (90, 5),  "social": (60, 10), "gov": (70, 15)}, 
#     "C": {"carbon": (65, 15), "social": (45, 10), "gov": (40, 10)}, 
#     "D": {"carbon": (85, 10), "social": (40, 10), "gov": (50, 10)},  
#     "F": {"carbon": (60, 10), "social": (70, 15), "gov": (60, 10)},  
#     "G": {"carbon": (30, 10), "social": (40, 10), "gov": (30, 10)},  
#     "J": {"carbon": (15, 5),  "social": (20, 10), "gov": (20, 5)},  
#     "K": {"carbon": (10, 5),  "social": (30, 10), "gov": (20, 10)}, 
#     "M": {"carbon": (20, 10), "social": (25, 10), "gov": (20, 10)}, 
# }

# def generate_esg_metrics(nace_code):
#     # if not defined, generate from the alternative range of score 
#     logic = nace_esg_logic.get(nace_code, {"carbon": (40, 15), "social": (40, 15), "gov": (40, 15)})
    
#     carbon = np.random.normal(logic["carbon"][0], logic["carbon"][1])
#     social = np.random.normal(logic["social"][0], logic["social"][1])
#     gov = np.random.normal(logic["gov"][0], logic["gov"][1])
    
#     return pd.Series([np.clip(carbon, 1, 100), np.clip(social, 1, 100), np.clip(gov, 1, 100)])

# df_copy[["CarbonIntensity", "SocialScore", "GovRisk"]] = df_copy["NACE"].apply(generate_esg_metrics)

# # calculate the weighted score 
# df_copy["ESGRiskScore"] = round((df_copy["CarbonIntensity"] * 0.5 + 
#                              df_copy["SocialScore"] * 0.25 + 
#                              df_copy["GovRisk"] * 0.25), 0)

#--------------------------------#
# for simulating IRS
#--------------------------------#
# model interest rate spread
# def estimate_spread(row):
#     # basic irs: 4.3%, data from world bank (92 - 06)
#     base_spread = 4.3 
    
#     # the longer the term, the higher irs is
#     term_premium = 0.5 if row["Term"] > 120 else 0
    
#     # higher risk for the bank (low GuarantyRate)
#     guaranty_rate = row["GuarantyRate"]
#     risk_premium = (1 - guaranty_rate) * 2.0
    
#     # fluctuation in the market
#     noise = np.random.normal(0, 0.2)
    
#     return round(base_spread + term_premium + risk_premium + noise, 1)

# df_copy["InterestRateSpread"] = df_copy.apply(estimate_spread, axis = 1)

# store cleaned data as csv file
if not PROCESSED_DATA.exists():
    df_copy.to_csv(PROCESSED_DATA, index= False)
    print("File created!")
else:
    print("File already existed.")
    
