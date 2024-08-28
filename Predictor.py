import pickle
import joblib
import numpy as np
from os import system, name

def clear():
    if name == 'nt': 
        _ = system('cls') 
    else: 
        _ = system('clear')

def gather_input():
    gre_score = input("Enter the GRE Score ==> ")
    toefl_score = input("Enter TOEFL Score ==> ")
    univ_rating = input("Enter University Rating (1-5) ==> ")
    sop = input("Enter SOP Strength (1.00 - 5.00) ==> ")
    lor = input("Enter LOR Strength (1.00 - 5.00) ==> ")
    cgpa = input("Enter CGPA(1.00 - 10.00) ==> ")
    research = input("Enter Research (0 or 1) ==> ")
    x = [float(gre_score),float(toefl_score),float(univ_rating),float(sop),float(lor),float(cgpa),float(research)]
    x = np.array([x])
    return x

clear()
print("*"*100)
print("Graduate Admission Predictor")
print("*"*100)
print("[+] Loading Model")
loaded_model = joblib.load('./Model/fin_model.pkl')
print("[+] Model Loaded\n")
x = gather_input()
op = loaded_model.predict(x)
print("\n[+] The chances for graduate admissions are ==> "+str(round(op[0],2)))
