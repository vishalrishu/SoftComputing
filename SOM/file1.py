# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:17:07 2020

@author: VISHAL
"""

import pandas as pd

x = pd.ExcelFile("D:\cancer_data_notes.xlsx")
print(x.sheet_names)

x1 = x.parse(x)
print(x1[])