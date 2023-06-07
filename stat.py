import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#uzyskanie danych
Cancer = pd.read_csv('stat_haberman.csv' ,names=['Wiek','Rok','Wezly','Przezycie'],delimiter=';')       
Cancer.head(12)
# Konwersja formatu roku z „RR” na „RRRR”
Cancer['Rok'] = Cancer['Rok'] + 1900 
# Uzyskiwanie opisu statystycznego O danych
print(' Kształt: {} \n Kolumna:\n{} \n Opis:\n{} \n Czy Dane zawierają wartości null \n  {} '\
      .format(Cancer.shape, Cancer.columns, Cancer.describe(), Cancer.isnull().all()))
# Podział ramki danych na Przezyl i Nierzezyl
df_Przezyl = Cancer[Cancer['Przezycie'] ==1 ]
df_unPrzezyl = Cancer[Cancer['Przezycie'] ==2 ]
print('-- Przezyl -- \n{} \n -- Nierzezyl -- \n{}'.format(str(df_Przezyl.shape[0]),str(df_unPrzezyl.shape[0])))

def CalPDFCDF(FeatureVariable, ClassVariable):
    sns.FacetGrid(data=Cancer, hue=ClassVariable, height=5) \
    .map(sns.distplot, FeatureVariable) \
    .add_legend()
CalPDFCDF('Rok','Przezycie')
CalPDFCDF('Wiek','Przezycie')
CalPDFCDF('Wezly','Przezycie')

plt.figure(1)
plt.figure(figsize=(10,5))

# Wykres histogramu

plt.title('Histogram częstości węzłów dla dwóch grup pacjentów')
plt.hist(df_Przezyl.Wezly,label = 'Przezył wiecej niz 5 lat')
plt.hist(df_unPrzezyl.Wezly,label = 'Przezył mniej niz 5 lat')
plt.xlabel('Wezly')
plt.ylabel('Wezly częstotliwość')
plt.legend()
plt.grid()

"""
plt.title('Histogram częstości roku operacji dla dwóch grup pacjentów')
plt.hist(df_Przezyl.Rok,label = 'Przezyl')
plt.hist(df_unPrzezyl.Rok,label = 'Nie przezyl')
plt.xlabel('Rok operacji')
plt.ylabel('Częstotliwość operacji')
plt.legend()
plt.grid()
"""

df_Przezyl[df_Przezyl['Rok'].between(1959,1964)].shape[0]
df_unPrzezyl[df_unPrzezyl['Rok'].between(1959,1964)].shape[0]



