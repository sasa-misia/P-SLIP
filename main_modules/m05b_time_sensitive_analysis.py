# %% === Import necessary modules
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing necessary modules from config
from config import (
    AnalysisEnvironment
)

# # Importing necessary modules from psliptools
# from psliptools.rasters import (
# )

# from psliptools.utilities import (
# )

# from psliptools.geometries import (
# )

# from psliptools.scattered import (
# )

# Importing necessary modules from main_modules
from main_modules.m00a_env_init import get_or_create_analysis_environment, setup_logger
logger = setup_logger()
logger.info("=== Analyzing time-sensitive data patterns ===")

# %% === Methods to analyze time-sensitive data
def analyze_rainfall_noise(
        env: AnalysisEnvironment
    ) -> None: # TODO: Check and refine this implementation (from MATLAB code)
    # Iterate over each station file
    for file in env.station_files:
        # Read the CSV file
        data = pd.read_csv(file, parse_dates=['DataOra'])

        # Preprocess data
        data = data[data['Valore'] != -999]
        data['Giorno'] = data['DataOra'].dt.date

        # Daily rainfall
        Tgiorno = data.groupby('Giorno')['Valore'].sum().reset_index()
        Tgiorno.columns = ['Giorno', 'PioggiaGiornaliera']

        # Method 1: Daily rainfall noise
        piogge_positive = Tgiorno[Tgiorno['PioggiaGiornaliera'] > 0]['PioggiaGiornaliera']
        soglia75 = np.percentile(piogge_positive, 75)
        rumore = piogge_positive[piogge_positive < soglia75]
        media_rumore = rumore.mean()

        # Plot Method 1
        idx_alto = Tgiorno['PioggiaGiornaliera'] > soglia75
        idx_basso = (Tgiorno['PioggiaGiornaliera'] > 0) & (Tgiorno['PioggiaGiornaliera'] <= soglia75)
        date = Tgiorno['Giorno']
        pioggia = Tgiorno['PioggiaGiornaliera']
        date_rumore = date[idx_basso]
        rumore_val = pioggia[idx_basso]
        date_picchi = date[idx_alto]
        picchi_val = pioggia[idx_alto]

        plt.figure()
        plt.bar(date_rumore, rumore_val, color='k')
        plt.bar(date_picchi, picchi_val, color=[1, 0.7, 0.4])
        plt.axhline(soglia75, color='r', linestyle='--', label='Soglia 75° percentile')
        plt.axhline(media_rumore, color='b', linestyle='-.', label='Media rumore')
        plt.xlabel('Data')
        plt.ylabel('Pioggia giornaliera (mm)')
        plt.title('Metodo 1 - Piogge giornaliere')
        plt.legend()
        plt.savefig(os.path.join(env.output_dir, f'GraficoMetodo1_{os.path.basename(file)}.png'))

        # Method 2: Monthly rainfall noise
        Tgiorno['MeseSolare'] = pd.to_datetime(Tgiorno['Giorno']).dt.to_period('M').dt.to_timestamp()
        Tpioggia = Tgiorno[Tgiorno['PioggiaGiornaliera'] > 0]
        Tmese = Tpioggia.groupby('MeseSolare')['PioggiaGiornaliera'].sum().reset_index()
        Tmese.columns = ['MeseSolare', 'PioggiaMensile']
        Tmese['NumGiorniPioggia'] = Tpioggia.groupby('MeseSolare').size().values
        Tmese['RumoreGiornaliero'] = Tmese['PioggiaMensile'] / Tmese['NumGiorniPioggia']

        perc75_rumore_mensile = np.percentile(Tmese['RumoreGiornaliero'], 75)
        perc95_rumore_mensile = np.percentile(Tmese['RumoreGiornaliero'], 95)
        rumore_mensile = Tmese['RumoreGiornaliero'][Tmese['RumoreGiornaliero'] < perc95_rumore_mensile]
        media_rumore_mensile = rumore_mensile.mean()

        # Plot Method 2
        plt.figure()
        plt.bar(Tmese['MeseSolare'], Tmese['RumoreGiornaliero'], color=[0.2, 0.6, 0.8])
        plt.axhline(media_rumore_mensile, color='k', linestyle='--', label='Media rumore giornaliero')
        plt.axhline(perc75_rumore_mensile, color='r', linestyle='--', label='75° percentile')
        plt.axhline(perc95_rumore_mensile, color='m', linestyle='--', label='95° percentile')
        plt.xlabel('Mese')
        plt.ylabel('Rumore giornaliero (mm/giorno piovoso)')
        plt.title(f'Rumore giornaliero mensile (Metodo 2), {Tmese["MeseSolare"].min()} - {Tmese["MeseSolare"].max()}')
        plt.legend()
        plt.savefig(os.path.join(env.output_dir, f'GraficoMetodo2_{os.path.basename(file)}.png'))

        # Method 3: 30-day moving average rainfall noise
        finestra = 30
        date = Tgiorno['Giorno']
        pioggia = Tgiorno['PioggiaGiornaliera']
        rumore30 = []
        numerogiornopiovosi = []
        date_inizio = []

        for i in range(len(date) - finestra + 1):
            idx = slice(i, i + finestra)
            sub = pioggia.iloc[idx]
            n_pos = (sub > 0).sum()
            if n_pos > 0:
                rumore30.append(sub.sum() / n_pos)
                numerogiornopiovosi.append(n_pos)
                date_inizio.append(date.iloc[i])

        perc95_rumoreMetodo3 = np.percentile(rumore30, 95)
        media_rumore_cumulateMobili = np.mean([x for x in rumore30 if x < perc95_rumoreMetodo3])

        # Plot Method 3
        plt.figure()
        plt.bar(date_inizio, rumore30, color=[0.2, 0.6, 0.8])
        plt.xlabel('Data inizio finestra mobile')
        plt.ylabel('Cumulata 30 giorni (mm)')
        plt.title('Cumulate mobili di 30 giorni')
        plt.savefig(os.path.join(env.output_dir, f'CumulateMobili_{os.path.basename(file)}.png'))

# %% === Main function
def main(
        base_dir: str=None,
        gui_mode: bool=False
    ) -> None:
    """Main function to obtain statistical information of time-sensitive data."""
    # Get the analysis environment
    env = get_or_create_analysis_environment(base_dir=base_dir, gui_mode=gui_mode, allow_creation=False)

    # Analyze rainfall noise
    analyze_rainfall_noise(env)

# %% === Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and obtain statistics of time-sensitive data')
    parser.add_argument('--base_dir', type=str, help='Base directory for analysis')
    parser.add_argument('--gui_mode', action='store_true', help='Run in GUI mode')
    args = parser.parse_args()

    main(
        base_dir=args.base_dir,
        gui_mode=args.gui_mode
    )