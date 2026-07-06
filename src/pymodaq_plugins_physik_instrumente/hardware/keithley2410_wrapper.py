# -*- coding: utf-8 -*-
"""
Wrapper Python pour le Keithley 2410 (sourcemètre).
Gère la communication bas niveau via pyvisa (GPIB).

Ce fichier est destiné à être placé dans le dossier hardware/ du plugin PyMoDAQ.
Il peut aussi être utilisé de manière autonome dans les scripts Langmuir.
"""

import numpy as np
import pyvisa


class Keithley2410:

    def __init__(self, adresse: str):
        """Connexion au Keithley 2410 via GPIB.

        Parameters
        ----------
        adresse : str
            Adresse VISA de l'instrument, ex: 'GPIB0::24::INSTR'
        """
        rm = pyvisa.ResourceManager()
        print("Appareils connectés :")
        [print('\t ->', element) for element in rm.list_resources()]
        self.instrument = rm.open_resource(adresse)
        self.instrument.timeout = 5000
        self.instrument.read_termination = '\n'

    def init_balayage(self, voltMin: float, voltMax: float, NV: int,
                      compliance: float = 700e-3, current_range: float = 20e-3):
        """Initialise le Keithley pour un balayage en tension avec mesure de courant.

        Parameters
        ----------
        voltMin : float - tension minimale du balayage [V]
        voltMax : float - tension maximale du balayage [V]
        NV : int - nombre de points de mesure
        compliance : float - limite de courant [A] (défaut 700 mA)
        current_range : float - plage de mesure du courant [A] (défaut 20 mA)

        Returns
        -------
        volts : array - liste des tensions de mesure
        """
        volts = np.linspace(voltMin, voltMax, NV)
        voltRang = abs(voltMin) + abs(voltMax)
        self.instrument.write('*rst')
        self.instrument.write(f':SENS:CURR:PROT {compliance}')
        self.instrument.write(':SOUR:FUNC VOLT')
        self.instrument.write(':SOUR:VOLT:MODE FIX')
        self.instrument.write(f':SOUR:VOLT:RANG {voltRang}')
        self.instrument.write(f':SOUR:VOLT:LEV {voltMin}')
        self.instrument.write(f':SENS:CURR:RANG {current_range}')
        return volts

    def set_voltage(self, volt: float):
        """Applique une tension sur la sortie du Keithley.

        Parameters
        ----------
        volt : float - tension à appliquer [V]
        """
        self.instrument.write(':OUTP ON')
        self.instrument.write(f':SOUR:VOLT:LEV {volt}')

    def get_voltage(self) -> float:
        """Lit la tension actuellement sourcée.

        Returns
        -------
        float - tension sourcée [V]
        """
        response = self.instrument.query(':SOUR:VOLT:LEV?')
        return float(response.strip())

    def get_current(self) -> float:
        """Lit le courant mesuré par le Keithley.

        Returns
        -------
        float - courant mesuré [A]
        """
        response = self.instrument.query('READ?')
        values = response.split(',')
        return float(values[1])

    def measure(self, volt: float) -> tuple:
        """Applique une tension et retourne (tension, courant) mesurés.

        Parameters
        ----------
        volt : float - tension à appliquer [V]

        Returns
        -------
        tuple : (tension [V], courant [A])
        """
        self.set_voltage(volt)
        response = self.instrument.query('READ?')
        values = response.split(',')
        return float(values[0]), float(values[1])

    def output_off(self):
        """Éteint la sortie du Keithley."""
        self.instrument.write(':OUTP OFF')

    def close(self):
        """Éteint la sortie et ferme la connexion."""
        self.output_off()
        self.instrument.close()