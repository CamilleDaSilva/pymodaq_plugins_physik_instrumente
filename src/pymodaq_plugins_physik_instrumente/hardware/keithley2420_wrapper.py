# -*- coding: utf-8 -*-
"""
Python wrapper for the Keithley 2420 (source-meter used as an ammeter for the
RPA collector current measurement).

Author: Laurianne
ONERA DPHY/CSE — PICOMAX-E, 2026
"""

import pyvisa


class Keithley2420:

    def __init__(self, adresse: str):
        rm = pyvisa.ResourceManager()

        print("Connected devices:")
        for element in rm.list_resources():
            print("\t ->", element)

        self.instrument = rm.open_resource(adresse)
        self.instrument.timeout = 5000
        self.instrument.read_termination = '\n'
        self.instrument.write_termination = '\n'

    def init_mesure(
        self,
        source_voltage: float = 0.0,
        compliance: float = 100e-3,
        current_range: float | None = None,
    ):
        """
        Initialise le Keithley en mesure de courant.

        Parameters
        ----------
        source_voltage : float
            Tension appliquée au collecteur.

        compliance : float
            Limite de courant.

        current_range : float or None
            None -> autorange
            float -> plage fixe en ampères.
        """

        inst = self.instrument

        inst.write("*RST")
        inst.write("*CLS")

        inst.write(":SYST:BEEP:STAT OFF")

        inst.write(":SOUR:FUNC VOLT")
        inst.write(":SOUR:VOLT:MODE FIX")
        inst.write(f":SOUR:VOLT:LEV {source_voltage}")

        inst.write(':SENS:FUNC "CURR"')

        # compliance
        inst.write(f":SENS:CURR:PROT {compliance}")

        # intégration (réduit le bruit)
        inst.write(":SENS:CURR:NPLC 5")

        if current_range is None:
            inst.write(":SENS:CURR:RANG:AUTO ON")
        else:
            inst.write(":SENS:CURR:RANG:AUTO OFF")
            inst.write(f":SENS:CURR:RANG {current_range}")

        inst.write(":FORM:ELEM CURR")

        inst.write(":OUTP ON")

    def set_source_voltage(self, volt: float):
        self.instrument.write(f":SOUR:VOLT:LEV {volt}")


    def read_current(self) -> float:
        return float(self.instrument.query(":READ?"))
    

    def measure(self) -> float:
        return float(self.instrument.query(":READ?"))

    def output_off(self):
        self.instrument.write(":OUTP OFF")

    def close(self):
        self.output_off()
        self.instrument.close()