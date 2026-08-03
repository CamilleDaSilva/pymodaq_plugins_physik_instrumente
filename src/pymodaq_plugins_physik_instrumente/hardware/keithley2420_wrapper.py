# -*- coding: utf-8 -*-
"""
Python wrapper for the Keithley 2420 (source-meter used as an ammeter for the
RPA collector current measurement).

Author: Laurianne
ONERA DPHY/CSE — PICOMAX-E, 2026
"""

import threading
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
        self._lock = threading.Lock()
        print(self.instrument.query('*IDN?'))

    def init_mesure(
        self,
        source_voltage: float = 0.0,
        compliance: float = 100e-3,
        current_range: float | None = 20e-3,
        nplc: float = 1.0,
    ):
        """
        Initialise le Keithley en mesure de courant (mode ampèremètre).

        Parameters
        ----------
        source_voltage : float
            Tension appliquée au collecteur (0 V pour usage ampèremètre pur).
        compliance : float
            Limite de courant.
        current_range : float or None
            None -> autorange
            float -> plage fixe en ampères.
        nplc : float
            Nombre de cycles secteur par mesure (précision vs vitesse).
        """
        inst = self.instrument

        with self._lock:
            inst.write("*RST")
            inst.write("*CLS")
            inst.write(":SYST:BEEP:STAT OFF")

            # Source 0 V (or small bias) so the instrument acts as ammeter
            inst.write(":SOUR:FUNC VOLT")
            inst.write(":SOUR:VOLT:MODE FIX")
            inst.write(f":SOUR:VOLT:LEV {source_voltage}")

            # Measure current
            inst.write(':SENS:FUNC "CURR"')
            inst.write(f":SENS:CURR:PROT {compliance}")
            inst.write(f":SENS:CURR:NPLC {nplc}")

            if current_range is None:
                inst.write(":SENS:CURR:RANG:AUTO ON")
            else:
                inst.write(":SENS:CURR:RANG:AUTO OFF")
                inst.write(f":SENS:CURR:RANG {current_range}")

            # Return only current (same as working script)
            inst.write(":FORM:ELEM CURR")

            inst.write(":OUTP ON")

    def set_source_voltage(self, volt: float):
        with self._lock:
            self.instrument.write(f":SOUR:VOLT:LEV {volt}")

    def read_current(self) -> float:
        with self._lock:
            return float(self.instrument.query(":READ?"))

    def measure(self) -> float:
        """Alias for read_current – returns the measured current in A."""
        with self._lock:
            return float(self.instrument.query(":READ?"))

    def output_off(self):
        with self._lock:
            self.instrument.write(":OUTP OFF")

    def close(self):
        self.output_off()
        with self._lock:
            self.instrument.close()
