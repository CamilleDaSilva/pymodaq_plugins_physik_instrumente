# -*- coding: utf-8 -*-
"""
Python wrapper for the Keithley 2420 (source-meter used as an ammeter for the
RPA collector current measurement).
Low-level communication handled via pyvisa (GPIB).
Meant to be placed in the hardware/ folder of the PyMoDAQ plugin, next to
keithley2410_wrapper.py. Can also be used standalone in scripts.

Keithley 2420 ratings: 60 V / 3 A (vs 1100 V / 1 A on the 2410) — defaults
below are adapted accordingly. Adjust compliance / current_range to your
actual RPA collector circuit before first use.

Author: Laurianne
ONERA DPHY/CSE — PICOMAX-E, 2026
"""
import numpy as np
import pyvisa


class Keithley2420:
    def __init__(self, adresse: str):
        """Connect to the Keithley 2420 via GPIB.

        Parameters
        ----------
        adresse : str
            VISA address of the instrument, e.g. 'GPIB0::17::INSTR'
        """
        rm = pyvisa.ResourceManager()
        print("Connected devices:")
        [print('\t ->', element) for element in rm.list_resources()]
        self.instrument = rm.open_resource(adresse)
        self.instrument.timeout = 5000
        self.instrument.read_termination = '\n'

    def init_mesure(self, source_voltage: float = 0.0,
                     compliance: float = 100e-3, current_range: float = 20e-3):
        """Initialize the Keithley 2420 to source a fixed voltage (typically
        0 V for a pure ammeter usage on the RPA collector) and measure current.

        Parameters
        ----------
        source_voltage : float - fixed voltage sourced on the collector [V]
        compliance : float - current compliance limit [A] (2420 max 3 A)
        current_range : float - current measurement range [A]
        """
        self.instrument.write('*rst')
        self.instrument.write(':SYST:BEEP:STAT OFF')
        self.instrument.write(f':SENS:CURR:PROT {compliance}')
        self.instrument.write(':SOUR:FUNC VOLT')
        self.instrument.write(':SOUR:VOLT:MODE FIX')
        self.instrument.write(f':SOUR:VOLT:LEV {source_voltage}')
        self.instrument.write(f':SENS:CURR:RANG {current_range}')
        self.instrument.write(':OUTP ON')

    def set_source_voltage(self, volt: float):
        """Change the fixed voltage sourced on the collector.

        Parameters
        ----------
        volt : float - voltage to apply [V]
        """
        self.instrument.write(f':SOUR:VOLT:LEV {volt}')

    def read_current(self) -> float:
        """Read the current measured by the Keithley 2420.

        Returns
        -------
        float - measured current [A]
        """
        response = self.instrument.query('READ?')
        values = response.split(',')
        return float(values[1])

    def measure(self) -> float:
        """Trigger a single current measurement and return it.

        Returns
        -------
        float - measured current [A]
        """
        response = self.instrument.query('READ?')
        values = response.split(',')
        return float(values[1])

    def output_off(self):
        """Turn off the Keithley output."""
        self.instrument.write(':OUTP OFF')

    def close(self):
        """Turn off the output and close the connection."""
        self.output_off()
        self.instrument.close()