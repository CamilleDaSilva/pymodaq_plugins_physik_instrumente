# -*- coding: utf-8 -*-
"""
Python wrapper for the Keithley 2410 (sourcemeter).
Handles low-level communication via pyvisa (GPIB).

This file is meant to be placed in the hardware/ folder of the PyMoDAQ plugin.
It can also be used standalone in Langmuir scripts.
"""

import numpy as np
import pyvisa


class Keithley2410:

    def __init__(self, adresse: str):
        """Connect to the Keithley 2410 via GPIB.

        Parameters
        ----------
        adresse : str
            VISA address of the instrument, e.g. 'GPIB0::24::INSTR'
        """
        rm = pyvisa.ResourceManager()
        print("Connected devices:")
        [print('\t ->', element) for element in rm.list_resources()]
        self.instrument = rm.open_resource(adresse)
        self.instrument.timeout = 5000
        self.instrument.read_termination = '\n'

    def init_scan(self, voltMin: float, voltMax: float, NV: int,
                      compliance: float = 700e-3, current_range: float = 20e-3):
        """Initialize the Keithley for a voltage sweep with current measurement.

        Parameters
        ----------
        voltMin : float - minimum sweep voltage [V]
        voltMax : float - maximum sweep voltage [V]
        NV : int - number of measurement points
        compliance : float - current limit [A] (default 700 mA)
        current_range : float - current measurement range [A] (default 20 mA)

        Returns
        -------
        volts : array - list of measurement voltages
        """
        volts = np.linspace(voltMin, voltMax, NV)
        voltRang = abs(voltMin) + abs(voltMax)
        self.instrument.write('*rst')
        self.instrument.write(':SYST:BEEP:STAT OFF')
        self.instrument.write(f':SENS:CURR:PROT {compliance}')
        self.instrument.write(':SOUR:FUNC VOLT')
        self.instrument.write(':SOUR:VOLT:MODE FIX')
        self.instrument.write(f':SOUR:VOLT:RANG {voltRang}')
        self.instrument.write(f':SOUR:VOLT:LEV {voltMin}')
        self.instrument.write(f':SENS:CURR:RANG {current_range}')
        return volts

    def set_voltage(self, volt: float):
        """Apply a voltage on the Keithley output.

        Parameters
        ----------
        volt : float - voltage to apply [V]
        """
        self.instrument.write(':OUTP ON')
        self.instrument.write(f':SOUR:VOLT:LEV {volt}')

    def get_voltage(self) -> float:
        """Read the voltage currently being sourced.

        Returns
        -------
        float - sourced voltage [V]
        """
        response = self.instrument.query(':SOUR:VOLT:LEV?')
        return float(response.strip())

    def read_current(self) -> float:
        """Read the current measured by the Keithley.

        Returns
        -------
        float - measured current [A]
        """
        response = self.instrument.query('READ?')
        values = response.split(',')
        return float(values[1])

    def measure(self, volt: float) -> tuple:
        """Apply a voltage and return the measured (voltage, current).

        Parameters
        ----------
        volt : float - voltage to apply [V]

        Returns
        -------
        tuple : (voltage [V], current [A])
        """
        self.set_voltage(volt)
        response = self.instrument.query('READ?')
        values = response.split(',')
        return float(values[0]), float(values[1])

    def output_off(self):
        """Turn off the Keithley output."""
        self.instrument.write(':OUTP OFF')

    def close(self):
        """Turn off the output and close the connection."""
        self.output_off()
        self.instrument.close()