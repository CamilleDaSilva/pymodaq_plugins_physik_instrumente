from qtpy import QtWidgets


class CompactLCD(QtWidgets.QWidget):
    """
    Small LCD panel displaying Langmuir results.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.vf_lcd = QtWidgets.QLCDNumber()
        self.i_intersection_lcd = QtWidgets.QLCDNumber()

        self.vf_lcd.setDigitCount(6)
        self.i_intersection_lcd.setDigitCount(6)

        layout = QtWidgets.QVBoxLayout()

        vf_label = QtWidgets.QLabel("Vf (V)")
        i_label = QtWidgets.QLabel("I intersection (mA)")

        layout.addWidget(vf_label)
        layout.addWidget(self.vf_lcd)

        layout.addWidget(i_label)
        layout.addWidget(self.i_intersection_lcd)

        self.setLayout(layout)


    def update_values(self, vf, i_intersection):
        """
        Update displayed values.
        """

        self.vf_lcd.display(vf)
        self.i_intersection_lcd.display(i_intersection)