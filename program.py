from PyQt5 import QtWidgets, uic
import sys
from PyQt5.QtGui import *

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('speak.ui', self)
        self.show()

        self.pushButton = self.findChild(QtWidgets.QPushButton, 'pushButton')
        self.pushButton.clicked.connect(self.buttonPressed)
        self.textEdit1 = self.findChild(QtWidgets.QTextEdit, 'textEdit1')
        self.textEdit2 = self.findChild(QtWidgets.QTextEdit, 'textEdit2')
        self.label = self.findChild(QtWidgets.QLabel, 'label')
        self.pushButton.setText("Будь ласка говоріть")

    def buttonPressed(self):        
        #self.pushButton.setText("Будь ласка говоріть")
        import testing as t
        value1 = t.text
        value2 = t.result
        if value2 == 'happy':
            s1='Щасливий'
            pixmap = QPixmap('happy.gif')
        if value2 == 'angry':
            s1='Злий'
            pixmap = QPixmap('angry.gif')
        if value2 == 'sad':
            s1='Сумний'
            pixmap = QPixmap('sad.gif')
        if value2 == 'surprised':
            s1='Здивований'
            pixmap = QPixmap('suprised.png')
        self.label.setPixmap(pixmap)
        #self.label.resize(150,150)
        self.label.setScaledContents(True)
        self.textEdit_1.setText("Ви сказали: {}".format(value1))
        self.textEdit_2.setText("Ідентифіковано емоцію: " + s1)
        
        

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
