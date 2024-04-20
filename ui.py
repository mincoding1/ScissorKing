# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui.ui'
##
## Created by: Qt User Interface Compiler version 5.15.8
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *  # type: ignore
from PySide2.QtGui import *  # type: ignore
from PySide2.QtWidgets import *  # type: ignore


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1384, 661)
        self.label1 = QLabel(Form)
        self.label1.setObjectName(u"label1")
        self.label1.setGeometry(QRect(10, 10, 640, 480))
        font = QFont()
        font.setStrikeOut(False)
        self.label1.setFont(font)
        self.label1.setStyleSheet(u"")
        self.label1.setFrameShape(QFrame.Box)
        self.label2 = QLabel(Form)
        self.label2.setObjectName(u"label2")
        self.label2.setGeometry(QRect(690, 10, 680, 480))
        self.label2.setStyleSheet(u"")
        self.label2.setFrameShape(QFrame.Box)
        self.game_label = QLabel(Form)
        self.game_label.setObjectName(u"game_label")
        self.game_label.setGeometry(QRect(330, 160, 681, 181))
        font1 = QFont()
        font1.setPointSize(40)
        font1.setBold(True)
        font1.setWeight(75)
        self.game_label.setFont(font1)
        self.game_label.setLayoutDirection(Qt.LeftToRight)
        self.game_label.setAlignment(Qt.AlignCenter)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Scissor King", None))
        self.label1.setText("")
        self.label2.setText("")
        self.game_label.setText(QCoreApplication.translate("Form", u"READY", None))
    # retranslateUi

