import sys
from PyQt5.QtWidgets import * 
from PyQt5 import QtGui
import numpy as np
                    
   
#Main Window
class App(QWidget):
	def __init__(self):
		super().__init__()
		self.title = 'PyQt5 - QTableWidget'
		self.left = 0
		self.top = 0
		self.width = 300
		self.height = 200

		self.reward_matrix = np.load('matrices/reward_matrix_teste.npy', allow_pickle=False)

		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		self.createTable()

		self.messageList = []

		self.layout = QVBoxLayout()
		self.layout.addWidget(self.tableWidget)
		self.setLayout(self.layout)

		#Show window
		self.show()

	def updateValue(self, row, column):
		print("Row %d and Column %d was clicked" % (row, column))
		self.tableWidget.item(row, column).setBackground(QtGui.QColor(100,100,150))
		self.reward_matrix[row][column] = 10
		np.save('matrices/reward_matrix_teste.npy', self.reward_matrix)

	def onClickedRow(self, index=None):
		print(index.row(), index.column(), self.messageList.data(index, QtCore.Qt.DisplayRole))

	#Create table
	def createTable(self):
		y, x = self.reward_matrix.shape

		self.tableWidget = QTableWidget()

		#self.tableWidget.cellClicked.connect(self.updateValue)
		self.tableWidget.clicked.connect(self.onClickedRow)

		#Row count
		self.tableWidget.setRowCount(y) 

		#Column count
		self.tableWidget.setColumnCount(x)  

		for i in range(y):
			for j in range(x):
				self.tableWidget.setItem(i,j, QTableWidgetItem())
				if(self.reward_matrix[i][j] >= 5):
					self.tableWidget.item(i, j).setBackground(QtGui.QColor(100,100,150))

		self.tableWidget.resizeColumnsToContents()
   
if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = App()
	sys.exit(app.exec_())

'''
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

window = tk.Tk()

reward_matrix = np.load('matrices/reward_matrix_1.npy', allow_pickle=False)

print(np.shape(reward_matrix))

y, x = reward_matrix.shape

for i in range(x):
    for j in range(y):
        frame = tk.Frame(
            master=window,
            relief=tk.FLAT,
            borderwidth=0
        )
        frame.grid(row=i, column=j)
        canvas = tk.Canvas(master=frame, width=1, height=1)
        canvas.create_rectangle(0, 0, 2, 2, fill="blue", width=0)
        canvas.pack()        

window.mainloop()
'''