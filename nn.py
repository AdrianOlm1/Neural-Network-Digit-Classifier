import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image
from BACKTRACKER import solveSudoku

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjusted for feature size
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 128 * 3 * 3)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Remove softmax here (handled by loss function)
############################################################################################################
###################### Preprocesses image for the NN
############################################################################################################

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

def preprocess(cell):
    cell = Image.fromarray(cell)
    cell = transform(cell)
    cell = cell.unsqueeze(0)
    return cell

############################################################################################################
###################### Turns image into grayscale and then performs adapative threshold on image
############################################################################################################

image = cv2.imread('test.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(image)
plt.title("Gray scaled image")
plt.show()

blurred = cv2.GaussianBlur(gray, (5,5), 0)

thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 2)

plt.imshow(thresh, cmap="gray")
plt.title("Threshold")
plt.show()

############################################################################################################
###################### Detects a sudoku grid
############################################################################################################

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours,key= cv2.contourArea, reverse=True)

sudoku_contour = None
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    aprox = cv2.approxPolyDP(contour, epsilon, True)
    if len(aprox) == 4:
        sudoku_contour = aprox
        break

grid_image = image.copy()
cv2.drawContours(grid_image, [sudoku_contour], -1, (0,255,0), 3)

plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
plt.title("Located Grid")
plt.show()

############################################################################################################
###################### Warps the photo to align the grid nicely
############################################################################################################

sudoku_contour = sudoku_contour.reshape(4, 2)

sudoku_contour = sorted(sudoku_contour, key=lambda x: (x[1], x[0]))
top_left, top_right = sorted(sudoku_contour[:2], key=lambda x: x[0])
bottom_left, bottom_right = sorted(sudoku_contour[2:], key=lambda x: x[0])

size = 450
pt1 = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
pt2 = np.array([[0,0], [size, 0], [size, size], [0, size]], dtype="float32")

matrix = cv2.getPerspectiveTransform(pt1, pt2)
warped = cv2.warpPerspective(gray, matrix ,(size,size))

plt.imshow(warped, cmap="gray")
plt.title("warped sudoku grid")
plt.show()

############################################################################################################
###################### Turns whole photo into cells to be preprocessed
############################################################################################################

cell_size = size // 9
cells = []

for i in range(9):
    row = []
    for j in range(9):
        x_start, y_start = j * cell_size, i*cell_size
        x_end, y_end = (j+1) *cell_size, (i+1) *cell_size

        cell = warped[y_start:y_end, x_start:x_end]
        row.append(cell)

    cells.append(row)

plt.imshow(cells[0][0], cmap="gray")
plt.title("Example of grid taken")
plt.show()

############################################################################################################
###################### Creates an array from the grid of photos and cnn model
############################################################################################################

fig, axes = plt.subplots(9, 9, figsize=(8, 8))

for i in range(9):
    for j in range(9):
        preprocessed_cell = preprocess(cells[i][j])  # Apply preprocessing
        img = preprocessed_cell.squeeze().numpy()  # Convert tensor to NumPy array

        ax = axes[i, j]
        ax.imshow(img, cmap="gray")  # Show preprocessed cell
        ax.axis("off")

plt.suptitle("Preprocessed Sudoku Grid Cells", fontsize=16)
plt.show()

model = CNN()
model.load_state_dict(torch.load("sudoku_cnn_weights.pth", map_location=torch.device("cpu")))

model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

sudoku_grid = np.zeros((9,9), dtype=int)

for i in range(9):
    for j in range(9):
        cell = preprocess(cells[i][j])

        with torch.no_grad():
            output = model(cell)
            digit = torch.argmax(output).item() 

        confidence = torch.softmax(output, dim=1).max().item()  # Convert logits to probabilities
        if confidence < 0.9:  # Check probability of the predicted digit
            digit = 0

        sudoku_grid[i][j] = digit

for row in sudoku_grid:
    print(" ".join(map(str, row)))

solveSudoku(sudoku_grid)
print("\nSolved Sudoku Grid:")
for row in sudoku_grid:
    print(" ".join(map(str, row)))