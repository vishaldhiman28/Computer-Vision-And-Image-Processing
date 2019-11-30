
Steps: 
      1. Color Conversion, Binary Thresold
      2. Finding Edge contour for receipt in image using four_point_transform
      3. Crop the receipt 
      4. Using Pytesseract to get each word presenet in image and bounding box coordinates for each word
      5. For given words change color using Bounding Box coordinates


Requirements:
             1. Numpy
             2. OpenCV
             3. PIL
             4. pytesseract
             5. Imutils