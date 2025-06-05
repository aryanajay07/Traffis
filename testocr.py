import easyocr
import cv2
try:
    reader = easyocr.Reader(['ne'], gpu=True)
except:
    print("GPU not available for EasyOCR. Using CPU.")
    reader = easyocr.Reader(['ne'], gpu=False)

image='plate3.png'
image=cv2.imread(image)

cv2.imshow("Image",image)
cv2.waitKey(2000)
# Convert to grayscale if needed
print("Lemgth of image:",len(image.shape))
if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image

cv2.imshow("Image",gray)
cv2.waitKey(2000)
ocr_result1 = reader.readtext(gray)
text1= max(ocr_result1, key=lambda x: x[2])[1] if ocr_result1 else "No text found"
print("After Gray:",text1)
# Apply adaptive thresholding
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

cv2.imshow("Image",binary)
cv2.waitKey(2000)
ocr_result2 = reader.readtext(binary) 
text2= max(ocr_result2, key=lambda x: x[2])[1] if ocr_result2 else "No text found"
print("After Binary:",text2)
# Denoise
denoised = cv2.fastNlMeansDenoising(binary)

cv2.imshow("Image",denoised)
cv2.waitKey(2000)
ocr_result3 = reader.readtext(denoised)
text3= max(ocr_result3, key=lambda x: x[2])[1] if ocr_result3 else "No text found"
print("After Denoise:",text3)
# Convert back to RGB for OCR
rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

cv2.imshow("Image",rgb)
cv2.waitKey(2000)
ocr_result4 = reader.readtext(rgb)
text4= max(ocr_result4, key=lambda x: x[2])[1] if ocr_result4 else "No text found"
print("After rgb:",text4)

ocr_result = reader.readtext(image)
if ocr_result:
    text = max(ocr_result, key=lambda x: x[2])[1]

print("Without preprocess",text)
    