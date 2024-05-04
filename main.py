'''
Diffraction Grating Project

Using the picture we took of the diffraction grating, we use Adobe Illustrator
to draw the red dots so that we can easily detect the red dots.

We then use OpenCV to detect the red dots and calculate the distance between
the red dots. 
'''
import cv2
import numpy as np
import math

'''
Part 1: Detecting the Red Dots
'''
# Load the image
image = cv2.imread('image_red_dot2.jpg')

# Define range for red color in BGR. )
lower_red = np.array([0,0,200]) # slight tolerance for lighting conditions
upper_red = np.array([50,50,255])

# Threshold the BGR image to get only red colors
mask = cv2.inRange(image, lower_red, upper_red)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Print the number of detected red dots
print(f"Number of red dots: {len(contours)}")

'''
Part 2: Calculating the Distance Between the Red Dots and the average distance
'''
# Label the red dots we need
left_of_center_dot = contours[3]
center_dot = contours[2]
right_of_center_dot = contours[1]

# start off by drawing the 2 pink circles on the left contour and the center contour
cv2.drawContours(image, [left_of_center_dot], -1, (255, 0, 255), 2)
cv2.drawContours(image, [center_dot], -1, (255, 0, 255), 2)
cv2.drawContours(image, [right_of_center_dot], -1, (255, 0, 255), 2)

# Calculate the center coordinates of the left dot
M_left = cv2.moments(left_of_center_dot)
cx_left = int(M_left['m10'] / M_left['m00'])
cy_left = int(M_left['m01'] / M_left['m00'])

# Calculate the center coordinates of the center dot
M_center = cv2.moments(center_dot)
cx_center = int(M_center['m10'] / M_center['m00'])
cy_center = int(M_center['m01'] / M_center['m00'])

# Calculate the center coordinates of the right dot
M_right = cv2.moments(right_of_center_dot)
cx_right = int(M_right['m10'] / M_right['m00'])
cy_right = int(M_right['m01'] / M_right['m00'])

# draw a line between the left and center dot, and the center and right dot
cv2.line(image, (cx_left, cy_left), (cx_center, cy_center), (255, 0, 0), 2)
cv2.line(image, (cx_center, cy_center), (cx_right, cy_right), (255, 0, 0), 2)

# distance of line between center and right dot, and center and left dot
distance1 = math.sqrt((cx_center - cx_left)**2 + (cy_center - cy_left)**2)
distance2 = math.sqrt((cx_center - cx_right)**2 + (cy_center - cy_right)**2)

# Convert the pixel distance to mm using the scale factor
image_length = 1130 # pixels
real_distance = 26 # cm

scale_factor = real_distance / image_length
distance1_cm = distance1 * scale_factor
distance2_cm = distance2 * scale_factor
print(f"Distance between left and center: {distance1_cm} mm")
print(f"Distance between center and right: {distance2_cm} mm")

# Calculate the average distance between the red dots and round it to 3 decimal places
average_distance = round((distance1_cm + distance2_cm) / 2, 3)
print(f"Average distance between red dots: {average_distance} mm")

'''
Part 3: Calculating the wavelength of the light
'''
m = 1
l = 10 # cm
# d - 1/ N where N is the number of lines per mm and times it by 0.0254 to convert inches to cm
d = (1/ 13400) * 0.0254 # cm
theta = math.degrees(math.atan(average_distance / l))
sin_theta = math.sin(math.radians(theta))
wavelength = (d * sin_theta) / m
wavelength_nm = round(wavelength * 1e9, 3)
print(f"Wavelength of the light: {wavelength_nm} nm")

'''
Part 4: Displaying the Image and Saving the Image
'''
cv2.imshow('Detected Red Dots', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('detected_red_dots.jpg', image)