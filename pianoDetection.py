import numpy as np
import cv2
import itertools as it

def get_Y_indices(line1,line2,img):
	for rho,theta in line1:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		if (x2-x1) == 0:
			x1_plot_1 = x1
			y1_plot_1 = 0
			x1_plot_2 = x1_plot_1
			y1_plot_2 = int(img.shape[0])
		else:
			m = (y2-y1) / (x2-x1)
			b = y0 - m * x0
			x1_plot_1 = 0
			y1_plot_1 = int(b)
			x1_plot_2 = int(img.shape[1])
			y1_plot_2 = int(m * x1_plot_2 + b)

	for rho,theta in line2:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		if (x2-x1) == 0:
			x2_plot_1 = x1
			y2_plot_1 = 0
			x2_plot_2 = x2_plot_1
			y2_plot_2 = int(img.shape[0])
		else:
			m = (y2-y1) / (x2-x1)
			b = y0 - m * x0
			x2_plot_1 = 0
			y2_plot_1 = int(b)
			x2_plot_2 = int(img.shape[1])
			y2_plot_2 = int(m * x2_plot_2 + b)
	return y1_plot_2, y2_plot_2

def detectKeyboard(img):
	# Resize to 160x120 to make processing faster
	resized = cv2.resize(img, (160,120), cv2.INTER_AREA)
	ratio = img.shape[0] / float(resized.shape[0])
	
	# Binarize image
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	sobely = cv2.Scharr(thresh,cv2.CV_64F,0,1)
	
	# Identify horizontal lines and filter out lines that are off by more than 5 degrees
	lines = cv2.HoughLines(sobely.astype(np.uint8), 1, np.pi/2, 100)
	lines = [l for l in lines if abs(l[0][1]-np.pi/2) < np.pi/180.0*5]

	# Compare brightness of lower one-third and upper one-third to determine the line-pair that crops the keyboard
	pairs = list(it.combinations(lines,2))
	keyboardLines = []
	maxDiff = 0

	# Cache the y-indices of the required horizontal lines to facilitate cropping
	bottomX_cached = 0
	topX_cached = 0

	for (line1,line2) in pairs:
		y1,y2 = get_Y_indices(line1,line2,img)

		topX = min(y1,y2)
		bottomX = max(y1,y2)
		twoThird = int(2./3 * (bottomX-topX) + topX)
		top = gray[topX:twoThird+1,:]
		bottom = gray[twoThird:bottomX+1,:]

		diff = np.mean(bottom) - np.mean(top)
		if  diff >= maxDiff and (bottomX-topX) > (bottomX_cached-topX_cached):
			maxDiff = diff
			keyboardLines = []
			topX_cached = topX
			bottomX_cached = bottomX
			keyboardLines.append(line1)
			keyboardLines.append(line2)

	for line in keyboardLines:
		for rho,theta in line:
		    a = np.cos(theta)
		    b = np.sin(theta)
		    x0 = a*rho
		    y0 = b*rho
		    x1 = int(x0 + 1000*(-b))
		    y1 = int(y0 + 1000*(a))
		    x2 = int(x0 - 1000*(-b))
		    y2 = int(y0 - 1000*(a))

		    if (x2-x1) == 0:
		    	x_plot_1 = x1
		    	y_plot_1 = 0
		    	x_plot_2 = x1
		    	y_plot_2 = int(img.shape[0])
		    else:
		    	m = (y2-y1) / (x2-x1)
		    	b = y0 - m * x0
		    	x_plot_1 = 0
		    	y_plot_1 = int(b)
		    	x_plot_2 = int(img.shape[1])
		    	y_plot_2 = int(m * x_plot_2 + b)

		    cv2.line(img,(int(x_plot_1*ratio),int(y_plot_1*ratio)), (int(x_plot_2*ratio),int(y_plot_2*ratio)),(0,255,0),2)
		    cv2.line(resized,(x_plot_1,y_plot_1),(x_plot_2,y_plot_2),(0,255,0),1)
	cv2.imshow('res',resized)
	cv2.imwrite('result.jpg', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return

def main():
	img = cv2.imread('keyboard-1.jpg')
	detectKeyboard(img)
	
	return

if __name__ == '__main__':
	main()
