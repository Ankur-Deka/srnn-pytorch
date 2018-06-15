# Function to convert world point to pixel point
def get_pix(pt,origin,res,size):
	# origin: [x,y] coordinate of (0,h-1) pixel in the world frame
	# res: metres per pixel
	# size: [width,height] in pixels
	# pt: [x,y] in world coordinate in metres
	pixel_per_metre=1/res
	pix=[]
	[h,w]=size
	[x,y]=pt
	[x0,y0]=origin
	pix.append(int((h-1)-(y-y0)*pixel_per_metre))
	pix.append(int((x-x0)*pixel_per_metre))
	return(pix)