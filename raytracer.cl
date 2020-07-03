unsigned char check_conti(__global unsigned char *deep, unsigned int col, 
	unsigned int lx, unsigned int ly, 
	unsigned int i, unsigned int j){
	int k,h,f,y,d;
	double m;
	k = lx-i;
	h = ly-j;
	if (k < 0){
		k *= -1;
	}
	if (h < 0){
		h *= -1;
	}

	if (k < h){
		if (j < ly){
			k = j;
			h = ly;
		}else{
			k = ly;
			h = j;
		}
		m = (1.0*lx-1.0*i);
		m /= (1.0*ly-1.0*j);
		f = ly;
		d = lx;
		for (int x=k; x<h; x++){
			y = m*(x-f)+1.0*d;
			if (deep[x*col+y] == 255){
				return 1;
			}
		}
	}else{
		if (i < lx){
			k = i;
			h = lx;
		}else{
			k = lx;
			h = i;
		}
		m = (1.0*ly-1.0*j);
		m /= (1.0*lx-1.0*i);
		f = lx;
		d = ly;
		for (int x=k; x<h; x++){
			y = m*(x-f)+1.0*d;
			if (deep[y*col+x] == 255){
				return 1;
			}
		}
	}
		
	return 0;
}

unsigned char change_color(unsigned char pixel, unsigned char color, unsigned int alpha){
	unsigned char result;
	double bri, sg;
	int brii;

	bri = 1.0*alpha;
	bri /= 255.0;

	if (pixel < color){
		sg = 1.0;
	}else if (pixel > color){
		sg = -1.0;
	}
	brii = pixel*(1.0+sg*bri);
	if (brii > 255){
		brii = 255;
	}else if (brii < 0){
		brii = 0;
	}
	result = brii;
	return result;
}


__kernel void raytracer(__global unsigned char *img, __global unsigned char *color,
	__global unsigned int *light, __global unsigned char *deep, unsigned int nlight,
	unsigned int col){
	unsigned int i = get_global_id(0);
	unsigned int j = get_global_id(1);

	img[j*4*col+4*i+3] = 0;
	
	int g, gr, grb=0, gb=0;
	unsigned int kb, b=0;
	double r, rb=DBL_MAX;
	if (deep[j*col+i] != 255){
		for (unsigned int x=0; x<nlight; x++){
			r = (i-light[4*x])*(i-light[4*x]);
			r += (j-light[4*x+1])*(j-light[4*x+1]);
			
			if (sqrt(r) <= light[4*x+2]){
				if (check_conti(deep, col, light[4*x], light[4*x+1], i, j)){
					continue;
				}
				b = 1;
				g = (light[4*x+3]/(light[4*x+2]*light[4*x+2]))*(r);
				g -= ((2*light[4*x+3])/light[4*x+2])*sqrt(r);
				g += light[4*x+3];
				
				if (g > 255){
					g = 255;
				}

				gr = (r*255)/(light[4*x+2]*light[4*x+2]);
				
				if (gr > 255){
					gr = 255;
				}

				if (g > gb){
					gb = g;
					kb = x;
				}
				if (sqrt(r) < rb){
					grb = gr;
					rb = sqrt(r);
				}			
			}
		
		}
		if (b){
			rb = 1.0*deep[j*col+i];
			rb /= 255.0;
			rb = 1.0-rb;
			gb *= rb;
			for (unsigned int k=0; k<3; k++){
				img[j*4*col+4*i+k] = change_color(img[j*4*col+4*i+k], color[3*kb+k], gb);
				img[j*4*col+4*i+k] = change_color(img[j*4*col+4*i+k], 0, grb);
			}
			img[j*4*col+4*i+3] = 255;
		}
			

	}
	if (img[j*4*col+4*i+3] == 0){
		for (unsigned int k=0; k<3; k++){
			img[j*4*col+4*i+k] = 0;
		}
		img[j*4*col+4*i+3] = 255;
	}
}