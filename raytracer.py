#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyopencl, cv2, numpy, json, sys

class RayTracer(object):
	def __init__(self):
		self.get_context()
		self.create_buffers()
		self.run_kernel()

	def get_context(self):
		platforms = pyopencl.get_platforms()
		if not platforms:
			raise EnvironmentError("No tienes plataformas OpenCL, intenta instalar los drivers.")

		self.device = None
		for plt in platforms:
			dev = plt.get_devices(pyopencl.device_type.ALL)
			if dev:
				for d in dev:
					if self.device is None:
						self.device = d
					elif d.max_clock_frequency > self.device.max_clock_frequency and d.max_compute_units > self.device.max_compute_units:
						self.device = d

		self.ctx = pyopencl.Context([self.device])
		self.queue = pyopencl.CommandQueue(self.ctx)
		self.program = pyopencl.Program(self.ctx, open("raytracer.cl", "r").read()).build()
		self.kernel = pyopencl.Kernel(self.program, "raytracer")
	
	"""
	void getLWI(void *x, void *y, uint32_t si, uint64_t max){
		uint64_t c = (uint64_t) pow(1.0*max, 1.0/si);
		for (uint32_t j=0; j<si; j++){
			if (((uint64_t*)x)[j] < c){
				((uint64_t*)y)[j] = ((uint64_t*)x)[j];
				continue;
			}
			((uint64_t*)y)[j] = c;
			while (((uint64_t*)x)[j]%((uint64_t*)y)[j] != 0){
				((uint64_t*)y)[j]--;
			}
		}
	}

	"""



	def create_work_items(self, shape):
		works = [0]*len(shape)
		ma = int(self.device.max_work_group_size**(1/len(shape)))
		for j in range(len(shape)):
			if shape[j] < ma:
				works[j] = sha[j]
			works[j] = ma
			while shape[j]%works[j] != 0:
				works[j] -= 1
		return tuple(works)

	def create_buffers(self):

		print("Creando luces . . .")
		lights = json.loads(open(sys.argv[3], "r").read())
		self.nlight = len(lights)//4
		self.lights = pyopencl.Buffer(self.ctx, flags=pyopencl.mem_flags.READ_WRITE|pyopencl.mem_flags.COPY_HOST_PTR, hostbuf=numpy.array(lights, dtype=numpy.uint32))
		print("Luces creadas ;-)")
		
		print("Creando colores . . .")
		self.colors = pyopencl.Buffer(self.ctx, flags=pyopencl.mem_flags.READ_WRITE|pyopencl.mem_flags.COPY_HOST_PTR, hostbuf=numpy.array(json.loads(open(sys.argv[4]).read()), dtype=numpy.uint8))
		print("Colores creados ;-)")

		print("Creando imagenes . . .")
		self.background = cv2.imread(sys.argv[2])
		self.background = cv2.cvtColor(self.background, cv2.COLOR_RGB2GRAY)
		self.back_buf = pyopencl.Buffer(self.ctx, flags=pyopencl.mem_flags.READ_WRITE|pyopencl.mem_flags.COPY_HOST_PTR, hostbuf=self.background)
		print(self.background)

		self.scene = cv2.imread(sys.argv[1])
		self.scene = cv2.cvtColor(self.scene, cv2.COLOR_RGB2RGBA)

		self.img = pyopencl.Buffer(self.ctx, flags=pyopencl.mem_flags.READ_WRITE|pyopencl.mem_flags.COPY_HOST_PTR, hostbuf=self.scene)
		self.img_map, _ = pyopencl.enqueue_map_buffer(self.queue, self.img, pyopencl.mem_flags.READ_WRITE, 0,
			self.scene.shape, numpy.uint8)
		print("Imagenes creadas")

	def run_kernel(self):
		print("Corremos el kernel")
		self.kernel.set_arg(0, self.img)
		self.kernel.set_arg(1, self.colors)
		self.kernel.set_arg(2, self.lights)
		self.kernel.set_arg(3, self.back_buf)
		self.kernel.set_arg(4, numpy.uint32(self.nlight))
		self.kernel.set_arg(5, numpy.uint32(self.scene.shape[1]))
		
		print("Corriendo . . .")
		pyopencl.enqueue_nd_range_kernel(self.queue, self.kernel, self.scene.shape[:2][::-1], self.create_work_items(self.scene.shape[:2][::-1])).wait()

		#self.i = cv2.add(self.scene, self.img_map)
		cv2.imwrite(sys.argv[5], self.img_map)
		print("Terminado")


if len(sys.argv) < 6:
	print(sys.argv[0], "[Image] [Image Geometric] [config] [color] [result]")
	sys.exit(1)

print(sys.argv)
RayTracer()