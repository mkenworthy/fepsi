from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.stats import binned_statistic

# Simple AO system written by Sebastiaan Haffert

def run_simple_ao():
	#The reference wavelength for the electric field propagation
	wavelength_0 = 0.75E-6

	# Create the input grid in the pupil
	D = 8
	Np = 128
	pupil_grid = make_pupil_grid(Np, D)

	# Create the telescope
	Dtel = 8
	aperture = circular_aperture(Dtel)(pupil_grid)
	telescope = Apodizer(aperture)

	# Make DM
	print("Make DM")
	num_act = 40
	pitch = Dtel/num_act
	act_positions = make_uniform_grid((num_act, num_act), num_act*pitch)
	mask = act_positions.as_('polar').r<(Dtel/2)
	masked_act_positions = act_positions.subset(mask)
	modes = make_gaussian_pokes(pupil_grid, masked_act_positions, pitch * np.sqrt(2) )
	modes = ModeBasis([mode*aperture for mode in modes])

	print("Make grids")
	# Create the focal plane with q pixels per wavelength_0/D and a field of view with num_airy Airy rings
	focal_plane = make_focal_grid(pupil_grid, q=4, num_airy=25, wavelength=wavelength_0)

	# Create propagators to go from the pupil to the focal plane
	pupil_to_focal = FraunhoferPropagator(pupil_grid, focal_plane, wavelength_0=wavelength_0)

	# Create coronagraph
	coronagraph = PerfectCoronagraph(aperture)

	# Make atmosphere
	r0 = 0.1
	ref_wavelength = 500e-9
	L0 = 50
	velocity = np.array([0, 25])
	speed = np.sqrt( velocity.dot(velocity) )
	r0_wave = r0 * (wavelength_0/ref_wavelength)**(6/5)
	t0 = D/velocity[1]

	# Make timescales
	delta_t = pupil_grid.delta[0]/speed
	t_end = 1
	num_sim = int( np.ceil(t_end/delta_t) )
	sim_time = np.linspace(0, t_end, num_sim)


	print("Setup the atmosphere")
	layers = [InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, ref_wavelength), L0, velocity, 0, 2)]
	print("Setup the AO")
	# Choose time lag of the AO system here
	ao_layers = [ModalAdaptiveOpticsLayer(layer, modes, lag=0) for layer in layers]

	atmosphere = MultiLayerAtmosphere(ao_layers, False)

	wf = Wavefront( Field( np.ones((Np**2,),dtype=np.complex), pupil_grid), 0.5E-6)#wavelength_0)
	wf.total_power=1

	Inorm = pupil_to_focal(telescope.forward(wf)).intensity.max()
	PSF = 0
	cPSF = 0
	
	print("Start simulation")
	for i, t in enumerate(sim_time):
		atmosphere.evolve_until(t+t0)
		
		wf_at = atmosphere.forward(wf)
		wft = telescope.forward(wf_at)
		wf_c = coronagraph.forward(wft)
		
		wf_foc = pupil_to_focal(wft)
		wf_foc_c = pupil_to_focal(wf_c)

		PSFi = wf_foc.intensity/Inorm
		cPSFi = wf_foc_c.intensity/Inorm

		PSF += wf_foc.intensity/Inorm
		cPSF += wf_foc_c.intensity/Inorm

		if i % 50 == 0:
			print("Time elapsed {:g} / {:g}".format(t, t_end))
			if True:
				print("Strehl {:g}".format( PSFi.max() ) )
				print("Wavefront RMS {:g}".format( np.std(atmosphere.phase_for(wavelength_0)[aperture>0]) ) )

				plt.clf()
				plt.subplot(1,3,1)
				imshow_field( np.log10(PSFi + 1E-15), vmin=-8, vmax=0)
				plt.colorbar()
				
				plt.subplot(1,3,2)
				imshow_field( np.log10(cPSFi + 1E-15), vmin=-8, vmax=0)
				plt.colorbar()
				
				plt.subplot(1,3,3)
				imshow_field( atmosphere.phase_for(wavelength_0), pupil_grid, vmin=-np.pi, vmax=np.pi )
				plt.colorbar()

				plt.draw()
				plt.pause(0.001)
	plt.close()

	Strehl = PSF.max()/sim_time.size
	print("Strehl {:g}".format(Strehl))
	
	signal, bins, N = binned_statistic(focal_plane.as_('polar').r, PSF/sim_time.size, statistic=np.mean, bins=(2*num_act))
	bin_center = (bins[1::]+bins[0:-1])/2

	contrast, bins, N = binned_statistic(focal_plane.as_('polar').r, PSF/sim_time.size, statistic=np.std, bins=(2*num_act))
	bin_center = (bins[1::]+bins[0:-1])/2

	plt.figure()
	plt.plot(bin_center * Dtel/wavelength_0, contrast)
	plt.plot(bin_center * Dtel/wavelength_0, signal)
	plt.yscale('log')

	plt.figure()
	plt.subplot(1,2,1)
	imshow_field( np.log10(PSF/sim_time.size + 1E-15), vmin=-5, vmax=0)
	plt.colorbar()
	
	plt.subplot(1,2,2)
	imshow_field( np.log10(cPSF/sim_time.size + 1E-15), vmin=-5, vmax=0)
	plt.colorbar()

	plt.show()

if __name__ == "__main__":
	run_simple_ao()
