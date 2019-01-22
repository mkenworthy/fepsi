from hcipy import *
import numpy as np
import matplotlib.pyplot as plt

# modified by M Kenworthy from psi_METIS

Dtel = 37.
r0 = 0.010 # pupil_diameters
L0 = 80./Dtel # pupil_diameters

wind_velocity = np.array([0.5, 0]) # pupil_diameters per sec

ao_framerate = 1000 # Hz

ao_actuators = 40

t_end = 3 # sec

central_obscuration = 0.27

ncpa_rms = 0.4 # rad

wfs_noise = 0#.2

num_photons = 1e4 # per exposure

np.random.seed(0)

#############################################################

pupil_grid = make_pupil_grid(128)
focal_grid = make_focal_grid(pupil_grid, 4, 16)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

aperture = make_obstructed_circular_aperture(1, central_obscuration)(pupil_grid)

actuator_grid = make_pupil_grid(ao_actuators)
ao_modes = make_gaussian_pokes(pupil_grid, actuator_grid, 1.2 / ao_actuators) ### not sure why 1.2/ ao-actuators?
transformation_matrix = ao_modes.transformation_matrix
reconstruction_matrix = inverse_tikhonov(transformation_matrix, 1e-4)

coro = PerfectCoronagraph(aperture)
#lyot_stop = make_obstructed_circular_aperture(0.98, 0.3)(pupil_grid)
#coro = OpticalSystem([VortexCoronagraph(pupil_grid, 2), Apodizer(lyot_stop)])

#layers = make_standard_atmospheric_layers(pupil_grid, 5)
layers = [ModalAdaptiveOpticsLayer(InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, 1), L0, wind_velocity), ao_modes, 3)]
#layers = [ModalAdaptiveOpticsLayer(InfiniteAtmosphericLayer(pupil_grid, Cn_squared_from_fried_parameter(r0, 1), L0, wind_velocity, use_interpolation=True), ao_modes, 3)]
atmosphere = MultiLayerAtmosphere(layers)

phi_I = 0
phi_2 = 0
phi_sum = 0
I_sum = 0

zernike_modes = make_zernike_basis(30, 1, pupil_grid, 2)

#ncpa = make_power_law_error(pupil_grid, ncpa_ptv, 1)
#ncpa = 0.1 * zernike(2,0)(pupil_grid) + 0.1 * zernike(3,3)(pupil_grid) - 0.2 * zernike(3,-1)(pupil_grid)
ncpa = zernike_modes.transformation_matrix.dot(np.random.randn(len(zernike_modes)) / (5 + np.arange(len(zernike_modes))))
ncpa *= ncpa_rms / np.std(ncpa[aperture > 0.5])
reconstructor_zernike = inverse_tikhonov(zernike_modes.transformation_matrix, 1e-3)


wf = Wavefront(aperture)
wf.total_power = num_photons
img_ref = prop(wf).power

mw = FFMpegWriter('METIS_PSI.mp4')

N = t_end * ao_framerate

correction = 0

for iter in range(20):
	I_sum = 0
	phi_sum = 0
	phi_2 = 0
	phi_I = 0

	for i, t in enumerate(np.arange(N) / ao_framerate):
		atmosphere.evolve_until(t + iter * t_end)
		print(t)

		wf = Wavefront(aperture)
		wf.total_power = num_photons
		wf_post_ao = atmosphere(wf)

		wf_post_coro = wf_post_ao.copy()
		wf_post_coro.electric_field *= np.exp(1j * ncpa) * np.exp(1j * correction)
		if coro is not None:
			wf_post_coro = coro(wf_post_coro)

		img = prop(wf_post_coro).power
		img_noisy = large_poisson(img) + 1e-10

		wfs_measurement = reconstruction_matrix.dot(np.angle(wf_post_ao.electric_field / wf_post_ao.electric_field.mean()))
		wfs_measurement_noisy = wfs_measurement * (1 + np.random.randn(len(wfs_measurement)) * wfs_noise)

		reconstructed_pupil = aperture * np.exp(1j * transformation_matrix.dot(wfs_measurement_noisy))
		reconstructed_pupil /= np.exp(1j * np.angle(reconstructed_pupil.mean()))
		reconstructed_pupil -= aperture

		if coro is None:
			reconstructed_electric_field = prop(Wavefront(reconstructed_pupil)).electric_field
		else:
			reconstructed_electric_field = prop(coro(Wavefront(reconstructed_pupil))).electric_field

		phi_I += reconstructed_electric_field * img_noisy
		phi_2 += np.abs(reconstructed_electric_field)**2
		phi_sum += reconstructed_electric_field
		I_sum += img_noisy

		psi_estimate = (phi_I - phi_sum * I_sum / (i+1)) / (phi_2 - np.abs(phi_sum / (i+1))**2)

		wf = Wavefront(psi_estimate * circular_aperture(15)(focal_grid))
		#wf.electric_field *= np.exp(-2j * focal_grid.as_('polar').theta)
		pup = prop.backward(wf)
		ncpa_estimate = pup.electric_field.imag

		if i % 50 == 0:
			m_original = reconstructor_zernike.dot(ncpa * aperture)
			m_corrected = reconstructor_zernike.dot((ncpa + correction) * aperture)
			rms = np.std(zernike_modes.transformation_matrix.dot(reconstructor_zernike.dot((ncpa + correction)))[aperture > 0.5]) * 3000 / 2 / np.pi

			plt.clf()
			plt.suptitle('iter=%d; t=%0.3f; npca=%0.1f nm rms' % (iter, t+iter*t_end, rms))
			plt.subplot(2,3,1)
			imshow_field(np.log10(I_sum / (i+1) / img_ref.max()), vmin=-4, vmax=-2)
			plt.title("Science Camera Intensity")
			plt.subplot(2,3,2)
			imshow_field(reconstructed_electric_field / np.abs(reconstructed_electric_field)**0.5)
			plt.title("Electric field")
			plt.subplot(2,3,3)
			imshow_field(transformation_matrix.dot(wfs_measurement) * aperture, pupil_grid, cmap='RdBu', mask=aperture)
			plt.title("Reconstructed WFS pupil")
			plt.subplot(2,3,4)
			imshow_field((psi_estimate / np.abs(psi_estimate)**0.5) * circular_aperture(15)(focal_grid))
			plt.title("NCPA in Science Camera")
			plt.subplot(2,3,5)
			imshow_field(ncpa_estimate * aperture, cmap='RdBu', mask=aperture)
			plt.title("NCPA Estimate in Pupil")
			ax = plt.subplot(2,3,6)
			#imshow_field((ncpa + correction) * aperture, cmap='RdBu', mask=aperture)
			plt.plot(m_original * 3000 / 2 / np.pi, 'ro-', label='Uncorrected')
			plt.plot(m_corrected * 3000 / 2 / np.pi, 'bo-', label='Corrected')
			plt.legend()
			plt.title("NCPA Zernike")
			plt.xlabel("Zernike number")
			plt.ylabel("Amplitude (nm r.m.s.)")
			ax.yaxis.set_label_position("right")
			plt.draw()
			plt.pause(0.01)
			mw.add_frame()
	correction += -0.05 * ncpa_estimate
mw.close()
