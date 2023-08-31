
import bilby


def get_likelihood_on_grid():
    likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=self.ifos, 
            waveform_generator=self.waveform_generator, 
            phase_marginalization=phase_marginalization,
            distance_marginalization=self.config["testing"]["distance_marginalisation"],
            priors=self.config["priors"])