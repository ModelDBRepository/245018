function [Y, last_spike_post, Canc,u] = neuron(epsp, chi, last_spike_post, tau_m, rho0, theta, delta_u, i)

[N_pc, N_action] = size(epsp); %no. place cells, no. action neurons
u = sum(epsp,1)'+chi*exp((-i+last_spike_post)/tau_m); %membrane potential
rho_tilda= rho0*exp((u-theta)/delta_u); %instanteneous rate
Y= rand(N_action,1)<= rho_tilda; %realization spike train
last_spike_post(Y==1)=i; %update time postsyn spike
Canc = 1-repmat(Y, 1, N_pc); %1 if postsyn neuron spiked, 0 otherwise

end