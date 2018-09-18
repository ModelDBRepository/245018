function [ conv1_pre, conv1_post,tot_conv, trace, W] = weights_update_stdp(A_plus, A_minus, tau_plus, tau_minus, X, Y, conv1_pre, conv1_post, trace, tau_e)

[a,b] = size(conv1_pre);
% tot_conv = total change to apply to the synapse * learning rate
conv_pre_old = convolution_type2(conv1_pre, tau_plus,  A_plus, zeros(a, b)); %pre trace without spikes - used for coincident spikes
conv_post_old = convolution_type2(conv1_post, tau_minus,A_minus, zeros(a, b)); %post trace without spikes - used for coincident spikes

%%STDP 
[conv_pre, conv1_pre] = convolution_type2 (conv1_pre, tau_plus,  A_plus, X); %trace given by pre-synaptic neuron, amplitude A+ and time constant tau+
[conv_post, conv1_post] = convolution_type2 (conv1_post, tau_minus,A_minus, Y); %trace given by post-synaptic neuron, amplitude A- and time constant tau-
W = (conv_pre.*Y + conv_post.*X).*(X+Y~=2)+((conv_pre_old.*Y + conv_post_old.*X)+(A_plus+A_minus)/2).*(X+Y==2); %total change in synapse due to stpd 

%%Eligibility Trace
[tot_conv, trace] = convolution_type2 (trace, tau_e,  1, W); %all weight changes filtered through the eligibility trace

end