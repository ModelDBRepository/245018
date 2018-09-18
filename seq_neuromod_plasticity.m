%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% +ACh simulations:     set ACh_flag = 1
%
% -ACh simulations:     set ACh_flag = 0 
%
% r-STDP simulations:   set ACh_flag = 0 and 
%                           STDP parameters as desired: 
%                              A_pre_post = pre-post window amplitude 
%                              A_post_pre = post-pre window amplitude 
%                              tau_pre_post = pre-post window time constant 
%                              tau_post_pre = post-pre window time constant 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all 
set(0,'defaultlinelinewidth',2,'DefaultAxesFontSize',24)

rew1_flag=1;  %reward 1 in the top-right corner                                                                
rew2_flag=0;  %reward 2 in the bottom-left corner (moved)
plot_flag = 1; %plot trajectories at every trial 
ACh_flag=1; % cholinergic depression (1=+ACh; 0=-ACh)

step = 1; %step 1ms

%% Task parameters

Trials = 40; %number of trials 
T_max=15*10^3; %maximum time trial 
starting_position= [0,0]; %starting position
t_rew=T_max; %time of reward - initialized to maximum
t_extreme=t_rew+300; %ttime of reward - initialized to maximu
t_end = T_max;

c = [1.5,1.5]; %centre reward 1 
r_goal=0.3; %radius goal area 1

c2 = [-1.5,-1.5]; %centre reward 2 (moved)
r_goal2=0.3; %radius goal area 2 (moved)

%% Place cells

space_pc = 0.4; %place cells separation distance
bounds_x = [-2,2]; %bounds open field, x axis
bounds_y = [-2,2]; %bounds open field, y axis
x_pc = bounds_x(1):space_pc:bounds_x(2); %place cells on axis x
n_x = length(x_pc); %nr of place cells on axis x
y_pc= bounds_y(1):space_pc:bounds_y(2); %place cells on axis y
n_y = length(y_pc); %nr of place cells on axis y
pos = zeros(1,2); %position of the agent at each timestep

%create grid
y = repmat(y_pc, n_x,1);
y= reshape(y,n_x*n_y,1);
x = repmat(x_pc,1,n_y);
pc = [x',y]; %place cells' centres (x,y) 
pc = round(pc*10)/10;
N_pc=length(pc); %number of place cells
rho_pc=400*10^(-3); %maximum firing rate place cells
sigma_pc=0.4; %pc separation distance

%% Action neurons - neuron model

eps0=20; %scaling constant epsp
tau_m=20; %membrane time constant
tau_s=5; %synaptic time rise epsp
chi=-5; %scaling constant refractory effect
rho0=60*10^(-3); %scaling rate
theta=16; %threshold
delta_u=2; %escape noise 

%% Action neurons - parameters

N_action=40; %number action neurons

%action selection
tau_gamma = 50; %raise time convolution action selection
v_gamma=20; %decay time convolution action selection 
theta_actor = 2*pi*[1:N_action]/N_action; %angles actions

%winner-take-all weights 
psi = 20; %the higher, the more narrow the range of excitation
w_minus = -300;
w_plus = 100;
diff_theta = repmat(theta_actor,N_action,1) - repmat(theta_actor',1, N_action);
f = exp(psi*cos(diff_theta)); %lateral connectivity function 
f = f - f.*eye(N_action);
normalised = sum(f);
normalised = normalised(1);
w_lateral = (w_minus/N_action+w_plus*f/normalised); %lateral connectivity action neurons 

%actions
a0=.08; 
actions = a0*[sin(theta_actor); cos(theta_actor)]; %possible actions (x,y)

dx = 0.01; %length of bouncing back from walls


%% synaptic plasticity parameters

A_pre_post=1;   %amplitude pre-post window
A_post_pre=1;   %amplitude post-pre window 
tau_pre_post= 10;   %time constant pre-post window
tau_post_pre= 10;   %time constant post-pre window
tau_e= 2*10^3; %time constant eligibility trace
eta_DA=0.01; %learning rate eligibility trace
eta_ACh = 10^-3*2; %learning rate acetylcholine

%feed-forward weights 
w_max=3; %upper bound feed-forward weights
w_min=1; %.pwer bound feed-forward weights
w_in = ones(N_pc, N_action)'*2; %initialization feed-forward weights

trace_pre_post= zeros(N_action, N_pc); %initialize pre-post trace  
trace_post_pre= zeros(N_action,N_pc);%initialize post-pre trace  
trace_tot = zeros(N_action,N_pc); %sum of the traces
eligibility_trace = zeros(N_action, N_pc); %total convolution 


%% initialise variables

i=0; %counter ms
tr=0; %counter trial

w_tot = [ones(N_pc,N_action)'.*w_in, w_lateral]; %total weigths 


X = zeros(N_pc,1); %matrix of spikes place cells
X_cut = zeros(N_pc+N_action, N_action);  %matrix of spikes place cells
Y_action_neurons= zeros(N_action, 1);  %matrix of spikes action neurons

time_reward= zeros(Trials,1); %stores time of reward 1
time_reward2= time_reward; %stores time of reward 2 (moved)
time_reward_old= time_reward; %stores time when agent enters the previously rewarded location 

epsp_rise=zeros(N_action+N_pc,N_action); %epsp rise compontent convolution
epsp_decay=zeros(N_action+N_pc,N_action); %epsp decay compontent convolution
epsp_tot=zeros(N_action+N_pc, N_action); %epsp

rho_action_neurons= zeros(N_action,1); %firing rate action neurons
rho_rise= rho_action_neurons;  %firing rate action neurons, rise compontent convolution
rho_decay = rho_action_neurons; %firing rate action neurons, decay compontent convolution

Canc = ones(N_pc+N_action,N_action)';
last_spike_post=zeros(N_action,1)-1000; %vector time last spike postsynaptic neuron

store_pos = zeros(T_max*Trials,2); %stores trajectories (for plotting)
firing_rate_store = zeros(N_action,T_max*Trials); %stores firing rates action neurons (for plotting)

%% initialize plot open field

figure('position',  [0, 0, 1000, 2000])
subplot(2,2,1)
reward_plot = plot(c(1)+r_goal*cos(-pi:2*pi/100:pi), c(2)+r_goal*sin(-pi:2*pi/100:pi), 'color', 'black'); %plot reward 1 
hold on
point_plot = plot(starting_position(1),starting_position(2), '.r', 'MarkerSize',10); %plot initial starting point

%plot walls
line([bounds_x(1) bounds_x(1)], [bounds_y(1) bounds_y(2)], 'color','black'); 
line([bounds_x(2) bounds_x(2)], [bounds_y(1) bounds_y(2)], 'color','black');
line([bounds_x(1) bounds_x(2)], [bounds_y(1) bounds_y(1)], 'color','black');
line([bounds_x(1) bounds_x(2)], [bounds_y(2) bounds_y(2)], 'color','black');
axis([-2 2 -2 2])

%% delete actions that lead out of the maze

%find index place cells that lie on the walls
sides(1,:) = (find(pc(:,2) == -2))'; %bottom wall, y=-2
sides(2,:) = (find(pc(:,2) == 2))'; %top wall, y=+2
sides(3,:) = (find(pc(:,1) == 2))'; %left wall, x=-2
sides(4,:) = (find(pc(:,1) == -2))'; %right wall, x=+2

%store index of actions forbidden from each side 
forbidden_actions(1,:) = 11:29; %actions that point south - theta in (180, 360) degrees approx
forbidden_actions(2,:) = [1:9, 31:40]; %actions that point north - theta in (0,180) degrees approx
forbidden_actions(3,:) = 1:19; %actions that point east - theta in (-90, 90) degrees approx
forbidden_actions(4,:) = 21:39; %actions that point west - theta in (90, 270) degrees approx

%kill connections between place cells on the walls and forbidden actions
w_walls = ones(N_action, N_pc+N_action); 
for g=1:4
    w_walls(forbidden_actions(g,:), sides(g,:))=0;
end


%% start simulation
w_tot_old = w_tot(1:N_action,1:N_pc); %store weights before start 

while i<T_max*Trials
    i=i+1;
    
    t=mod(i,T_max);
    
    %% reset new trial
    
    if t==1
        pos = starting_position; %initialize position at origin (centre open field)
        rew_found=0; %flag that signals when the reward is found
        
        tr=tr+1; %trial number 
        t_rew=T_max; %time of reward - initialized at T_max at the beginning of the trial
        
        %initialisation variables - reset between trials
        Y_action_neurons= zeros(N_action, 1); 
        X_cut = zeros(N_pc+N_action, N_action);
        epsp_rise=zeros(N_action+N_pc,N_action);
        epsp_decay=zeros(N_action+N_pc,N_action);
        epsp_tot=zeros(N_action+N_pc, N_action);
        rho_action_neurons= zeros(N_action,1);
        rho_rise=  zeros(N_action,1);
        rho_decay =  zeros(N_action,1);
        Canc = ones(N_pc+N_action,N_action)';
        last_spike_post=zeros(N_action,1)-1000;
        trace_pre_pos= zeros(N_action, N_pc);
        trace_post_pre= zeros(N_action,N_pc);
        trace_tot = zeros(N_action,N_pc);
        eligibility_trace = zeros(N_action, N_pc);
        
        %change reward location in the second half of the experiment
        if tr== (Trials/2)+1
            rew1_flag=0;
            rew2_flag=1;
            
            
            subplot(2,2,1)
            delete(reward_plot)
            reward_plot_old = plot(c(1)+r_goal*cos(-pi:2*pi/100:pi), c(2)+r_goal*sin(-pi:2*pi/100:pi), 'color', 'black', 'linestyle', '--'); %plot reward 1 
            reward_plot_new = plot(c2(1)+r_goal2*cos(-pi:2*pi/100:pi), c2(2)+r_goal2*sin(-pi:2*pi/100:pi), 'color', 'black'); %plot reward 2
        end
    end
    
    %% place cells
    
    rhos = rho_pc.*exp(-sum((repmat(pos,n_x*n_y,1)-pc).^2,2)/(sigma_pc^2)); %rate inhomogeneous poisson process
    prob = rhos;
    %turn place cells off after reward is reached 
    if t>t_rew
        prob=0;
    end
    X = (rand(1,N_pc)<=prob')'; %spike train pcs
    
    store_pos(i,:) = pos; %store position (for plotting)
    
    %% reward
    
    % agent enters reward 1 in the first half of the trial
    if sum((pos-c).^2)<=r_goal^2 && rew_found==0 && rew1_flag==1
        rew_found=1; %flag reward found (so that trial is ended soon)
        t_rew=t; %time of reward
        time_reward(tr) = t; %store time of reward
    end
    
    % agent enters reward 2 in the second half of the trial
    if sum((pos-c2).^2)<=r_goal2^2 && rew_found==0 && rew2_flag==1
        rew_found=1;  %flag reward 2 found (so that trial is ended soon)
        t_rew=t; %time of reward 2
        time_reward2(tr) = t; %store time of reward 2
        
    end
    
    % agent enters reward 1 in the second half of the trial (previously rewarded location) 
    if sum((pos-c).^2)<=r_goal^2 && rew1_flag==0 && rew2_flag==1
         %this location is no longer rewarded, so the trial is not ended
        time_reward_old(tr)=t; %store time of entrance old reward location
    end

    
    %% action neurons 
    
    % reset after last post-synaptic spike
    X_cut = repmat([X; Y_action_neurons],1,N_action);
    X_cut = X_cut.*Canc';
    epsp_rise=epsp_rise.*Canc';
    epsp_decay=epsp_decay.*Canc';
    
    % neuron model
    [epsp_tot, epsp_decay, epsp_rise] = convolution (epsp_decay, epsp_rise, tau_m, tau_s, eps0, X_cut, w_tot.*w_walls); %EPSP in the model * weights
    [Y_action_neurons,last_spike_post, Canc] = neuron(epsp_tot, chi, last_spike_post, tau_m, rho0, theta, delta_u, i); %sums EPSP, calculates potential and spikes
    
    % smooth firing rate of the action neurons
    [rho_action_neurons, rho_decay, rho_rise] = convolution (rho_decay, rho_rise, tau_gamma, v_gamma, 1, Y_action_neurons);
    firing_rate_store(:,i) = rho_action_neurons; %store action neurons' firing rates
    
    % select action
    a = ((rho_action_neurons'*actions')/N_action);
    a(isnan(a))=0;
    
    %% synaptic plasticity
    
    %STDP with symmetric window 
    [ trace_pre_pos, trace_post_pre,eligibility_trace, trace_tot, W] = weights_update_stdp(A_pre_post, A_post_pre, tau_pre_post, tau_post_pre, repmat(X',N_action,1) , repmat(Y_action_neurons,1, N_pc), trace_pre_pos, trace_post_pre, trace_tot, tau_e);
    
    % online weights update (effective only with acetylcholine - ACh_flag=1)
    w_tot(1:N_action,1:N_pc)= w_tot(1:N_action,1:N_pc)-eta_ACh*W*(ACh_flag);
    
    %weights limited between lower and upper bounds
    w_tot(w_tot(:,1:N_pc)>w_max)=w_max;
    w_tot(w_tot(:,1:N_pc)<w_min)=w_min;

    %% position update
    
    pos = pos+a;
    
    %check if agent is out of boundaries. If it is, bounce back in the opposite direction
    if pos(1)<=bounds_x(1)
        pos = pos+dx*[1,0];
    else
        if pos(1)>= bounds_x(2)
            pos = pos+dx*[-1,0];
        else
            if pos(2)<=bounds_y(1)
                pos = pos+dx*[0,1];
            else
                if pos(2)>=bounds_y(2)
                    pos = pos+dx*[0,-1];
                end
            end
        end
    end
    
    %time when trial end is 300ms after reward is found
    t_extreme = t_rew+300;
    if t> t_extreme && t<T_max
        i = (ceil(i/T_max))*T_max-1; %set i counter to the end of the trial
        t_end = t_extreme; %for plotting
    end
    
    if t==0
        
        t=T_max;
        
        %% update weights - end of trial
        
        % if the reward is not found, no change. 
        % if the reward is found, weights are retroactively potentiated through an eligibility trace
        w_tot(1:N_action,1:N_pc)= w_tot(1:N_action,1:N_pc)*(1-rew_found) + (w_tot_old+eta_DA*eligibility_trace)*rew_found;
        
        %weights limited between lower and upper bounds
        w_tot(w_tot(:,1:N_pc)>w_max)=w_max;
        w_tot(w_tot(:,1:N_pc)<w_min)=w_min;
        
        %store weights before the beginning of next trial (for updates in case reward is found) 
        w_tot_old = w_tot(1:N_action,1:N_pc);
        
        %calculate policy
        ac =actions*(w_tot_old.*w_walls(:,1:N_pc))/a0; %vector of preferred actions according to the weights 
        ac(:,unique(sort(reshape(sides, length(sides)*4, 1))))=0; %do not count actions AT the boundaries (just for plotting)
        
        
        %% plot
        if plot_flag==1
            
            %display trajectory of the agent in each trial
            subplot(2,2,1)
            f3=plot(store_pos( (floor((i-1)/T_max))*T_max+1:(floor((i-1)/T_max))*T_max+t_end,1), store_pos((floor((i-1)/T_max))*T_max+1:(floor((i-1)/T_max))*T_max+t_end,2), 'red'); %trajectory
            hold on
            delete(point_plot)
            point_plot = plot(starting_position(1),starting_position(2), '.r', 'MarkerSize',10); %starting point
            title(['Trial ', num2str(tr)])
            
            %display action neurons firing rates (activity bump)
            subplot(2,2,2)
            imagesc(firing_rate_store(:,(floor((i-1)/T_max))*T_max+1:(floor((i-1)/T_max))*T_max+t_end));
            colorbar
            title('Action neurons firing rates')
            
            %display weights over the open field, averaged over action neurons
            subplot(2,2,3)
            w_plot = mean(w_tot(:,1:N_pc)); %use weights as they were at the beginning of the trial 
            w_plot = reshape(w_plot,sqrt(N_pc),sqrt(N_pc));
            imagesc(w_plot')
            set(gca,'YDir','normal')
            colorbar
            title('Mean weights')
            
            %plot policy as a vector field
            subplot(2,2,4)
            f4=quiver(pc(:,1), pc(:,2), ac(1,:)', ac(2,:)', 'linewidth', 2, 'color', 'black');
            axis([-2 2 -2 2])
            title('Agent''s policy')

            drawnow
            %pause
            delete(f3)
            delete(f4)
        end
        %%
        t_end = T_max;
        
    end
    
end
