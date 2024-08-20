function [globals] = step0a_set_globals
    % T= number of time periods
    globals.T=3;
    
    % D = number of non-terminal actions that can be taken each period. 
    %               The terminal action is always available.
    globals.D=2; 
    
    % disc = discount factor
    globals.disc=0.95; 
    
    % Nsims = number of simulations used in evaluating the multivariate decision problem. 
    globals.Nsims=100000; 
end

