function [pi] = step0e_makepi(states,beliefparms)

    % this function creates the [1 x (Nstates-1)] vector that gives the
    % probability of a high-quality realisation of q in every state

    % NB. in this example beliefs are completely unrestricted. As Nstates
    % increases, it may be useful to put a functional form on beliefs.
    % Beliefs may be any function of information in the states matrix. This
    % is why this function takes the states matrix as an argument even
    % though it is unused in this particular example. 

    pi=[NaN, beliefparms];

end

