function [state_transition] = step0c_setup_state_transition

% for each state x action, this function creates gives rownumber of the 
% next period state it maps onto

globals=step0a_set_globals;
states=step0b_setup_states; 
Nstates=size(states,1);

state_transition=NaN(Nstates,globals.D);

for k=1:globals.D
    state_transition(1,k)=1+k;
end 

for i=Nstates:-1:2
    t=states(i,1);
    history=states(i,2:(t));
    lag_history=history(1:(end-1));
    lag_action=history(end);

    [row,col] = find(states(:,1)==t-1 & states(:,2:(t-1))==repmat(lag_history,Nstates,1));
    row=mode(row);

    if isnan(row)==0
        state_transition(row,lag_action)=i;
    end

end

end