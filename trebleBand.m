function impulse_response = trebleBand(fs, fc, A)

    M = 512; % Approximately half the length of the filter

    n = -M:M;
    
    % Impulse response using MATLAB's sinc function, normalized by
    % (2*fc/fs) since MATLAB uses a different formula for sinc.
    h_n = A .* (sinc(2.*n) - (sinc(2*fc.*n./fs) .* (2*fc/fs)));
    
    % Window function for Blackman window
    w = 0.5 + 0.5.*cos((pi*n)./(M));
    
    % Window function for Hamming window
    % w = 0.54 + 0.46.*cos((pi*n)./(M)); % Not going to use this
    
    impulse_response = h_n .* w;
    % impulse_response = h_n; % uncomment to do rectangular windowing

end