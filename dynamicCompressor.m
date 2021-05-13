function [outputAudio] = dynamicCompressor(filepath,transition1,transition2,thresholdBass,thresholdMids,thresholdTreble,ratioBass,ratioMids,ratioTreble,makeupGain)
% Purpose: Applies dynamic compresssion to an input file
%   This function relies on externally generated filters. Specifically,
%   this function will use windowed FIR impulse responses for the bass,
%   mids, and treble bands. Note that this only works on mono input audio.
%   If the input has multiple channels, the first (typically the left)
%   channel will be used.
% filepath: a string containing the path and file name for the input audio
%     file
% transition1: cutoff frequency (in Hz) for the first transition between
%     the low frequencies and the mid frequencies
% transition2: cutoff frequency (in Hz) for the second transition between
%     the mid frequencies and the treble frequencies
% thresholdBass: bass threshold that activates the compressor if passed
% thresholdMids: mids threshold that activates the compressor if passed
% thresholdTreble: treble threshold that actives the compressor if passed
% ratioBass: The strength of the compressor in the bass range
% ratioMids: The strength of the compressor in the mids range
% ratioTreble: The strength of the compressor in the treble range
% makeupGain: The amount of gain to apply to the signal after processing
    
    % Reads the input file and gets the sampling frequency
    [y,fs] = audioread(filepath);
    
    N = 1025; % filter length, 2M+1 from filter functions
    
    segmentSize = N; % The amount of samples to process at a time.
                     % This must be greater than the filter size, N or
                     % else the output will have transients during every
                     % sample
                        
    % Converting thresholds into magnitude instead of dB
    thresholdBass_A = db2mag(thresholdBass);
    thresholdMids_A = db2mag(thresholdMids);
    thresholdTreble_A = db2mag(thresholdTreble);

    % If the user has the parallel computing toolbox and a CUDA enabled
    % graphics card, store arrays on the GPU and do matrix math using the
    % GPU cores instead of the CPU cores.
    try
        gpu = gpuDevice; % This will throw an exception if the user does
                         % not have the toolbox and a CUDA GPU installed.

        % throw("") % used for debugging purposes to enter the catch block
        fprintf("GPU acceleration enabled\n")
        y = gpuArray(y); % Store input audio in the GPU's memory
        
        % Generate the three filters for dividing the input signal into
        % lows (bass), mids, and highs (treble).
        lowPass = gpuArray(bassBand(fs, transition1, 1));
        bandPass = gpuArray(midBand(fs, transition1, transition2, 1));
        highPass = gpuArray(trebleBand(fs, transition2, 1));
        
        % Initialize the output array that will hold the data after
        % compression.
        y_filtered = gpuArray(zeros(floor(size(y,1)/segmentSize)* ... 
            segmentSize-(2*segmentSize),1));
        
    catch % This will execute if the program cannot be GPU accelerated
        fprintf("GPU acceleration disabled\n")
        
        % Generate the three filters for dividing the input signal into
        % lows (bass), mids, and highs (treble).
        lowPass = bassBand(fs, transition1, 1);
        bandPass = midBand(fs, transition1, transition2, 1);
        highPass = trebleBand(fs, transition2, 1);
        
        % Initialize the output array that will hold the data after
        % compression.
        y_filtered = zeros(floor(size(y,1)/segmentSize)*segmentSize-(2*segmentSize),1);
    end

    % Iterates through the input audio sequence in blocks of size 
    % segmentSize, skipping the first and last blocks in order to prevent
    % the output signal from having transients. This is usually acceptable
    % since only a small fraction of the signal is lost.
    for i = 2:floor(size(y,1)/segmentSize)-1 
    
        % Get the low frequency component of this section by convolving a
        % sequence containing the current segment and the last segment of
        % the input signal with the lowpass filter. We use an old segment
        % in this convolution to ensure that the output contains no
        % transients.
        lowSegment = conv(lowPass,y((i-2)*segmentSize+1:i*segmentSize,1));
        
        % The segment is trimmed to remove the previous segment and the
        % transients at the beginning and end of the convolution.
        lowSegment = lowSegment(segmentSize:end-N);
        
        % The maximum magnitude in this segment is calculated to see if
        % there should be any attenuation applied to it.
        maxLow = max(abs(lowSegment));
        
        % If the maximum amplitude in this segment passes the user defined
        % threshold amplitude for the bass band then the signal should be
        % attenuated.
        if maxLow > thresholdBass_A
            
            % Convert the maximum amplitude into decibels
            maxLow_dB = mag2db(maxLow);
            
            % Calculate target amplitude in decibels
            new_dB = ((maxLow_dB - thresholdBass) / ratioBass) + ...
                thresholdBass;
            
            % Convert target amplitude from decibels to magnitude
            target_A = db2mag(new_dB);
            
            % Scale the segment's amplitude to the target amplitude
            lowSegment = lowSegment .* (target_A / maxLow);
            
        end
        
        % Get the middle frequency component of this section by convolving
        % a sequence containing the current segment and the last segment of
        % the input signal with the bandpass filter. We use an old segment
        % in this convolution to ensure that the output contains no
        % transients.
        midSegment = conv(bandPass,y((i-2)*segmentSize+1:i*segmentSize,1));
        
        % The segment is trimmed to remove the previous segment and the
        % transients at the beginning and end of the convolution.
        midSegment = midSegment(segmentSize:end-N);
        
        % The maximum magnitude in this segment is calculated to see if
        % there should be any attenuation applied to it.
        maxMid = max(abs(midSegment));
        
        % If the maximum amplitude in this segment passes the user defined
        % threshold amplitude for the mid band then the signal should be
        % attenuated.
        if maxMid > thresholdMids_A
            
            % Convert the maximum amplitude into decibels
            maxMid_dB = mag2db(maxMid);
            
            % Calculate target amplitude in decibels
            new_dB = ((maxMid_dB - thresholdMids) / ratioMids) + ...
                thresholdMids;
            
            % Convert target amplitude from decibels to magnitude
            target_A = db2mag(new_dB);
            
            % Scale the segment's amplitude to the target amplitude
            midSegment = midSegment .* (target_A / maxMid);
        end
        
        % Get the high frequency component of this section by convolving a
        % sequence containing the current segment and the last segment of
        % the input signal with the highpass filter. We use an old segment
        % in this convolution to ensure that the output contains no
        % transients.
        highSegment = conv(highPass,y((i-2)*segmentSize+1:i*segmentSize,1));
        
        % The segment is trimmed to remove the previous segment and the
        % transients at the beginning and end of the convolution.
        highSegment = highSegment(segmentSize:end-N);
        
        % The maximum magnitude in this segment is calculated to see if
        % there should be any attenuation applied to it.
        maxHigh = max(abs(highSegment));
        
        % If the maximum amplitude in this segment passes the user defined
        % threshold amplitude for the treble band then the signal should be
        % attenuated.
        if maxHigh > thresholdTreble_A
            
            % Convert the maximum amplitude into decibels
            maxHigh_dB = mag2db(maxHigh);
            
            % Calculate target amplitude in decibels
            new_dB = ((maxHigh_dB - thresholdTreble) / ratioTreble) + ...
                thresholdTreble;
            
            % Convert target amplitude from decibels to magnitude
            target_A = db2mag(new_dB);
            
            % Scale the segment's amplitude to the target amplitude
            highSegment = highSegment .* (target_A / maxHigh);
        end
        
        % Finally add the three segments together to make an output segment
        y_filtered((i-2)*segmentSize+1:(i-1)*segmentSize) = lowSegment ...
            + midSegment + highSegment;
                                             
    end
    
    % Apply the overall output gain to the output signal. This is needed
    % because compression only attenuates signals which would make the
    % overall output signal quieter than the input. This gain (typically
    % called "makeup gain") simply raises the output volume.
    y_filtered = y_filtered .* db2mag(makeupGain);

    % If the math so far has been done on the GPU, we can now convert it
    % back to a CPU array.
    try
        % This statement will fail if the array is not on the GPU.
        outputAudio = gather(y_filtered);
    catch
        outputAudio = y_filtered;
    end
    
    % Break down the path to the file into three strings
    [path,name,ext] = fileparts(filepath);
    
    % Write the output into a file with the same name as the input file
    % suffixed with _compressed
    audiowrite(path+"\"+name+"_compressed"+ext, outputAudio,fs);
    
end