function sounddata = GenerateSoundwave( imagedata, stepsize, frequency )
    if nargin < 3
        frequency = 10;
    end
    if nargin < 2
        stepsize = 1;
    end
    
    breaktime = 0.5;
    
    % load data image
    data = imread(imagedata);
    
    % duration in seconds
    duration = (stepsize + breaktime) * nnz(data);
    fs = 8192;
    t = 0:1/fs:duration;
    mask = sin(2*pi*frequency*t);
    sounddata(1,:) = mask;
    sounddata(2,:) = mask;
    sounddata(3,:) = mask;
    sounddata(4,:) = mask;
     
    current_time = 0;
    [current_y, current_x] = find(data==127);
    last_y = current_y;
    last_x = current_x;
    [step, new_x, new_y] = march(current_x, current_y, last_x, last_y, data);
    renew()
    while (step)
        data(current_y,current_x) = 200;
        ix1 = time2index(current_time, fs);
        current_time = current_time + stepsize;
        ix2 = time2index(current_time, fs);
        current_time = current_time + breaktime;
        switch step
            case 1  
                %go right
                sounddata(1,ix1:ix2) = 0;
                sounddata(3,ix1:ix2) = 0;
                sounddata(4,ix1:ix2) = 0;
            case 2
                %go left
                sounddata(1,ix1:ix2) = 0;
                sounddata(2,ix1:ix2) = 0;
                sounddata(4,ix1:ix2) = 0;
            case 3
                %go down
                sounddata(1,ix1:ix2) = 0;
                sounddata(2,ix1:ix2) = 0;
                sounddata(3,ix1:ix2) = 0;
            otherwise
                % go up
                sounddata(2,ix1:ix2) = 0;
                sounddata(3,ix1:ix2) = 0;
                sounddata(4,ix1:ix2) = 0;                
        end        
        [step, new_x, new_y] = march(current_x, current_y, last_x, last_y, data);
        renew()
    end
    % trim sound
    ix = time2index(current_time, fs);
    sounddata(:,ix:end) = [];
    % show path
    figure
    imagesc(data)
    
%     s = daq.createSession('directsound');
%     noutchan = 4;
%     addAudioOutputChannel(s, 'Audio5', 1:noutchan);
%     s.Rate = fs;
%     sounddata=sounddata';
%     queueOutputData(s, sounddata);
%     startBackground(s);
    function renew()
        last_x = current_x;
        last_y = current_y;
        current_x = new_x;
        current_y = new_y;
    end
end

function [step, new_x, new_y] = march(x, y, last_x, last_y, data)
    step = 0;
    new_x = x;
    new_y = y;
    % Move right
    if (255 == data(y, x+1) && (x+1)~= last_x)
        step = 1;
        new_x = x + 1;
        return
    end
    % Move left
    if (255 == data(y, x-1) && (x-1)~= last_x)
        step = 2;
        new_x = x - 1;
        return
    end
    % Move down
    if (255 == data(y+1, x) && (y+1)~= last_y)
        step = 3;
        new_y = y + 1;
        return
    end
    % Move up
    if (255 == data(y-1, x) && (y-1)~= last_y)
        step = 4;
        new_y = y - 1;
        return
    end    
end

function index = time2index(time, fs)
    index = round(time * fs + 1);
end