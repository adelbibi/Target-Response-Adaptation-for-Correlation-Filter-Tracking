function [positions, rect_results, time] = tracker(video_path, img_files, pos, target_sz, ...
    padding, kernel, lambda1,lambda2,trans_samples_2D, output_sigma_factor, interp_factor, cell_size, ...
    features, show_visualization,scale_step,number_scales)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').
%     IMG_FILES is a cell array of image file names.
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).
%    TIME is the tracker execution time, without video loading/rendering.
%
%   Joao F. Henriques, 2014
%
%   revised by: Yang Li, August, 2014
%   http://ihpdep.github.io


addpath('./utility');
temp = load('w2crs');
w2c = temp.w2crs;
%if the target is large, lower the resolution, we don't need that much
%detail
resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
resize_image = 0;
if resize_image,
    pos = floor(pos / 2);
    target_sz = floor(target_sz / 2);
end
resize_image2 = (prod(target_sz) <= 750);  %diagonal size >= threshold
if resize_image2,
    pos = floor(pos * 2);
    target_sz = floor(target_sz * 2);
end

%window size, taking padding into account
sz = target_sz * (1 + padding); % square area, ignores the target aspect ratio
% set the size to exactly match the cell size
window_sz = round(sz / cell_size) * cell_size;

% window_sz = floor(target_sz * (1 + padding));
% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);


%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

%store pre-computed cosine window
cos_window = hann(size(yf,1)) * hann(size(yf,2))';
num_on_sides = (number_scales-1)/2;
top = (1+scale_step):scale_step:(1+num_on_sides*scale_step);
bottom = (1-num_on_sides*scale_step):scale_step:(1-scale_step);
search_size = [1 bottom top];
% search_size = [1  0.985 0.99 0.995 1.005 1.01 1.015];%

if show_visualization,  %create video interface
    update_visualization = show_video(img_files, video_path, resize_image);
end


%note: variables ending with 'f' are in the Fourier domain.

time = 0;  %to calculate FPS
positions = zeros(numel(img_files), 2);  %to calculate precision
rect_results = zeros(numel(img_files), 4);  %to calculate
response = zeros(size(cos_window,1),size(cos_window,2),size(search_size,2));
szid = 0;

for frame = 1:numel(img_files),
    %load image
    im = imread([video_path img_files{frame}]);
    % 		if size(im,3) > 1,
    % 			im = rgb2gray(im);
    % 		end
    if resize_image,
        im = imresize(im, 0.5);
    end
    if resize_image2,
        im = imresize(im, 2);
    end
    
    tic()
    if(frame ==1)
        %obtain a subwindow for training at newly estimated target position
        %patch = get_subwindow(im, pos, window_sz);
        target_sz = target_sz * search_size(szid+1);
        tmp_sz = floor((target_sz * (1 + padding)));
        param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
            tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
        param0 = affparam2mat(param0);
        patch = uint8(warpimg(double(im), param0, window_sz));
        xf = fft2(get_features(patch, features, cell_size, cos_window,w2c));
        
        %Kernel Ridge Regression, calculate alphas (in Fourier domain)
        switch kernel.type
            case 'gaussian',
                kf = gaussian_correlation(xf, xf, kernel.sigma);
            case 'polynomial',
                kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
            case 'linear',
                kf = linear_correlation(xf, xf);
        end
        alphaf = yf ./ (kf + lambda1);   %equation for fast training
        model_alphaf = alphaf;
        model_xf = xf;
        r_max = size(xf,1);
        c_max = size(xf,2);
        trans_plus_1 = (round(trans_samples_2D)/cell_size) + ones(size(trans_samples_2D,1),2);
        vert_delta = trans_plus_1(:,1);
        horiz_delta = trans_plus_1(:,2);
        vert_delta(vert_delta<=0) = vert_delta(vert_delta<=0) + r_max;
        horiz_delta(horiz_delta<=0) = horiz_delta(horiz_delta<=0) + c_max;
        y_o = zeros(r_max,c_max);
        lin_indx = sub2ind([r_max c_max],vert_delta, horiz_delta);
        num_trans = length(lin_indx);        
    else        
        %obtain a subwindow for detection at the position from last
        %frame, and convert to Fourier domain (its size is unchanged)
        %patch = get_subwindow(im, pos, window_sz);
        for i=1:size(search_size,2)
            tmp_sz = floor((target_sz * (1 + padding))*search_size(i));
            param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
                tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
            param0 = affparam2mat(param0);
            patch = uint8(warpimg(double(im), param0, window_sz));
            zf = fft2(get_features(patch, features, cell_size, cos_window,w2c));            
            switch kernel.type
                case 'gaussian',
                    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                case 'polynomial',
                    kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                case 'linear',
                    kzf = linear_correlation(zf, model_xf);
            end
            response(:,:,i) = real(ifft2(kzf .* model_alphaf /lambda1 ));  %equation for fast detection
        end
        %target location is at the maximum response. we must take into
        %account the fact that, if the target doesn't move, the peak
        %will appear at the top-left corner, not at the center (this is
        %discussed in the paper). the responses wrap around cyclically.
        [vert_delta,tmp, ~] = find(response == max(response(:)), 1);
        szid = floor(tmp/(size(cos_window,2)+1));
        horiz_delta = tmp - (szid * size(cos_window,2));
        if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
            vert_delta = vert_delta - size(zf,1);
        end
        if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
            horiz_delta = horiz_delta - size(zf,2);
        end
        tmp_sz = floor((target_sz * (1 + padding))*search_size(szid+1));
        current_size = tmp_sz(2)/window_sz(2);
        pos = pos + current_size*cell_size * [vert_delta - 1, horiz_delta - 1];
        %% Training
        %obtain a subwindow for training at newly estimated target position
        %patch = get_subwindow(im, pos, window_sz);
        target_sz = target_sz * search_size(szid+1);
        tmp_sz = floor((target_sz * (1 + padding)));
        param0 = [pos(2), pos(1), tmp_sz(2)/window_sz(2), 0,...
            tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
        param0 = affparam2mat(param0);
        patch = uint8(warpimg(double(im), param0, window_sz));
        xf = fft2(get_features(patch, features, cell_size, cos_window,w2c));        
        y_exact_particles = zeros(num_trans,1);
        pos(pos<=0) = 1;
        if(pos(1)>= size(im,1))
            pos(1) = size(im,1);
        end
        if(pos(2)>= size(im,2))
            pos(2) = size(im,2);
        end
        %% Particles' Scoring of Sampling
        for jj=1:num_trans
             pos_n = round(trans_samples_2D(jj,:)) + pos;
             param0 = [pos_n(2), pos_n(1), tmp_sz(2)/window_sz(2), 0,...
                 tmp_sz(1)/window_sz(2)/(window_sz(1)/window_sz(2)),0];
             param0 = affparam2mat(param0);
             patch = uint8(warpimg(double(im), param0, window_sz));
            [y_exact_particles(jj)] = tracker_particles(patch, kernel, cell_size, features,cos_window,model_alphaf,model_xf,lambda1,w2c);
        end
        y_o = zeros(r_max,c_max);
        y_o(lin_indx) = y_exact_particles';
        %% Generating new target y_o
        y_o = gaussian_shaped_multi_targets(output_sigma, y_o, floor(window_sz / cell_size));
        switch kernel.type
            case 'gaussian',
                kf = gaussian_correlation(xf, xf, kernel.sigma);
            case 'polynomial',
                kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
            case 'linear',
                kf = linear_correlation(xf, xf);
                kftwice = linear_correlationtwice(xf, xf);
        end
        rat_lamb = lambda2/lambda1;
        num = ((rat_lamb * kf) + lambda2).*fft2(y_o);
        den = (rat_lamb/lambda1)*kftwice  +  ((1+2*lambda2)/lambda1)*kf + (1 + lambda2);
        alphaf = ((num./den));
        
        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;       
     
    end
    
 
    
    
    %save position and timing
    positions(frame,:) = pos;
    time = time + toc();
  
    box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    rect_results(frame,:)=box;
    %visualization
    if show_visualization,
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early
        
        drawnow
        % 			pause(0.05)  %uncomment to run slower
    end
    
end

if resize_image,
    positions = positions * 2;
    rect_results = rect_results*2;
end
if resize_image2,
    positions = positions / 2;
    rect_results = rect_results/2;
end
end

