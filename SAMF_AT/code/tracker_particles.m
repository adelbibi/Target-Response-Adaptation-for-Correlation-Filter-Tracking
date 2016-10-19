function [y_exact] = tracker_particles(patch,kernel,cell_size, ...
    features,cos_window,model_alphaf,model_xf,lambda1,w2c)

zf = fft2(get_features(patch, features, cell_size, cos_window,w2c));
%calculate response of the classifier at all shifts
switch kernel.type
    case 'gaussian',
        kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
    case 'polynomial',
        kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
    case 'linear',
        kzf = linear_correlation(zf, model_xf);
end
response = real(ifft2(kzf .* model_alphaf /lambda1 ));  %equation for fast detection
y_exact = response(1,1); %This is the response for the exact patch extracted using previous filter
end

